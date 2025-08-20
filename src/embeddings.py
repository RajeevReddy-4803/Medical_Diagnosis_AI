import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration
from typing import List, Dict, Any, Optional, Tuple
import logging
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class MedicalEmbeddingManager:
    """Manages embeddings for medical documents using BERT and Sentence Transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", bert_model: str = "bert-base-uncased"):
        self.sentence_model = SentenceTransformer(model_name)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.bert_model = AutoModel.from_pretrained(bert_model)
        self.embeddings_cache = {}
        self.document_embeddings = None
        self.documents = []
        
    def encode_documents(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """Encode documents into embeddings"""
        self.documents = documents
        texts = [doc['content'] for doc in documents]
        
        logger.info(f"Encoding {len(texts)} documents...")
        embeddings = self.sentence_model.encode(texts, show_progress_bar=True)
        self.document_embeddings = embeddings
        
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query into embedding"""
        return self.sentence_model.encode([query])[0]
    
    def get_bert_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Get BERT embeddings for texts"""
        inputs = self.bert_tokenizer(texts, padding=True, truncation=True, 
                                   return_tensors="pt", max_length=512)
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
            
        return embeddings
    
    def similarity_search(self, query: str, top_k: int = 5, 
                         threshold: float = 0.7) -> List[Tuple[Dict[str, Any], float]]:
        """Find most similar documents to query"""
        if self.document_embeddings is None:
            raise ValueError("Documents not encoded yet. Call encode_documents first.")
        
        query_embedding = self.encode_query(query)
        similarities = cosine_similarity([query_embedding], self.document_embeddings)[0]
        
        # Get top-k results above threshold
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        
        for idx in top_indices:
            if similarities[idx] >= threshold:
                results.append((self.documents[idx], float(similarities[idx])))
        
        return results
    
    def save_embeddings(self, filepath: str):
        """Save embeddings to file"""
        data = {
            'embeddings': self.document_embeddings,
            'documents': self.documents
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath: str):
        """Load embeddings from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.document_embeddings = data['embeddings']
        self.documents = data['documents']
        logger.info(f"Embeddings loaded from {filepath}")

class T5ResponseGenerator:
    """Generate responses using T5 model"""
    
    def __init__(self, model_name: str = "t5-small"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
    def generate_response(self, context: str, question: str, max_length: int = 512) -> str:
        """Generate response based on context and question"""
        # Format input for T5
        input_text = f"question: {question} context: {context}"
        
        inputs = self.tokenizer.encode(input_text, return_tensors="pt", 
                                     max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                temperature=0.7,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def fine_tune_on_medical_data(self, training_data: List[Dict[str, Any]], 
                                 epochs: int = 3, batch_size: int = 8):
        """Fine-tune T5 on medical data"""
        from torch.utils.data import DataLoader, Dataset
        from transformers import AdamW
        
        class MedicalDataset(Dataset):
            def __init__(self, data, tokenizer, max_length=512):
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                input_text = f"question: {item['question']}"
                target_text = item['answer']
                
                input_encoding = self.tokenizer(
                    input_text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                target_encoding = self.tokenizer(
                    target_text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': input_encoding['input_ids'].flatten(),
                    'attention_mask': input_encoding['attention_mask'].flatten(),
                    'labels': target_encoding['input_ids'].flatten()
                }
        
        dataset = MedicalDataset(training_data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        logger.info("Fine-tuning completed")

class HybridRetriever:
    """Hybrid retrieval combining dense and sparse methods"""
    
    def __init__(self, embedding_manager: MedicalEmbeddingManager):
        self.embedding_manager = embedding_manager
        self.keyword_weights = {}
        
    def build_keyword_index(self, documents: List[Dict[str, Any]]):
        """Build keyword index for sparse retrieval"""
        from collections import Counter
        import re
        
        for doc in documents:
            words = re.findall(r'\w+', doc['content'].lower())
            word_counts = Counter(words)
            
            doc_id = doc['id']
            self.keyword_weights[doc_id] = word_counts
    
    def sparse_retrieval(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Sparse retrieval based on keyword matching"""
        query_words = set(re.findall(r'\w+', query.lower()))
        scores = {}
        
        for doc_id, word_counts in self.keyword_weights.items():
            score = 0
            for word in query_words:
                if word in word_counts:
                    score += word_counts[word]
            scores[doc_id] = score
        
        # Sort by score and return top-k
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]
    
    def hybrid_search(self, query: str, top_k: int = 5, 
                     dense_weight: float = 0.7, sparse_weight: float = 0.3) -> List[Tuple[Dict[str, Any], float]]:
        """Combine dense and sparse retrieval"""
        # Dense retrieval
        dense_results = self.embedding_manager.similarity_search(query, top_k=top_k * 2)
        
        # Sparse retrieval
        sparse_results = self.sparse_retrieval(query, top_k=top_k * 2)
        sparse_dict = {doc_id: score for doc_id, score in sparse_results}
        
        # Combine scores
        combined_scores = {}
        for doc, dense_score in dense_results:
            doc_id = doc['id']
            sparse_score = sparse_dict.get(doc_id, 0)
            
            # Normalize sparse score
            max_sparse = max(sparse_dict.values()) if sparse_dict else 1
            normalized_sparse = sparse_score / max_sparse if max_sparse > 0 else 0
            
            combined_score = (dense_weight * dense_score + 
                            sparse_weight * normalized_sparse)
            combined_scores[doc_id] = (doc, combined_score)
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.values(), 
                              key=lambda x: x[1], reverse=True)
        
        return sorted_results[:top_k]