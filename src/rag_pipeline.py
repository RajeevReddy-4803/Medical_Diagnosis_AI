from typing import List, Dict, Any, Optional, Tuple
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import numpy as np
from .embeddings import MedicalEmbeddingManager, T5ResponseGenerator, HybridRetriever
from .data_processor import MedicalDataProcessor
from config import CONFIG

logger = logging.getLogger(__name__)

class MedicalRAGPipeline:
    """RAG pipeline for medical conversational search"""
    
    def __init__(self):
        self.embedding_manager = MedicalEmbeddingManager()
        self.t5_generator = T5ResponseGenerator()
        self.hybrid_retriever = HybridRetriever(self.embedding_manager)
        self.data_processor = MedicalDataProcessor()
        self.knowledge_base = []
        self.conversation_history = []
        
        # LangChain components
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG['rag'].chunk_size,
            chunk_overlap=CONFIG['rag'].chunk_overlap
        )
        self.vectorstore = None
        self.qa_chain = None
        
    def initialize_pipeline(self):
        """Initialize the RAG pipeline"""
        logger.info("Initializing Medical RAG Pipeline...")
        
        # Load and process data
        self.data_processor.load_datasets()
        self.knowledge_base = self.data_processor.create_medical_knowledge_base()
        
        # Create embeddings
        self.embedding_manager.encode_documents(self.knowledge_base)
        
        # Build hybrid retriever
        self.hybrid_retriever.build_keyword_index(self.knowledge_base)
        
        # Setup LangChain components
        self._setup_langchain_components()
        
        logger.info("Pipeline initialization completed")
    
    def _setup_langchain_components(self):
        """Setup LangChain components for RAG"""
        # Create documents for LangChain
        documents = []
        for doc in self.knowledge_base:
            documents.append(Document(
                page_content=doc['content'],
                metadata=doc['metadata']
            ))
        
        # Split documents
        split_docs = self.text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(
            model_name=CONFIG['model'].sentence_transformer
        )
        
        self.vectorstore = FAISS.from_documents(split_docs, embeddings)
        
        # Create custom prompt template
        prompt_template = """
        You are a medical AI assistant. Use the following context to answer the question about medical conditions, symptoms, and health information.
        
        Context: {context}
        
        Question: {question}
        
        Provide a helpful, accurate, and informative answer based on the context. If you're not sure about something, say so.
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self._create_llm(),
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": CONFIG['rag'].top_k_retrieval}
            ),
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def _create_llm(self):
        """Create LLM for LangChain"""
        from transformers import pipeline
        
        # Use T5 for text generation
        pipe = pipeline(
            "text2text-generation",
            model="t5-small",
            max_length=512,
            temperature=0.7,
            do_sample=True
        )
        
        return HuggingFacePipeline(pipeline=pipe)
    
    def process_query(self, query: str, use_hybrid: bool = True, 
                     include_history: bool = True) -> Dict[str, Any]:
        """Process a conversational query"""
        logger.info(f"Processing query: {query}")
        
        # Add conversation context if requested
        if include_history and self.conversation_history:
            context_query = self._add_conversation_context(query)
        else:
            context_query = query
        
        # Retrieve relevant documents
        if use_hybrid:
            retrieved_docs = self.hybrid_retriever.hybrid_search(
                context_query, 
                top_k=CONFIG['rag'].top_k_retrieval
            )
        else:
            retrieved_docs = self.embedding_manager.similarity_search(
                context_query,
                top_k=CONFIG['rag'].top_k_retrieval,
                threshold=CONFIG['rag'].similarity_threshold
            )
        
        # Generate response using multiple methods
        responses = {}
        
        # Method 1: T5 Generation
        if retrieved_docs:
            context = self._build_context(retrieved_docs)
            responses['t5_response'] = self.t5_generator.generate_response(
                context, query
            )
        
        # Method 2: LangChain QA
        if self.qa_chain:
            responses['langchain_response'] = self.qa_chain.run(query)
        
        # Method 3: Template-based response
        responses['template_response'] = self._generate_template_response(
            query, retrieved_docs
        )
        
        # Select best response (you can implement more sophisticated selection)
        final_response = self._select_best_response(responses, retrieved_docs)
        
        # Update conversation history
        self.conversation_history.append({
            'query': query,
            'response': final_response,
            'retrieved_docs': len(retrieved_docs),
            'confidence': self._calculate_confidence(retrieved_docs)
        })
        
        return {
            'response': final_response,
            'retrieved_documents': retrieved_docs,
            'confidence': self._calculate_confidence(retrieved_docs),
            'sources': [doc[0]['id'] for doc in retrieved_docs],
            'conversation_id': len(self.conversation_history)
        }
    
    def _add_conversation_context(self, query: str) -> str:
        """Add conversation history context to query"""
        if not self.conversation_history:
            return query
        
        # Get last few exchanges
        recent_history = self.conversation_history[-3:]
        context_parts = []
        
        for exchange in recent_history:
            context_parts.append(f"Previous Q: {exchange['query']}")
            context_parts.append(f"Previous A: {exchange['response']}")
        
        context = " ".join(context_parts)
        return f"Context: {context} Current question: {query}"
    
    def _build_context(self, retrieved_docs: List[Tuple[Dict[str, Any], float]]) -> str:
        """Build context from retrieved documents"""
        context_parts = []
        max_length = CONFIG['rag'].max_context_length
        current_length = 0
        
        for doc, score in retrieved_docs:
            content = doc['content']
            if current_length + len(content) <= max_length:
                context_parts.append(content)
                current_length += len(content)
            else:
                # Truncate to fit
                remaining = max_length - current_length
                if remaining > 100:  # Only add if meaningful content can fit
                    context_parts.append(content[:remaining])
                break
        
        return " ".join(context_parts)
    
    def _generate_template_response(self, query: str, 
                                  retrieved_docs: List[Tuple[Dict[str, Any], float]]) -> str:
        """Generate response using templates"""
        if not retrieved_docs:
            return "I don't have enough information to answer your question. Could you please provide more details or ask about a specific medical condition?"
        
        # Analyze query type
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['symptom', 'sign', 'feel']):
            return self._generate_symptom_response(retrieved_docs)
        elif any(word in query_lower for word in ['risk', 'cause', 'factor']):
            return self._generate_risk_factor_response(retrieved_docs)
        elif any(word in query_lower for word in ['what is', 'define', 'explain']):
            return self._generate_definition_response(retrieved_docs)
        elif any(word in query_lower for word in ['statistic', 'data', 'common']):
            return self._generate_statistics_response(retrieved_docs)
        else:
            # General response
            best_doc = retrieved_docs[0][0]
            return f"Based on the available information: {best_doc['content']}"
    
    def _generate_symptom_response(self, retrieved_docs: List[Tuple[Dict[str, Any], float]]) -> str:
        """Generate symptom-focused response"""
        symptoms = []
        diseases = set()
        
        for doc, score in retrieved_docs:
            if 'symptoms' in doc.get('metadata', {}):
                symptoms.extend(doc['metadata']['symptoms'])
                diseases.add(doc['disease'])
        
        if symptoms:
            unique_symptoms = list(set(symptoms))
            disease_list = ", ".join(diseases)
            return f"Common symptoms for {disease_list} include: {', '.join(unique_symptoms[:5])}. If you're experiencing these symptoms, please consult with a healthcare professional for proper evaluation."
        
        return retrieved_docs[0][0]['content'] if retrieved_docs else "No symptom information available."
    
    def _generate_risk_factor_response(self, retrieved_docs: List[Tuple[Dict[str, Any], float]]) -> str:
        """Generate risk factor response"""
        risk_factors = []
        diseases = set()
        
        for doc, score in retrieved_docs:
            if 'risk_factors' in doc.get('metadata', {}):
                risk_factors.extend(doc['metadata']['risk_factors'])
                diseases.add(doc['disease'])
        
        if risk_factors:
            unique_factors = list(set(risk_factors))
            disease_list = ", ".join(diseases)
            return f"Risk factors for {disease_list} include: {', '.join(unique_factors)}. Understanding these risk factors can help in prevention and early detection."
        
        return retrieved_docs[0][0]['content'] if retrieved_docs else "No risk factor information available."
    
    def _generate_definition_response(self, retrieved_docs: List[Tuple[Dict[str, Any], float]]) -> str:
        """Generate definition response"""
        for doc, score in retrieved_docs:
            if doc['type'] == 'overview':
                return doc['content']
        
        return retrieved_docs[0][0]['content'] if retrieved_docs else "No definition available."
    
    def _generate_statistics_response(self, retrieved_docs: List[Tuple[Dict[str, Any], float]]) -> str:
        """Generate statistics response"""
        for doc, score in retrieved_docs:
            if doc['type'] == 'statistics':
                return doc['content']
        
        return "Statistical information is limited. Please consult medical literature or healthcare professionals for detailed statistics."
    
    def _select_best_response(self, responses: Dict[str, str], 
                            retrieved_docs: List[Tuple[Dict[str, Any], float]]) -> str:
        """Select the best response from multiple generation methods"""
        # Simple selection logic - can be improved with more sophisticated methods
        if 'langchain_response' in responses and responses['langchain_response']:
            return responses['langchain_response']
        elif 'template_response' in responses and responses['template_response']:
            return responses['template_response']
        elif 't5_response' in responses and responses['t5_response']:
            return responses['t5_response']
        else:
            return "I'm sorry, I couldn't generate a proper response. Please try rephrasing your question."
    
    def _calculate_confidence(self, retrieved_docs: List[Tuple[Dict[str, Any], float]]) -> float:
        """Calculate confidence score for the response"""
        if not retrieved_docs:
            return 0.0
        
        # Average similarity score
        avg_score = np.mean([score for _, score in retrieved_docs])
        
        # Adjust based on number of retrieved documents
        doc_count_factor = min(len(retrieved_docs) / CONFIG['rag'].top_k_retrieval, 1.0)
        
        return float(avg_score * doc_count_factor)
    
    def fine_tune_models(self):
        """Fine-tune models on medical data"""
        logger.info("Starting model fine-tuning...")
        
        # Prepare training data
        training_data = self.data_processor.prepare_training_data()
        
        # Fine-tune T5
        self.t5_generator.fine_tune_on_medical_data(training_data)
        
        logger.info("Model fine-tuning completed")
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversation_history
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def save_pipeline_state(self, filepath: str):
        """Save pipeline state"""
        import pickle
        
        state = {
            'knowledge_base': self.knowledge_base,
            'conversation_history': self.conversation_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        # Save embeddings separately
        self.embedding_manager.save_embeddings(f"{filepath}_embeddings.pkl")
        
        logger.info(f"Pipeline state saved to {filepath}")
    
    def load_pipeline_state(self, filepath: str):
        """Load pipeline state"""
        import pickle
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.knowledge_base = state['knowledge_base']
        self.conversation_history = state['conversation_history']
        
        # Load embeddings
        self.embedding_manager.load_embeddings(f"{filepath}_embeddings.pkl")
        
        logger.info(f"Pipeline state loaded from {filepath}")