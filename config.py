import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for ML models and embeddings"""
    bert_model: str = "bert-base-uncased"
    sentence_transformer: str = "all-MiniLM-L6-v2"
    t5_model: str = "t5-small"
    embedding_dim: int = 384
    max_sequence_length: int = 512

@dataclass
class RAGConfig:
    """Configuration for RAG pipeline"""
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.7
    max_context_length: int = 2000

@dataclass
class AWSConfig:
    """AWS configuration"""
    region: str = os.getenv("AWS_REGION", "us-east-1")
    s3_bucket: str = os.getenv("S3_BUCKET", "medical-rag-bucket")
    dynamodb_table: str = os.getenv("DYNAMODB_TABLE", "medical-conversations")
    lambda_function: str = os.getenv("LAMBDA_FUNCTION", "medical-rag-api")

@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    max_concurrent_requests: int = 100
    request_timeout: int = 30

# Global configuration instance
CONFIG = {
    "model": ModelConfig(),
    "rag": RAGConfig(),
    "aws": AWSConfig(),
    "api": APIConfig()
}