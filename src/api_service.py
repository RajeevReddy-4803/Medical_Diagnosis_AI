from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import asyncio
import uvicorn
from datetime import datetime
import uuid
from .rag_pipeline import MedicalRAGPipeline
from config import CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Medical RAG Conversational Search API",
    description="AI-powered medical conversational search using RAG pipeline",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
rag_pipeline = None

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    use_hybrid: bool = True
    include_history: bool = True
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    confidence: float
    sources: List[str]
    retrieved_documents: int
    conversation_id: int
    session_id: str
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    pipeline_initialized: bool
    knowledge_base_size: int
    timestamp: datetime

class ConversationHistory(BaseModel):
    session_id: str
    history: List[Dict[str, Any]]

# Session management
sessions = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline on startup"""
    global rag_pipeline
    logger.info("Starting Medical RAG API service...")
    
    try:
        rag_pipeline = MedicalRAGPipeline()
        rag_pipeline.initialize_pipeline()
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Medical RAG API service...")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if rag_pipeline else "unhealthy",
        pipeline_initialized=rag_pipeline is not None,
        knowledge_base_size=len(rag_pipeline.knowledge_base) if rag_pipeline else 0,
        timestamp=datetime.now()
    )

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a conversational query"""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Get or create session
        if session_id not in sessions:
            sessions[session_id] = {
                'created_at': datetime.now(),
                'query_count': 0,
                'pipeline': MedicalRAGPipeline()
            }
            sessions[session_id]['pipeline'].initialize_pipeline()
        
        session = sessions[session_id]
        session['query_count'] += 1
        
        # Process query
        result = session['pipeline'].process_query(
            request.query,
            use_hybrid=request.use_hybrid,
            include_history=request.include_history
        )
        
        return QueryResponse(
            response=result['response'],
            confidence=result['confidence'],
            sources=result['sources'],
            retrieved_documents=len(result['retrieved_documents']),
            conversation_id=result['conversation_id'],
            session_id=session_id,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/conversation/{session_id}", response_model=ConversationHistory)
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    history = sessions[session_id]['pipeline'].get_conversation_history()
    
    return ConversationHistory(
        session_id=session_id,
        history=history
    )

@app.delete("/conversation/{session_id}")
async def clear_conversation_history(session_id: str):
    """Clear conversation history for a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    sessions[session_id]['pipeline'].clear_conversation_history()
    
    return {"message": "Conversation history cleared", "session_id": session_id}

@app.get("/sessions")
async def list_sessions():
    """List active sessions"""
    session_info = []
    for session_id, session_data in sessions.items():
        session_info.append({
            'session_id': session_id,
            'created_at': session_data['created_at'],
            'query_count': session_data['query_count']
        })
    
    return {"sessions": session_info, "total_sessions": len(sessions)}

@app.post("/fine-tune")
async def fine_tune_models(background_tasks: BackgroundTasks):
    """Fine-tune models on medical data"""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    background_tasks.add_task(rag_pipeline.fine_tune_models)
    
    return {"message": "Fine-tuning started in background"}

@app.get("/knowledge-base/stats")
async def get_knowledge_base_stats():
    """Get knowledge base statistics"""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    kb = rag_pipeline.knowledge_base
    
    # Analyze knowledge base
    disease_counts = {}
    type_counts = {}
    
    for doc in kb:
        disease = doc.get('disease', 'unknown')
        doc_type = doc.get('type', 'unknown')
        
        disease_counts[disease] = disease_counts.get(disease, 0) + 1
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
    
    return {
        "total_documents": len(kb),
        "diseases": disease_counts,
        "document_types": type_counts,
        "last_updated": datetime.now()
    }

@app.post("/search")
async def search_knowledge_base(query: str, top_k: int = 5):
    """Search knowledge base directly"""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        results = rag_pipeline.embedding_manager.similarity_search(
            query, top_k=top_k, threshold=0.5
        )
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "document_id": doc['id'],
                "disease": doc['disease'],
                "type": doc['type'],
                "content": doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                "similarity_score": score,
                "metadata": doc.get('metadata', {})
            })
        
        return {
            "query": query,
            "results": formatted_results,
            "total_results": len(formatted_results)
        }
        
    except Exception as e:
        logger.error(f"Error searching knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run(
        "src.api_service:app",
        host=CONFIG['api'].host,
        port=CONFIG['api'].port,
        reload=True,
        log_level="info"
    )