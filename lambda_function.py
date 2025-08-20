"""
AWS Lambda function for Medical RAG Conversational Search
"""
import json
import logging
import os
import sys
from typing import Dict, Any

# Add src to path
sys.path.append('/opt/python/lib/python3.9/site-packages')
sys.path.append('.')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global variables for Lambda optimization
rag_pipeline = None
aws_integration = None

def initialize_components():
    """Initialize components (called once per Lambda container)"""
    global rag_pipeline, aws_integration
    
    try:
        from src.rag_pipeline import MedicalRAGPipeline
        from src.aws_integration import AWSIntegration
        
        logger.info("Initializing RAG pipeline...")
        rag_pipeline = MedicalRAGPipeline()
        rag_pipeline.initialize_pipeline()
        
        logger.info("Initializing AWS integration...")
        aws_integration = AWSIntegration()
        
        logger.info("Components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    AWS Lambda handler for medical conversational search
    
    Expected event format:
    {
        "httpMethod": "POST",
        "body": "{\"query\": \"What are symptoms of diabetes?\", \"session_id\": \"optional\"}"
    }
    """
    global rag_pipeline, aws_integration
    
    try:
        # Initialize components if not already done
        if not rag_pipeline or not aws_integration:
            initialize_components()
        
        # Parse HTTP request
        http_method = event.get('httpMethod', 'POST')
        path = event.get('path', '/')
        
        if http_method == 'GET' and path == '/health':
            return handle_health_check()
        
        if http_method != 'POST':
            return create_response(405, {'error': 'Method not allowed'})
        
        # Parse request body
        body_str = event.get('body', '{}')
        if isinstance(body_str, str):
            body = json.loads(body_str)
        else:
            body = body_str
        
        # Route to appropriate handler
        if path == '/query' or path == '/':
            return handle_query(body)
        elif path == '/search':
            return handle_search(body)
        else:
            return create_response(404, {'error': 'Endpoint not found'})
            
    except json.JSONDecodeError:
        logger.error("Invalid JSON in request body")
        return create_response(400, {'error': 'Invalid JSON'})
    
    except Exception as e:
        logger.error(f"Lambda handler error: {str(e)}")
        return create_response(500, {'error': 'Internal server error'})

def handle_health_check() -> Dict[str, Any]:
    """Handle health check requests"""
    return create_response(200, {
        'status': 'healthy',
        'pipeline_initialized': rag_pipeline is not None,
        'timestamp': str(datetime.now())
    })

def handle_query(body: Dict[str, Any]) -> Dict[str, Any]:
    """Handle conversational query requests"""
    global rag_pipeline, aws_integration
    
    try:
        # Extract parameters
        query = body.get('query', '').strip()
        session_id = body.get('session_id', str(uuid.uuid4()))
        use_hybrid = body.get('use_hybrid', True)
        include_history = body.get('include_history', True)
        
        if not query:
            return create_response(400, {'error': 'Query is required'})
        
        logger.info(f"Processing query: {query[:100]}...")
        
        # Process query using RAG pipeline
        result = rag_pipeline.process_query(
            query=query,
            use_hybrid=use_hybrid,
            include_history=include_history
        )
        
        # Store conversation in DynamoDB
        conversation_stored = aws_integration.store_conversation(
            session_id=session_id,
            query=query,
            response=result['response'],
            metadata={
                'confidence': result['confidence'],
                'sources': result['sources'],
                'retrieved_documents': len(result['retrieved_documents']),
                'use_hybrid': use_hybrid,
                'include_history': include_history
            }
        )
        
        # Log metrics to CloudWatch
        aws_integration.log_metrics('QueryProcessed', 1)
        aws_integration.log_metrics('Confidence', result['confidence'], 'None')
        aws_integration.log_metrics('RetrievedDocuments', len(result['retrieved_documents']))
        
        # Prepare response
        response_data = {
            'response': result['response'],
            'confidence': result['confidence'],
            'sources': result['sources'],
            'retrieved_documents': len(result['retrieved_documents']),
            'conversation_id': result['conversation_id'],
            'session_id': session_id,
            'conversation_stored': conversation_stored
        }
        
        logger.info(f"Query processed successfully. Confidence: {result['confidence']:.3f}")
        
        return create_response(200, response_data)
        
    except Exception as e:
        logger.error(f"Error handling query: {str(e)}")
        aws_integration.log_metrics('QueryError', 1)
        return create_response(500, {'error': f'Query processing failed: {str(e)}'})

def handle_search(body: Dict[str, Any]) -> Dict[str, Any]:
    """Handle knowledge base search requests"""
    global rag_pipeline
    
    try:
        query = body.get('query', '').strip()
        top_k = body.get('top_k', 5)
        threshold = body.get('threshold', 0.5)
        
        if not query:
            return create_response(400, {'error': 'Query is required'})
        
        logger.info(f"Searching knowledge base: {query[:100]}...")
        
        # Search knowledge base
        results = rag_pipeline.embedding_manager.similarity_search(
            query=query,
            top_k=top_k,
            threshold=threshold
        )
        
        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'document_id': doc['id'],
                'disease': doc['disease'],
                'type': doc['type'],
                'content': doc['content'][:300] + '...' if len(doc['content']) > 300 else doc['content'],
                'similarity_score': float(score),
                'metadata': doc.get('metadata', {})
            })
        
        response_data = {
            'query': query,
            'results': formatted_results,
            'total_results': len(formatted_results),
            'parameters': {
                'top_k': top_k,
                'threshold': threshold
            }
        }
        
        logger.info(f"Search completed. Found {len(formatted_results)} results")
        
        return create_response(200, response_data)
        
    except Exception as e:
        logger.error(f"Error handling search: {str(e)}")
        return create_response(500, {'error': f'Search failed: {str(e)}'})

def create_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    """Create standardized HTTP response"""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization'
        },
        'body': json.dumps(body, default=str)
    }

# Import required modules at module level for Lambda optimization
try:
    import uuid
    from datetime import datetime
    logger.info("Lambda function loaded successfully")
except ImportError as e:
    logger.error(f"Import error: {str(e)}")