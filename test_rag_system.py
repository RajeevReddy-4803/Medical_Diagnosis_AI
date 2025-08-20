"""
Test script for Medical RAG Conversational Search System
"""
import asyncio
import json
import requests
import time
from typing import List, Dict, Any
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystemTester:
    """Test the Medical RAG system"""
    
    def __init__(self, api_url: str = None, local_test: bool = True):
        self.api_url = api_url
        self.local_test = local_test
        self.test_results = []
        
        if local_test:
            # Import for local testing
            from src.rag_pipeline import MedicalRAGPipeline
            self.pipeline = MedicalRAGPipeline()
            self.pipeline.initialize_pipeline()
    
    def test_local_pipeline(self):
        """Test local RAG pipeline"""
        logger.info("Testing local RAG pipeline...")
        
        test_queries = [
            "What are the symptoms of diabetes?",
            "How is heart disease diagnosed?",
            "What causes Parkinson's disease?",
            "Tell me about lung cancer risk factors",
            "What is the treatment for hypothyroidism?",
            "How can I prevent diabetes?",
            "What are the early signs of heart disease?",
            "Is Parkinson's disease hereditary?",
            "What are the stages of lung cancer?",
            "How common is thyroid disease?"
        ]
        
        results = []
        total_time = 0
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"Testing query {i}/{len(test_queries)}: {query}")
            
            start_time = time.time()
            result = self.pipeline.process_query(query)
            end_time = time.time()
            
            response_time = end_time - start_time
            total_time += response_time
            
            test_result = {
                'query': query,
                'response': result['response'],
                'confidence': result['confidence'],
                'sources': result['sources'],
                'retrieved_docs': len(result['retrieved_documents']),
                'response_time': response_time
            }
            
            results.append(test_result)
            
            logger.info(f"Response time: {response_time:.2f}s, Confidence: {result['confidence']:.3f}")
            logger.info(f"Response: {result['response'][:100]}...")
            print("-" * 80)
        
        # Calculate metrics
        avg_response_time = total_time / len(test_queries)
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        
        logger.info(f"Average response time: {avg_response_time:.2f}s")
        logger.info(f"Average confidence: {avg_confidence:.3f}")
        
        return results
    
    def test_api_endpoint(self):
        """Test API endpoint"""
        if not self.api_url:
            logger.error("API URL not provided")
            return []
        
        logger.info(f"Testing API endpoint: {self.api_url}")
        
        test_queries = [
            "What are the symptoms of diabetes?",
            "How is heart disease diagnosed?",
            "What causes Parkinson's disease?",
            "Tell me about lung cancer risk factors",
            "What is hypothyroidism?"
        ]
        
        results = []
        session_id = "test_session_123"
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"Testing API query {i}/{len(test_queries)}: {query}")
            
            payload = {
                "query": query,
                "session_id": session_id,
                "use_hybrid": True,
                "include_history": True
            }
            
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.api_url}/query",
                    json=payload,
                    timeout=30
                )
                end_time = time.time()
                
                response_time = end_time - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    test_result = {
                        'query': query,
                        'response': data.get('response', ''),
                        'confidence': data.get('confidence', 0),
                        'sources': data.get('sources', []),
                        'retrieved_docs': data.get('retrieved_documents', 0),
                        'response_time': response_time,
                        'status_code': response.status_code
                    }
                    
                    logger.info(f"✓ Success - Response time: {response_time:.2f}s")
                    logger.info(f"Confidence: {data.get('confidence', 0):.3f}")
                    
                else:
                    logger.error(f"✗ API Error - Status: {response.status_code}")
                    test_result = {
                        'query': query,
                        'error': f"HTTP {response.status_code}",
                        'response_time': response_time,
                        'status_code': response.status_code
                    }
                
                results.append(test_result)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"✗ Request failed: {str(e)}")
                results.append({
                    'query': query,
                    'error': str(e),
                    'status_code': 0
                })
            
            time.sleep(1)  # Rate limiting
        
        return results
    
    def test_health_endpoint(self):
        """Test health endpoint"""
        if not self.api_url:
            return None
        
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                logger.info("✓ Health check passed")
                logger.info(f"Status: {data.get('status')}")
                return data
            else:
                logger.error(f"✗ Health check failed: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ Health check error: {str(e)}")
            return None
    
    def test_search_endpoint(self):
        """Test search endpoint"""
        if not self.api_url:
            return []
        
        search_queries = [
            "diabetes symptoms",
            "heart disease",
            "parkinson tremor",
            "lung cancer smoking",
            "thyroid function"
        ]
        
        results = []
        
        for query in search_queries:
            try:
                payload = {
                    "query": query,
                    "top_k": 3,
                    "threshold": 0.5
                }
                
                response = requests.post(
                    f"{self.api_url}/search",
                    json=payload,
                    timeout=15
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results.append({
                        'query': query,
                        'total_results': data.get('total_results', 0),
                        'results': data.get('results', [])
                    })
                    logger.info(f"✓ Search '{query}': {data.get('total_results', 0)} results")
                else:
                    logger.error(f"✗ Search failed for '{query}': {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"✗ Search error for '{query}': {str(e)}")
        
        return results
    
    def run_comprehensive_test(self):
        """Run comprehensive test suite"""
        logger.info("Starting comprehensive test suite...")
        
        test_results = {
            'timestamp': time.time(),
            'local_test': None,
            'api_test': None,
            'health_check': None,
            'search_test': None
        }
        
        # Test local pipeline
        if self.local_test:
            logger.info("\n" + "="*50)
            logger.info("TESTING LOCAL PIPELINE")
            logger.info("="*50)
            test_results['local_test'] = self.test_local_pipeline()
        
        # Test API endpoints
        if self.api_url:
            logger.info("\n" + "="*50)
            logger.info("TESTING API ENDPOINTS")
            logger.info("="*50)
            
            # Health check
            test_results['health_check'] = self.test_health_endpoint()
            
            # Query endpoint
            test_results['api_test'] = self.test_api_endpoint()
            
            # Search endpoint
            test_results['search_test'] = self.test_search_endpoint()
        
        # Save results
        self.save_test_results(test_results)
        
        # Generate report
        self.generate_test_report(test_results)
        
        return test_results
    
    def save_test_results(self, results: Dict[str, Any]):
        """Save test results to file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Test results saved to {filename}")
    
    def generate_test_report(self, results: Dict[str, Any]):
        """Generate test report"""
        logger.info("\n" + "="*60)
        logger.info("TEST REPORT")
        logger.info("="*60)
        
        # Local test results
        if results['local_test']:
            local_results = results['local_test']
            avg_time = sum(r['response_time'] for r in local_results) / len(local_results)
            avg_confidence = sum(r['confidence'] for r in local_results) / len(local_results)
            
            logger.info(f"LOCAL PIPELINE:")
            logger.info(f"  ✓ Queries processed: {len(local_results)}")
            logger.info(f"  ✓ Average response time: {avg_time:.2f}s")
            logger.info(f"  ✓ Average confidence: {avg_confidence:.3f}")
        
        # API test results
        if results['api_test']:
            api_results = results['api_test']
            successful = [r for r in api_results if r.get('status_code') == 200]
            failed = [r for r in api_results if r.get('status_code') != 200]
            
            logger.info(f"API ENDPOINTS:")
            logger.info(f"  ✓ Successful queries: {len(successful)}/{len(api_results)}")
            logger.info(f"  ✗ Failed queries: {len(failed)}")
            
            if successful:
                avg_time = sum(r['response_time'] for r in successful) / len(successful)
                avg_confidence = sum(r['confidence'] for r in successful) / len(successful)
                logger.info(f"  ✓ Average response time: {avg_time:.2f}s")
                logger.info(f"  ✓ Average confidence: {avg_confidence:.3f}")
        
        # Health check
        if results['health_check']:
            health = results['health_check']
            logger.info(f"HEALTH CHECK:")
            logger.info(f"  ✓ Status: {health.get('status')}")
            logger.info(f"  ✓ Pipeline initialized: {health.get('pipeline_initialized')}")
        
        # Search test
        if results['search_test']:
            search_results = results['search_test']
            total_searches = len(search_results)
            total_results = sum(r['total_results'] for r in search_results)
            
            logger.info(f"SEARCH FUNCTIONALITY:")
            logger.info(f"  ✓ Search queries: {total_searches}")
            logger.info(f"  ✓ Total results found: {total_results}")
        
        logger.info("="*60)

def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Medical RAG System')
    parser.add_argument('--api-url', help='API URL for testing')
    parser.add_argument('--local-only', action='store_true', help='Test local pipeline only')
    parser.add_argument('--api-only', action='store_true', help='Test API only')
    
    args = parser.parse_args()
    
    # Determine test mode
    local_test = not args.api_only
    api_url = args.api_url if not args.local_only else None
    
    # Load deployment info if available
    if not api_url and Path('deployment_info.json').exists():
        with open('deployment_info.json', 'r') as f:
            deployment_info = json.load(f)
            api_url = deployment_info.get('api_url')
            logger.info(f"Using API URL from deployment info: {api_url}")
    
    # Run tests
    tester = RAGSystemTester(api_url=api_url, local_test=local_test)
    results = tester.run_comprehensive_test()
    
    logger.info("Testing completed!")

if __name__ == '__main__':
    main()