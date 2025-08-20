import boto3
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
from botocore.exceptions import ClientError
from config import CONFIG

logger = logging.getLogger(__name__)

class AWSIntegration:
    """AWS integration for scalable deployment"""
    
    def __init__(self):
        self.region = CONFIG['aws'].region
        self.s3_bucket = CONFIG['aws'].s3_bucket
        self.dynamodb_table = CONFIG['aws'].dynamodb_table
        
        # Initialize AWS clients
        self.s3_client = boto3.client('s3', region_name=self.region)
        self.dynamodb = boto3.resource('dynamodb', region_name=self.region)
        self.lambda_client = boto3.client('lambda', region_name=self.region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=self.region)
        
        # DynamoDB table
        self.table = None
        self._initialize_dynamodb()
    
    def _initialize_dynamodb(self):
        """Initialize DynamoDB table"""
        try:
            self.table = self.dynamodb.Table(self.dynamodb_table)
            # Test table access
            self.table.load()
            logger.info(f"Connected to DynamoDB table: {self.dynamodb_table}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                logger.info(f"Creating DynamoDB table: {self.dynamodb_table}")
                self._create_dynamodb_table()
            else:
                logger.error(f"Error accessing DynamoDB: {str(e)}")
                raise
    
    def _create_dynamodb_table(self):
        """Create DynamoDB table for conversations"""
        try:
            table = self.dynamodb.create_table(
                TableName=self.dynamodb_table,
                KeySchema=[
                    {
                        'AttributeName': 'session_id',
                        'KeyType': 'HASH'
                    },
                    {
                        'AttributeName': 'timestamp',
                        'KeyType': 'RANGE'
                    }
                ],
                AttributeDefinitions=[
                    {
                        'AttributeName': 'session_id',
                        'AttributeType': 'S'
                    },
                    {
                        'AttributeName': 'timestamp',
                        'AttributeType': 'S'
                    }
                ],
                BillingMode='PAY_PER_REQUEST'
            )
            
            # Wait for table to be created
            table.wait_until_exists()
            self.table = table
            logger.info(f"Created DynamoDB table: {self.dynamodb_table}")
            
        except ClientError as e:
            logger.error(f"Error creating DynamoDB table: {str(e)}")
            raise
    
    def store_conversation(self, session_id: str, query: str, response: str, 
                          metadata: Dict[str, Any]) -> bool:
        """Store conversation in DynamoDB"""
        try:
            item = {
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'response': response,
                'metadata': metadata,
                'ttl': int((datetime.now().timestamp() + 86400 * 30))  # 30 days TTL
            }
            
            self.table.put_item(Item=item)
            return True
            
        except ClientError as e:
            logger.error(f"Error storing conversation: {str(e)}")
            return False
    
    def get_conversation_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation history from DynamoDB"""
        try:
            response = self.table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key('session_id').eq(session_id),
                ScanIndexForward=False,  # Most recent first
                Limit=limit
            )
            
            return response.get('Items', [])
            
        except ClientError as e:
            logger.error(f"Error retrieving conversation history: {str(e)}")
            return []
    
    def upload_model_to_s3(self, model_path: str, s3_key: str) -> bool:
        """Upload model files to S3"""
        try:
            self.s3_client.upload_file(model_path, self.s3_bucket, s3_key)
            logger.info(f"Uploaded model to S3: s3://{self.s3_bucket}/{s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            return False
    
    def download_model_from_s3(self, s3_key: str, local_path: str) -> bool:
        """Download model files from S3"""
        try:
            self.s3_client.download_file(self.s3_bucket, s3_key, local_path)
            logger.info(f"Downloaded model from S3: s3://{self.s3_bucket}/{s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Error downloading from S3: {str(e)}")
            return False
    
    def deploy_lambda_function(self, function_code: bytes, function_name: str) -> bool:
        """Deploy Lambda function"""
        try:
            # Check if function exists
            try:
                self.lambda_client.get_function(FunctionName=function_name)
                # Update existing function
                response = self.lambda_client.update_function_code(
                    FunctionName=function_name,
                    ZipFile=function_code
                )
                logger.info(f"Updated Lambda function: {function_name}")
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    # Create new function
                    response = self.lambda_client.create_function(
                        FunctionName=function_name,
                        Runtime='python3.9',
                        Role=f'arn:aws:iam::{boto3.client("sts").get_caller_identity()["Account"]}:role/lambda-execution-role',
                        Handler='lambda_function.lambda_handler',
                        Code={'ZipFile': function_code},
                        Description='Medical RAG API Lambda function',
                        Timeout=30,
                        MemorySize=512
                    )
                    logger.info(f"Created Lambda function: {function_name}")
                else:
                    raise
            
            return True
            
        except ClientError as e:
            logger.error(f"Error deploying Lambda function: {str(e)}")
            return False
    
    def log_metrics(self, metric_name: str, value: float, unit: str = 'Count'):
        """Log custom metrics to CloudWatch"""
        try:
            self.cloudwatch.put_metric_data(
                Namespace='MedicalRAG',
                MetricData=[
                    {
                        'MetricName': metric_name,
                        'Value': value,
                        'Unit': unit,
                        'Timestamp': datetime.now()
                    }
                ]
            )
        except ClientError as e:
            logger.error(f"Error logging metrics: {str(e)}")
    
    def create_api_gateway(self, lambda_function_arn: str) -> Optional[str]:
        """Create API Gateway for Lambda function"""
        try:
            apigateway = boto3.client('apigateway', region_name=self.region)
            
            # Create REST API
            api_response = apigateway.create_rest_api(
                name='medical-rag-api',
                description='Medical RAG Conversational Search API'
            )
            
            api_id = api_response['id']
            
            # Get root resource
            resources = apigateway.get_resources(restApiId=api_id)
            root_resource_id = resources['items'][0]['id']
            
            # Create resource
            resource_response = apigateway.create_resource(
                restApiId=api_id,
                parentId=root_resource_id,
                pathPart='query'
            )
            
            resource_id = resource_response['id']
            
            # Create method
            apigateway.put_method(
                restApiId=api_id,
                resourceId=resource_id,
                httpMethod='POST',
                authorizationType='NONE'
            )
            
            # Set integration
            apigateway.put_integration(
                restApiId=api_id,
                resourceId=resource_id,
                httpMethod='POST',
                type='AWS_PROXY',
                integrationHttpMethod='POST',
                uri=f'arn:aws:apigateway:{self.region}:lambda:path/2015-03-31/functions/{lambda_function_arn}/invocations'
            )
            
            # Deploy API
            apigateway.create_deployment(
                restApiId=api_id,
                stageName='prod'
            )
            
            api_url = f'https://{api_id}.execute-api.{self.region}.amazonaws.com/prod'
            logger.info(f"Created API Gateway: {api_url}")
            
            return api_url
            
        except ClientError as e:
            logger.error(f"Error creating API Gateway: {str(e)}")
            return None

class LambdaHandler:
    """Lambda function handler for serverless deployment"""
    
    def __init__(self):
        self.rag_pipeline = None
        self.aws_integration = AWSIntegration()
    
    def lambda_handler(self, event, context):
        """Main Lambda handler function"""
        try:
            # Initialize pipeline if not already done
            if not self.rag_pipeline:
                from .rag_pipeline import MedicalRAGPipeline
                self.rag_pipeline = MedicalRAGPipeline()
                self.rag_pipeline.initialize_pipeline()
            
            # Parse request
            body = json.loads(event.get('body', '{}'))
            query = body.get('query', '')
            session_id = body.get('session_id', str(uuid.uuid4()))
            
            if not query:
                return {
                    'statusCode': 400,
                    'body': json.dumps({'error': 'Query is required'})
                }
            
            # Process query
            result = self.rag_pipeline.process_query(query)
            
            # Store conversation
            self.aws_integration.store_conversation(
                session_id=session_id,
                query=query,
                response=result['response'],
                metadata={
                    'confidence': result['confidence'],
                    'sources': result['sources'],
                    'retrieved_documents': len(result['retrieved_documents'])
                }
            )
            
            # Log metrics
            self.aws_integration.log_metrics('QueryProcessed', 1)
            self.aws_integration.log_metrics('Confidence', result['confidence'], 'None')
            
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'response': result['response'],
                    'confidence': result['confidence'],
                    'sources': result['sources'],
                    'session_id': session_id
                })
            }
            
        except Exception as e:
            logger.error(f"Lambda handler error: {str(e)}")
            
            return {
                'statusCode': 500,
                'body': json.dumps({'error': 'Internal server error'})
            }

# Global handler instance for Lambda
lambda_handler_instance = LambdaHandler()

def lambda_handler(event, context):
    """Entry point for AWS Lambda"""
    return lambda_handler_instance.lambda_handler(event, context)