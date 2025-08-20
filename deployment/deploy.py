"""
Deployment script for Medical RAG Conversational Search on AWS
"""
import boto3
import json
import zipfile
import os
import logging
from pathlib import Path
import subprocess
import sys
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AWSDeployment:
    """Handle AWS deployment of Medical RAG system"""
    
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.apigateway = boto3.client('apigateway', region_name=region)
        self.iam = boto3.client('iam', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        
        # Configuration
        self.function_name = 'medical-rag-api'
        self.api_name = 'medical-rag-conversational-search'
        self.bucket_name = 'medical-rag-models-bucket'
        self.table_name = 'medical-conversations'
        self.role_name = 'medical-rag-lambda-role'
    
    def create_deployment_package(self) -> str:
        """Create Lambda deployment package"""
        logger.info("Creating Lambda deployment package...")
        
        # Create temporary directory
        package_dir = Path('lambda_package')
        package_dir.mkdir(exist_ok=True)
        
        # Install dependencies
        logger.info("Installing dependencies...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install',
            '-r', 'requirements.txt',
            '-t', str(package_dir)
        ], check=True)
        
        # Copy source code
        logger.info("Copying source code...")
        import shutil
        
        # Copy src directory
        if Path('src').exists():
            shutil.copytree('src', package_dir / 'src', dirs_exist_ok=True)
        
        # Copy lambda function
        shutil.copy('lambda_function.py', package_dir)
        
        # Copy config
        shutil.copy('config.py', package_dir)
        
        # Copy datasets
        if Path('Datasets').exists():
            shutil.copytree('Datasets', package_dir / 'Datasets', dirs_exist_ok=True)
        
        # Create zip file
        zip_path = 'lambda_deployment.zip'
        logger.info(f"Creating zip file: {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(package_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(package_dir)
                    zipf.write(file_path, arcname)
        
        # Cleanup
        shutil.rmtree(package_dir)
        
        logger.info(f"Deployment package created: {zip_path}")
        return zip_path
    
    def create_iam_role(self) -> str:
        """Create IAM role for Lambda function"""
        logger.info("Creating IAM role...")
        
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "lambda.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        try:
            # Create role
            response = self.iam.create_role(
                RoleName=self.role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description='Role for Medical RAG Lambda function'
            )
            role_arn = response['Role']['Arn']
            
            # Attach policies
            policies = [
                'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
                'arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess',
                'arn:aws:iam::aws:policy/AmazonS3FullAccess',
                'arn:aws:iam::aws:policy/CloudWatchFullAccess'
            ]
            
            for policy_arn in policies:
                self.iam.attach_role_policy(
                    RoleName=self.role_name,
                    PolicyArn=policy_arn
                )
            
            logger.info(f"IAM role created: {role_arn}")
            return role_arn
            
        except self.iam.exceptions.EntityAlreadyExistsException:
            # Role already exists
            response = self.iam.get_role(RoleName=self.role_name)
            role_arn = response['Role']['Arn']
            logger.info(f"Using existing IAM role: {role_arn}")
            return role_arn
    
    def create_s3_bucket(self):
        """Create S3 bucket for model storage"""
        logger.info("Creating S3 bucket...")
        
        try:
            if self.region == 'us-east-1':
                self.s3.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            logger.info(f"S3 bucket created: {self.bucket_name}")
            
        except self.s3.exceptions.BucketAlreadyOwnedByYou:
            logger.info(f"S3 bucket already exists: {self.bucket_name}")
        except Exception as e:
            logger.error(f"Error creating S3 bucket: {str(e)}")
    
    def create_dynamodb_table(self):
        """Create DynamoDB table for conversations"""
        logger.info("Creating DynamoDB table...")
        
        try:
            table = self.dynamodb.create_table(
                TableName=self.table_name,
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
            logger.info(f"DynamoDB table created: {self.table_name}")
            
        except Exception as e:
            if 'ResourceInUseException' in str(e):
                logger.info(f"DynamoDB table already exists: {self.table_name}")
            else:
                logger.error(f"Error creating DynamoDB table: {str(e)}")
    
    def deploy_lambda_function(self, zip_path: str, role_arn: str) -> str:
        """Deploy Lambda function"""
        logger.info("Deploying Lambda function...")
        
        with open(zip_path, 'rb') as zip_file:
            zip_content = zip_file.read()
        
        try:
            # Try to update existing function
            response = self.lambda_client.update_function_code(
                FunctionName=self.function_name,
                ZipFile=zip_content
            )
            logger.info(f"Updated existing Lambda function: {self.function_name}")
            
        except self.lambda_client.exceptions.ResourceNotFoundException:
            # Create new function
            response = self.lambda_client.create_function(
                FunctionName=self.function_name,
                Runtime='python3.9',
                Role=role_arn,
                Handler='lambda_function.lambda_handler',
                Code={'ZipFile': zip_content},
                Description='Medical RAG Conversational Search API',
                Timeout=300,  # 5 minutes
                MemorySize=1024,  # 1GB
                Environment={
                    'Variables': {
                        'AWS_REGION': self.region,
                        'S3_BUCKET': self.bucket_name,
                        'DYNAMODB_TABLE': self.table_name
                    }
                }
            )
            logger.info(f"Created new Lambda function: {self.function_name}")
        
        function_arn = response['FunctionArn']
        return function_arn
    
    def create_api_gateway(self, lambda_arn: str) -> str:
        """Create API Gateway"""
        logger.info("Creating API Gateway...")
        
        try:
            # Create REST API
            api_response = self.apigateway.create_rest_api(
                name=self.api_name,
                description='Medical RAG Conversational Search API',
                endpointConfiguration={'types': ['REGIONAL']}
            )
            
            api_id = api_response['id']
            
            # Get root resource
            resources = self.apigateway.get_resources(restApiId=api_id)
            root_resource_id = resources['items'][0]['id']
            
            # Create resources and methods
            endpoints = [
                {'path': 'query', 'method': 'POST'},
                {'path': 'search', 'method': 'POST'},
                {'path': 'health', 'method': 'GET'}
            ]
            
            for endpoint in endpoints:
                # Create resource
                resource_response = self.apigateway.create_resource(
                    restApiId=api_id,
                    parentId=root_resource_id,
                    pathPart=endpoint['path']
                )
                
                resource_id = resource_response['id']
                
                # Create method
                self.apigateway.put_method(
                    restApiId=api_id,
                    resourceId=resource_id,
                    httpMethod=endpoint['method'],
                    authorizationType='NONE'
                )
                
                # Set integration
                self.apigateway.put_integration(
                    restApiId=api_id,
                    resourceId=resource_id,
                    httpMethod=endpoint['method'],
                    type='AWS_PROXY',
                    integrationHttpMethod='POST',
                    uri=f'arn:aws:apigateway:{self.region}:lambda:path/2015-03-31/functions/{lambda_arn}/invocations'
                )
                
                # Add CORS
                self.apigateway.put_method(
                    restApiId=api_id,
                    resourceId=resource_id,
                    httpMethod='OPTIONS',
                    authorizationType='NONE'
                )
                
                self.apigateway.put_integration(
                    restApiId=api_id,
                    resourceId=resource_id,
                    httpMethod='OPTIONS',
                    type='MOCK',
                    requestTemplates={'application/json': '{"statusCode": 200}'}
                )
                
                self.apigateway.put_integration_response(
                    restApiId=api_id,
                    resourceId=resource_id,
                    httpMethod='OPTIONS',
                    statusCode='200',
                    responseParameters={
                        'method.response.header.Access-Control-Allow-Headers': "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'",
                        'method.response.header.Access-Control-Allow-Methods': "'GET,POST,OPTIONS'",
                        'method.response.header.Access-Control-Allow-Origin': "'*'"
                    }
                )
                
                self.apigateway.put_method_response(
                    restApiId=api_id,
                    resourceId=resource_id,
                    httpMethod='OPTIONS',
                    statusCode='200',
                    responseParameters={
                        'method.response.header.Access-Control-Allow-Headers': True,
                        'method.response.header.Access-Control-Allow-Methods': True,
                        'method.response.header.Access-Control-Allow-Origin': True
                    }
                )
            
            # Add Lambda permission
            self.lambda_client.add_permission(
                FunctionName=self.function_name,
                StatementId='api-gateway-invoke',
                Action='lambda:InvokeFunction',
                Principal='apigateway.amazonaws.com',
                SourceArn=f'arn:aws:execute-api:{self.region}:*:{api_id}/*/*'
            )
            
            # Deploy API
            self.apigateway.create_deployment(
                restApiId=api_id,
                stageName='prod'
            )
            
            api_url = f'https://{api_id}.execute-api.{self.region}.amazonaws.com/prod'
            logger.info(f"API Gateway created: {api_url}")
            
            return api_url
            
        except Exception as e:
            logger.error(f"Error creating API Gateway: {str(e)}")
            raise
    
    def deploy_full_stack(self) -> Dict[str, str]:
        """Deploy complete stack"""
        logger.info("Starting full stack deployment...")
        
        try:
            # Create AWS resources
            self.create_s3_bucket()
            self.create_dynamodb_table()
            role_arn = self.create_iam_role()
            
            # Wait for role to propagate
            import time
            time.sleep(10)
            
            # Create deployment package
            zip_path = self.create_deployment_package()
            
            # Deploy Lambda
            lambda_arn = self.deploy_lambda_function(zip_path, role_arn)
            
            # Create API Gateway
            api_url = self.create_api_gateway(lambda_arn)
            
            # Cleanup
            os.remove(zip_path)
            
            deployment_info = {
                'api_url': api_url,
                'lambda_function': self.function_name,
                'lambda_arn': lambda_arn,
                's3_bucket': self.bucket_name,
                'dynamodb_table': self.table_name,
                'region': self.region
            }
            
            logger.info("Deployment completed successfully!")
            logger.info(f"API URL: {api_url}")
            
            return deployment_info
            
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            raise

def main():
    """Main deployment function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy Medical RAG to AWS')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    args = parser.parse_args()
    
    deployment = AWSDeployment(region=args.region)
    
    try:
        result = deployment.deploy_full_stack()
        
        print("\n" + "="*50)
        print("DEPLOYMENT SUCCESSFUL!")
        print("="*50)
        print(f"API URL: {result['api_url']}")
        print(f"Region: {result['region']}")
        print(f"Lambda Function: {result['lambda_function']}")
        print(f"S3 Bucket: {result['s3_bucket']}")
        print(f"DynamoDB Table: {result['dynamodb_table']}")
        print("="*50)
        
        # Save deployment info
        with open('deployment_info.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print("Deployment info saved to deployment_info.json")
        
    except Exception as e:
        print(f"Deployment failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()