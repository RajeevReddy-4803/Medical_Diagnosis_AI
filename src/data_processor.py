import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MedicalDataProcessor:
    """Process medical datasets for RAG pipeline"""
    
    def __init__(self):
        self.datasets = {}
        self.processed_documents = []
        
    def load_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all medical datasets"""
        dataset_files = {
            'diabetes': 'Datasets/diabetes_data.csv',
            'heart_disease': 'Datasets/heart_disease_data.csv',
            'parkinsons': 'Datasets/parkinson_data.csv',
            'lung_cancer': 'Datasets/prepocessed_lungs_data.csv'
        }
        
        for name, file_path in dataset_files.items():
            try:
                if Path(file_path).exists():
                    self.datasets[name] = pd.read_csv(file_path)
                    logger.info(f"Loaded {name} dataset with {len(self.datasets[name])} records")
                else:
                    logger.warning(f"Dataset file not found: {file_path}")
            except Exception as e:
                logger.error(f"Error loading {name} dataset: {str(e)}")
                
        return self.datasets
    
    def create_medical_knowledge_base(self) -> List[Dict[str, Any]]:
        """Create structured knowledge base from datasets"""
        knowledge_base = []
        
        # Medical knowledge templates
        disease_info = {
            'diabetes': {
                'description': 'Diabetes is a chronic condition that affects how your body processes blood sugar (glucose).',
                'symptoms': ['Increased thirst', 'Frequent urination', 'Extreme fatigue', 'Blurred vision', 'Slow-healing wounds'],
                'risk_factors': ['Age', 'Family history', 'Obesity', 'Physical inactivity', 'High blood pressure'],
                'parameters': ['Glucose', 'BMI', 'Blood Pressure', 'Insulin', 'Age', 'Pregnancies']
            },
            'heart_disease': {
                'description': 'Heart disease refers to several types of heart conditions that affect heart function.',
                'symptoms': ['Chest pain', 'Shortness of breath', 'Fatigue', 'Irregular heartbeat', 'Dizziness'],
                'risk_factors': ['High cholesterol', 'High blood pressure', 'Smoking', 'Diabetes', 'Obesity'],
                'parameters': ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Chest Pain Type']
            },
            'parkinsons': {
                'description': 'Parkinson\'s disease is a progressive nervous system disorder that affects movement.',
                'symptoms': ['Tremor', 'Bradykinesia', 'Rigid muscles', 'Impaired posture', 'Speech changes'],
                'risk_factors': ['Age', 'Heredity', 'Sex', 'Exposure to toxins'],
                'parameters': ['Voice measurements', 'Frequency variations', 'Amplitude variations', 'Noise ratios']
            },
            'lung_cancer': {
                'description': 'Lung cancer is a type of cancer that begins in the lungs and can spread to other parts of the body.',
                'symptoms': ['Persistent cough', 'Chest pain', 'Shortness of breath', 'Weight loss', 'Fatigue'],
                'risk_factors': ['Smoking', 'Secondhand smoke', 'Radon exposure', 'Asbestos exposure', 'Family history'],
                'parameters': ['Smoking history', 'Age', 'Gender', 'Symptoms', 'Environmental factors']
            }
        }
        
        # Create knowledge documents
        for disease, info in disease_info.items():
            # Main disease information
            knowledge_base.append({
                'id': f"{disease}_overview",
                'disease': disease,
                'type': 'overview',
                'content': f"{disease.title()}: {info['description']}",
                'metadata': {
                    'disease': disease,
                    'category': 'overview',
                    'symptoms': info['symptoms'],
                    'risk_factors': info['risk_factors'],
                    'parameters': info['parameters']
                }
            })
            
            # Symptoms document
            symptoms_text = f"Common symptoms of {disease} include: " + ", ".join(info['symptoms'])
            knowledge_base.append({
                'id': f"{disease}_symptoms",
                'disease': disease,
                'type': 'symptoms',
                'content': symptoms_text,
                'metadata': {
                    'disease': disease,
                    'category': 'symptoms',
                    'symptoms': info['symptoms']
                }
            })
            
            # Risk factors document
            risk_factors_text = f"Risk factors for {disease} include: " + ", ".join(info['risk_factors'])
            knowledge_base.append({
                'id': f"{disease}_risk_factors",
                'disease': disease,
                'type': 'risk_factors',
                'content': risk_factors_text,
                'metadata': {
                    'disease': disease,
                    'category': 'risk_factors',
                    'risk_factors': info['risk_factors']
                }
            })
        
        # Add statistical insights from datasets
        for disease_name, df in self.datasets.items():
            if not df.empty:
                stats_content = self._generate_statistical_insights(disease_name, df)
                knowledge_base.append({
                    'id': f"{disease_name}_statistics",
                    'disease': disease_name,
                    'type': 'statistics',
                    'content': stats_content,
                    'metadata': {
                        'disease': disease_name,
                        'category': 'statistics',
                        'sample_size': len(df)
                    }
                })
        
        self.processed_documents = knowledge_base
        logger.info(f"Created knowledge base with {len(knowledge_base)} documents")
        return knowledge_base
    
    def _generate_statistical_insights(self, disease: str, df: pd.DataFrame) -> str:
        """Generate statistical insights from dataset"""
        insights = []
        
        # Basic statistics
        total_records = len(df)
        insights.append(f"Analysis based on {total_records} patient records for {disease}.")
        
        # Target distribution if available
        target_cols = ['target', 'Outcome', 'LUNG_CANCER', 'status']
        target_col = None
        for col in target_cols:
            if col in df.columns:
                target_col = col
                break
        
        if target_col:
            positive_cases = df[target_col].sum()
            positive_rate = (positive_cases / total_records) * 100
            insights.append(f"Positive cases: {positive_cases} ({positive_rate:.1f}%)")
        
        # Age statistics if available
        age_cols = ['age', 'Age', 'AGE']
        age_col = None
        for col in age_cols:
            if col in df.columns:
                age_col = col
                break
        
        if age_col:
            mean_age = df[age_col].mean()
            insights.append(f"Average age in dataset: {mean_age:.1f} years")
        
        # Gender distribution if available
        gender_cols = ['sex', 'Gender', 'GENDER']
        gender_col = None
        for col in gender_cols:
            if col in df.columns:
                gender_col = col
                break
        
        if gender_col:
            if df[gender_col].dtype == 'object':
                gender_dist = df[gender_col].value_counts()
                insights.append(f"Gender distribution: {dict(gender_dist)}")
            else:
                male_count = df[gender_col].sum()
                female_count = len(df) - male_count
                insights.append(f"Gender distribution: Male: {male_count}, Female: {female_count}")
        
        return " ".join(insights)
    
    def prepare_training_data(self) -> List[Dict[str, Any]]:
        """Prepare training data for fine-tuning"""
        training_data = []
        
        # Create question-answer pairs from knowledge base
        for doc in self.processed_documents:
            disease = doc['disease']
            content = doc['content']
            doc_type = doc['type']
            
            # Generate different types of questions
            if doc_type == 'overview':
                questions = [
                    f"What is {disease}?",
                    f"Tell me about {disease}",
                    f"Explain {disease} to me",
                    f"What do you know about {disease}?"
                ]
            elif doc_type == 'symptoms':
                questions = [
                    f"What are the symptoms of {disease}?",
                    f"How do I know if I have {disease}?",
                    f"What are the signs of {disease}?",
                    f"List symptoms of {disease}"
                ]
            elif doc_type == 'risk_factors':
                questions = [
                    f"What are the risk factors for {disease}?",
                    f"Who is at risk for {disease}?",
                    f"What increases my risk of {disease}?",
                    f"What causes {disease}?"
                ]
            elif doc_type == 'statistics':
                questions = [
                    f"What are the statistics for {disease}?",
                    f"How common is {disease}?",
                    f"Show me data about {disease}",
                    f"What does the research say about {disease}?"
                ]
            
            for question in questions:
                training_data.append({
                    'question': question,
                    'answer': content,
                    'disease': disease,
                    'category': doc_type,
                    'metadata': doc.get('metadata', {})
                })
        
        logger.info(f"Prepared {len(training_data)} training examples")
        return training_data
    
    def save_processed_data(self, output_dir: str = "processed_data"):
        """Save processed data to files"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save knowledge base
        with open(f"{output_dir}/knowledge_base.json", 'w') as f:
            json.dump(self.processed_documents, f, indent=2)
        
        # Save training data
        training_data = self.prepare_training_data()
        with open(f"{output_dir}/training_data.json", 'w') as f:
            json.dump(training_data, f, indent=2)
        
        logger.info(f"Saved processed data to {output_dir}")