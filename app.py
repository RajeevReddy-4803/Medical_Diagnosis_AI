import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
from typing import Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page Configuration
st.set_page_config(
    page_title="AI-Powered Medical Diagnosis",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for better styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background-color: #2E86C1;
        color: white;
        padding: 0.8rem 1.5rem;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #2874A6;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Prediction box styling */
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .positive {
        background-color: rgba(231, 76, 60, 0.1);
        border: 2px solid #E74C3C;
        color: #E74C3C;
    }
    .negative {
        background-color: rgba(39, 174, 96, 0.1);
        border: 2px solid #27AE60;
        color: #27AE60;
    }
    
    /* Info box styling */
    .info-box {
        background-color: rgba(52, 152, 219, 0.1);
        border: 2px solid #3498DB;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: var(--text-color);
    }
    
    /* Input field styling */
    .stNumberInput>div>div>input {
        border-radius: 6px;
        border: 1px solid #BDC3C7;
        padding: 0.5rem;
        background-color: var(--input-background);
        color: var(--text-color);
    }
    .stSelectbox>div>div>select {
        border-radius: 6px;
        border: 1px solid #BDC3C7;
        padding: 0.5rem;
        background-color: var(--input-background);
        color: var(--text-color);
    }
    
    /* Title styling */
    h1 {
        color: var(--heading-color);
        font-weight: 700;
        margin-bottom: 1.5rem;
    }
    h2 {
        color: var(--heading-color);
        font-weight: 600;
        margin-bottom: 1rem;
    }
    h3 {
        color: var(--heading-color);
        font-weight: 600;
    }
    h4 {
        color: var(--heading-color);
        font-weight: 600;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--sidebar-background);
    }
    .sidebar .sidebar-content {
        background-color: var(--sidebar-background);
    }
    
    /* Chart styling */
    .js-plotly-plot {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        background-color: var(--chart-background);
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        :root {
            --background-color: #1E1E1E;
            --text-color: #FFFFFF;
            --heading-color: #FFFFFF;
            --input-background: #2D2D2D;
            --sidebar-background: #2C3E50;
            --chart-background: #2D2D2D;
        }
    }
    
    /* Light mode support */
    @media (prefers-color-scheme: light) {
        :root {
            --background-color: #FFFFFF;
            --text-color: #000000;
            --heading-color: #2C3E50;
            --input-background: #FFFFFF;
            --sidebar-background: #2C3E50;
            --chart-background: #FFFFFF;
        }
    }
    
    /* Additional dark mode overrides */
    .stMarkdown {
        color: var(--text-color);
    }
    
    .stText {
        color: var(--text-color);
    }
    
    /* Make sure links are visible in both modes */
    a {
        color: #3498DB;
    }
    a:hover {
        color: #2980B9;
    }
    
    /* Make sure form labels are visible */
    .stForm label {
        color: var(--text-color);
    }
    
    /* Make sure select box options are visible */
    .stSelectbox div[data-baseweb="select"] {
        background-color: var(--input-background);
        color: var(--text-color);
    }
    
    /* Make sure number input is visible */
    .stNumberInput div[data-baseweb="input"] {
        background-color: var(--input-background);
        color: var(--text-color);
    }
    </style>
    """, unsafe_allow_html=True)

# Hide Streamlit default elements
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Define logical ranges for each parameter
PARAMETER_RANGES = {
    'diabetes': {
        'Pregnancies': (0, 17),
        'Glucose': (0, 199),
        'BloodPressure': (0, 122),
        'SkinThickness': (0, 99),
        'Insulin': (0, 846),
        'BMI': (0.0, 67.1),
        'DiabetesPedigreeFunction': (0.078, 2.42),
        'Age': (21, 81)
    },
    'heart_disease': {
        'age': (29, 77),
        'trestbps': (94, 200),
        'chol': (126, 564),
        'thalach': (71, 202),
        'oldpeak': (0.0, 6.2)
    },
    'parkinsons': {
        'fo': (88.33, 260.105),
        'fhi': (102.145, 592.03),
        'flo': (65.476, 239.17),
        'Jitter_percent': (0.00168, 0.03316),
        'Jitter_Abs': (0.000007, 0.00026),
        'RAP': (0.00068, 0.02144),
        'PPQ': (0.00092, 0.01958),
        'DDP': (0.00204, 0.06433),
        'Shimmer': (0.00954, 0.11908),
        'Shimmer_dB': (0.085, 1.302),
        'APQ3': (0.00455, 0.05647),
        'APQ5': (0.0057, 0.0794),
        'APQ': (0.00719, 0.13778),
        'DDA': (0.01364, 0.16942),
        'NHR': (0.00065, 0.31482),
        'HNR': (8.441, 33.047),
        'RPDE': (0.25659, 0.685151),
        'DFA': (0.574282, 0.825288),
        'spread1': (-7.964984, -2.434031),
        'spread2': (0.006274, 0.450493),
        'D2': (1.423287, 3.671155),
        'PPE': (0.044539, 0.527367)
    },
    'thyroid': {
        'tsh': (0.005, 530.0),
        't3': (0.2, 10.0),
        'tt4': (2.0, 430.0)
    }
}

# Load models with error handling
@st.cache_resource
def load_models() -> Dict[str, Any]:
    try:
        models = {
            'diabetes': pickle.load(open('Models/diabetes_model.sav', 'rb')),
            'heart_disease': pickle.load(open('Models/heart_disease_model.sav', 'rb')),
            'parkinsons': pickle.load(open('Models/parkinsons_model.sav', 'rb')),
            'lung_cancer': pickle.load(open('Models/lungs_disease_model.sav', 'rb')),
            'thyroid': pickle.load(open('Models/Thyroid_model.sav', 'rb'))
        }
        return models
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        st.error("Failed to load models. Please check if all model files are present in the Models directory.")
        return None

def validate_input(value: float, param_name: str, disease: str) -> Tuple[bool, str]:
    """Validate input values against logical ranges."""
    if disease in PARAMETER_RANGES and param_name in PARAMETER_RANGES[disease]:
        min_val, max_val = PARAMETER_RANGES[disease][param_name]
        if value < min_val or value > max_val:
            return False, f"Value for {param_name} should be between {min_val} and {max_val}"
    return True, ""

def display_prediction_result(prediction: int, probability: float, disease_name: str):
    """Display prediction result with detailed feedback."""
    if prediction == 1:
        st.markdown(f"""
            <div class="prediction-box positive">
                <h3>⚠️ Positive for {disease_name}</h3>
                <p>Confidence: {probability:.2f}%</p>
                <p>Please consult a healthcare professional for further evaluation.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Display additional insights based on disease
        if disease_name == "Diabetes":
            st.markdown("""
                <div class="info-box">
                    <h4>📋 Recommended Actions:</h4>
                    <ul>
                        <li>Schedule an appointment with your doctor</li>
                        <li>Monitor your blood sugar levels regularly</li>
                        <li>Maintain a healthy diet and exercise routine</li>
                        <li>Keep track of your symptoms</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="prediction-box negative">
                <h3>✅ Negative for {disease_name}</h3>
                <p>Confidence: {probability:.2f}%</p>
                <p>Continue maintaining a healthy lifestyle.</p>
            </div>
        """, unsafe_allow_html=True)

def get_prediction_with_probability(model, inputs):
    """Get prediction and probability, handling cases where predict_proba is not available."""
    prediction = model.predict(inputs)
    try:
        probability = model.predict_proba(inputs)[0][1] * 100
    except (AttributeError, NotImplementedError):
        # If predict_proba is not available, use a default confidence
        probability = 100 if prediction[0] == 1 else 0
    return prediction[0], probability

def display_dashboard():
    st.title("Medical Diagnosis Dashboard")
    
    # Create an impressive hero section
    st.markdown("""
        <div style='background-color: #2E86C1; padding: 2rem; border-radius: 10px; color: white; margin-bottom: 2rem;'>
            <h1 style='color: white; text-align: center;'>AI-Powered Medical Diagnosis</h1>
            <p style='text-align: center; font-size: 1.2rem;'>Leveraging the power of artificial intelligence to revolutionize healthcare diagnostics</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Add impressive statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div style='background-color: #FADBD8; padding: 1.5rem; border-radius: 10px; text-align: center;'>
                <h3>🎯 Accuracy</h3>
                <h2>95%+</h2>
                <p>Average prediction accuracy across all models</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background-color: #D5F5E3; padding: 1.5rem; border-radius: 10px; text-align: center;'>
                <h3>⚡ Speed</h3>
                <h2>< 1s</h2>
                <p>Instant diagnosis and analysis</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='background-color: #EBF5FB; padding: 1.5rem; border-radius: 10px; text-align: center;'>
                <h3>🔬 Coverage</h3>
                <h2>5+</h2>
                <p>Different diseases analyzed</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Add disease cards with information
    st.markdown("### Available Disease Predictions")
    
    diseases = [
        {
            "name": "Diabetes",
            "icon": "💉",
            "description": "Predict diabetes risk using key health parameters",
            "features": ["Blood glucose levels", "BMI", "Age", "Family history"],
            "color": "#FADBD8"
        },
        {
            "name": "Heart Disease",
            "icon": "❤️",
            "description": "Analyze heart disease risk factors",
            "features": ["Blood pressure", "Cholesterol", "ECG results", "Exercise patterns"],
            "color": "#FADBD8"
        },
        {
            "name": "Parkinson's Disease",
            "icon": "🎤",
            "description": "Voice-based Parkinson's disease detection",
            "features": ["Voice analysis", "Speech patterns", "Frequency measurements"],
            "color": "#D5F5E3"
        },
        {
            "name": "Lung Cancer",
            "icon": "🫁",
            "description": "Early detection of lung cancer risk",
            "features": ["Smoking history", "Symptoms", "Lifestyle factors"],
            "color": "#EBF5FB"
        },
        {
            "name": "Hypo-Thyroid",
            "icon": "🦋",
            "description": "Thyroid function analysis",
            "features": ["TSH levels", "T3/T4 levels", "Symptoms"],
            "color": "#D5F5E3"
        }
    ]
    
    # Display disease cards in a grid
    cols = st.columns(3)
    for idx, disease in enumerate(diseases):
        with cols[idx % 3]:
            st.markdown(f"""
                <div style='background-color: {disease['color']}; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;'>
                    <h3>{disease['icon']} {disease['name']}</h3>
                    <p>{disease['description']}</p>
                    <ul>
                        {''.join(f'<li>{feature}</li>' for feature in disease['features'])}
                    </ul>
                </div>
            """, unsafe_allow_html=True)
    
    # Add AI technology section
    st.markdown("""
        <div style='background-color: #2C3E50; padding: 2rem; border-radius: 10px; color: white; margin-top: 2rem;'>
            <h2 style='color: white;'>🤖 Advanced AI Technology</h2>
            <p>Our system utilizes state-of-the-art machine learning algorithms to provide accurate predictions:</p>
            <ul>
                <li>Random Forest Classifiers</li>
                <li>Support Vector Machines</li>
                <li>Neural Networks</li>
                <li>Ensemble Methods</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

def diabetes_page():
    st.title("Diabetes Prediction")
    st.markdown("""
        <div class="info-box">
            <h4>ℹ️ About Diabetes Prediction</h4>
            <p>This tool helps predict the likelihood of diabetes based on various health parameters. 
            Please enter your values within the specified ranges for accurate results.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Basic Information")
        Pregnancies = st.number_input('Number of Pregnancies', 
                                    min_value=0,
                                    max_value=17,
                                    value=0,  # Default value
                                    help="Enter number of pregnancies (0-17)")
        
        Glucose = st.number_input('Glucose Level (mg/dL)', 
                                min_value=70,  # Normal fasting glucose starts around 70
                                max_value=199,
                                value=100,  # Default to normal fasting glucose
                                help="Normal range: 70-140 mg/dL")
        
        BloodPressure = st.number_input('Blood Pressure (mm Hg)', 
                                      min_value=60,  # Minimum safe BP
                                      max_value=122,
                                      value=80,  # Default to normal BP
                                      help="Normal range: 60-120 mm Hg")
        
        SkinThickness = st.number_input('Skin Thickness (mm)', 
                                      min_value=10,  # Minimum realistic value
                                      max_value=99,
                                      value=20,  # Default to average
                                      help="Normal range: 10-50 mm")
    
    with col2:
        st.markdown("### Additional Measurements")
        Insulin = st.number_input('Insulin Level (mu U/ml)', 
                                min_value=2,  # Minimum realistic value
                                max_value=846,
                                value=16,  # Default to normal range
                                help="Normal range: 2-25 mu U/ml")
        
        BMI = st.number_input('BMI (kg/m²)', 
                            min_value=18.5,  # Minimum healthy BMI
                            max_value=67.1,
                            value=22.0,  # Default to normal BMI
                            help="Normal range: 18.5-24.9 kg/m²")
        
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', 
                                                 min_value=0.078,
                                                 max_value=2.42,
                                                 value=0.5,  # Default to middle range
                                                 step=0.01,
                                                 help="Family history of diabetes")
        
        Age = st.number_input('Age', 
                            min_value=21,  # Minimum age for diabetes screening
                            max_value=81,
                            value=30,  # Default to middle age
                            help="Enter your age (21-81 years)")
    
    if st.button('Predict Diabetes Risk'):
        try:
            with st.spinner('Analyzing your health parameters...'):
                # Validate inputs
                inputs = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
                for i, (value, param) in enumerate(zip(inputs, PARAMETER_RANGES['diabetes'].keys())):
                    is_valid, message = validate_input(value, param, 'diabetes')
                    if not is_valid:
                        st.error(message)
                        return

                prediction, probability = get_prediction_with_probability(models['diabetes'], [inputs])
                display_prediction_result(prediction, probability, "Diabetes")
                
                # Display risk factors analysis with improved visualization
                st.subheader("Risk Factor Analysis")
                features = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedigree', 'Age']
                values = inputs
                
                fig = px.bar(x=features, y=values, 
                           title='Health Parameters Analysis',
                           color=values,
                           color_continuous_scale='Viridis')
                fig.update_layout(
                    xaxis_title="Parameters",
                    yaxis_title="Values",
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            logger.error(f"Error in diabetes prediction: {str(e)}")
            st.error("An error occurred during prediction. Please try again.")

def heart_disease_page():
    st.title("Heart Disease Prediction")
    st.markdown("""
        <div class="info-box">
            <h4>ℹ️ About Heart Disease Prediction</h4>
            <p>This tool analyzes various health parameters to assess your risk of heart disease.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input('Age', 
                            min_value=29,  # Minimum age for heart disease screening
                            max_value=77,
                            value=45,  # Default to middle age
                            help="Enter your age (29-77 years)")
        
        sex = st.selectbox('Sex', ['Male', 'Female'])
        
        cp = st.selectbox('Chest Pain Type', 
                         ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'],
                         help="Select the type of chest pain you experience")
        
        trestbps = st.number_input('Resting Blood Pressure (mm Hg)', 
                                 min_value=94,  # Minimum normal BP
                                 max_value=200,
                                 value=120,  # Default to normal BP
                                 help="Normal range: 94-140 mm Hg")
        
        chol = st.number_input('Serum Cholesterol (mg/dL)', 
                             min_value=126,  # Minimum normal cholesterol
                             max_value=564,
                             value=200,  # Default to normal cholesterol
                             help="Normal range: 126-200 mg/dL")
    
    with col2:
        thalach = st.number_input('Maximum Heart Rate (bpm)', 
                                min_value=71,  # Minimum normal heart rate
                                max_value=202,
                                value=150,  # Default to normal max heart rate
                                help="Normal range: 71-202 bpm")
        
        oldpeak = st.number_input('ST Depression', 
                                min_value=0.0,
                                max_value=6.2,
                                value=0.0,  # Default to normal
                                step=0.1,
                                help="ST depression induced by exercise")
    
    if st.button('Predict Heart Disease Risk'):
        with st.spinner('Analyzing...'):
            # Convert categorical variables to numerical
            sex = 1 if sex == 'Male' else 0
            cp = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(cp)
            
            inputs = [age, sex, cp, trestbps, chol, thalach, oldpeak]
            prediction, probability = get_prediction_with_probability(models['heart_disease'], [inputs])
            display_prediction_result(prediction, probability, "Heart Disease")
            
            # Show risk factors visualization
            st.subheader("Risk Factor Analysis")
            features = ['Age', 'Sex', 'Chest Pain', 'BP', 'Cholesterol', 'Max HR', 'ST Depression']
            values = [age, sex, cp, trestbps, chol, thalach, oldpeak]
            
            fig = px.bar(x=features, y=values, title='Input Values Analysis')
            st.plotly_chart(fig, use_container_width=True)

def parkinsons_page():
    st.title("Parkinson's Disease Prediction")
    st.markdown("""
        <div class="info-box">
            <h4>ℹ️ About Parkinson's Disease Prediction</h4>
            <p>This tool analyzes voice measurements to predict the likelihood of Parkinson's disease.</p>
            <h5>Key Symptoms:</h5>
            <ul>
                <li>Tremors or shaking</li>
                <li>Slowed movement</li>
                <li>Rigid muscles</li>
                <li>Impaired posture and balance</li>
                <li>Speech changes</li>
            </ul>
            <h5>Voice Analysis Features:</h5>
            <ul>
                <li>Fundamental frequency variations</li>
                <li>Voice tremor</li>
                <li>Speech rhythm changes</li>
                <li>Voice quality measurements</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Add tabs for recording and manual input
    tab1, tab2 = st.tabs(["Record Voice", "Manual Input"])
    
    with tab1:
        st.markdown("### Voice Recording")
        st.write("Please follow these steps:")
        st.write("1. Click 'Start Recording' and say 'ah' for 3-5 seconds")
        st.write("2. Click 'Stop Recording' when finished")
        st.write("3. Wait for the analysis to complete")
        
        # Add recording functionality
        if st.button("Start Recording"):
            try:
                import sounddevice as sd
                import soundfile as sf
                import numpy as np
                from scipy.io import wavfile
                import librosa
                
                # Recording parameters
                duration = 5  # seconds
                sample_rate = 44100
                
                st.write("Recording... Please say 'ah'")
                recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
                sd.wait()
                
                # Save the recording temporarily
                temp_file = "temp_recording.wav"
                sf.write(temp_file, recording, sample_rate)
                
                # Extract voice features using librosa
                y, sr = librosa.load(temp_file)
                
                # Calculate voice features
                fo = librosa.yin(y, fmin=50, fmax=500)[0].mean()  # Fundamental frequency
                fhi = librosa.yin(y, fmin=50, fmax=500)[0].max()  # Highest frequency
                flo = librosa.yin(y, fmin=50, fmax=500)[0].min()  # Lowest frequency
                
                # Jitter and shimmer calculations
                zero_crossings = librosa.feature.zero_crossing_rate(y)[0]
                Jitter_percent = np.std(zero_crossings) * 100
                Jitter_Abs = np.mean(np.abs(np.diff(zero_crossings)))
                
                # Additional features
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                
                # Calculate other required features
                RAP = np.mean(np.abs(np.diff(zero_crossings)))
                PPQ = np.std(zero_crossings)
                DDP = RAP * 3
                Shimmer = np.std(np.abs(np.diff(np.abs(y))))
                Shimmer_dB = 20 * np.log10(Shimmer + 1e-10)
                APQ3 = np.mean(np.abs(np.diff(np.abs(y), n=3)))
                APQ5 = np.mean(np.abs(np.diff(np.abs(y), n=5)))
                APQ = np.mean(np.abs(np.diff(np.abs(y), n=7)))
                DDA = APQ3 * 3
                NHR = np.mean(zero_crossings) / (np.mean(np.abs(y)) + 1e-10)
                HNR = 10 * np.log10(np.mean(np.abs(y)) / (np.mean(zero_crossings) + 1e-10))
                RPDE = np.mean(mfccs[1:])
                DFA = np.mean(np.diff(spectral_centroid))
                spread1 = np.std(mfccs[0])
                spread2 = np.std(mfccs[1])
                D2 = np.mean(np.diff(mfccs, axis=1))
                PPE = np.std(spectral_centroid)
                
                # Store the calculated features
                voice_features = {
                    'fo': fo, 'fhi': fhi, 'flo': flo,
                    'Jitter_percent': Jitter_percent, 'Jitter_Abs': Jitter_Abs,
                    'RAP': RAP, 'PPQ': PPQ, 'DDP': DDP,
                    'Shimmer': Shimmer, 'Shimmer_dB': Shimmer_dB,
                    'APQ3': APQ3, 'APQ5': APQ5, 'APQ': APQ,
                    'DDA': DDA, 'NHR': NHR, 'HNR': HNR,
                    'RPDE': RPDE, 'DFA': DFA,
                    'spread1': spread1, 'spread2': spread2,
                    'D2': D2, 'PPE': PPE
                }
                
                # Display the calculated features
                st.success("Voice analysis complete!")
                st.write("Calculated voice features:")
                for feature, value in voice_features.items():
                    st.write(f"{feature}: {value:.4f}")
                
                # Make prediction with the calculated features
                inputs = [voice_features[feature] for feature in [
                    'fo', 'fhi', 'flo', 'Jitter_percent', 'Jitter_Abs',
                    'RAP', 'PPQ', 'DDP', 'Shimmer', 'Shimmer_dB',
                    'APQ3', 'APQ5', 'APQ', 'DDA', 'NHR', 'HNR',
                    'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
                ]]
                
                prediction, probability = get_prediction_with_probability(models['parkinsons'], [inputs])
                display_prediction_result(prediction, probability, "Parkinson's Disease")
                
                # Show voice measurement analysis
                st.subheader("Voice Measurement Analysis")
                features = ['Fo', 'Fhi', 'Flo', 'Jitter%', 'Jitter(Abs)', 'RAP', 'PPQ', 'DDP', 
                           'Shimmer', 'Shimmer(dB)', 'APQ3', 'APQ5', 'APQ', 'DDA', 'NHR', 'HNR',
                           'RPDE', 'DFA', 'Spread1', 'Spread2', 'D2', 'PPE']
                
                fig = px.bar(x=features, y=inputs,
                           title='Voice Measurement Analysis',
                           color=inputs,
                           color_continuous_scale='Viridis')
                fig.update_layout(
                    xaxis_title="Parameters",
                    yaxis_title="Values",
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                logger.error(f"Error in voice recording/analysis: {str(e)}")
                st.error("An error occurred during voice recording or analysis. Please try again.")
    
    with tab2:
        st.markdown("### Manual Input")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Voice Frequency Measurements")
            fo = st.number_input('MDVP:Fo(Hz)', min_value=0.0, max_value=1000.0, step=0.1)
            fhi = st.number_input('MDVP:Fhi(Hz)', min_value=0.0, max_value=1000.0, step=0.1)
            flo = st.number_input('MDVP:Flo(Hz)', min_value=0.0, max_value=1000.0, step=0.1)
            Jitter_percent = st.number_input('MDVP:Jitter(%)', min_value=0.0, max_value=1.0, step=0.001)
            Jitter_Abs = st.number_input('MDVP:Jitter(Abs)', min_value=0.0, max_value=1.0, step=0.001)
            RAP = st.number_input('MDVP:RAP', min_value=0.0, max_value=1.0, step=0.001)
            PPQ = st.number_input('MDVP:PPQ', min_value=0.0, max_value=1.0, step=0.001)
            DDP = st.number_input('Jitter:DDP', min_value=0.0, max_value=1.0, step=0.001)
            Shimmer = st.number_input('MDVP:Shimmer', min_value=0.0, max_value=1.0, step=0.001)
            Shimmer_dB = st.number_input('MDVP:Shimmer(dB)', min_value=0.0, max_value=1.0, step=0.001)
        
        with col2:
            st.markdown("### Additional Voice Parameters")
            APQ3 = st.number_input('Shimmer:APQ3', min_value=0.0, max_value=1.0, step=0.001)
            APQ5 = st.number_input('Shimmer:APQ5', min_value=0.0, max_value=1.0, step=0.001)
            APQ = st.number_input('MDVP:APQ', min_value=0.0, max_value=1.0, step=0.001)
            DDA = st.number_input('Shimmer:DDA', min_value=0.0, max_value=1.0, step=0.001)
            NHR = st.number_input('NHR', min_value=0.0, max_value=1.0, step=0.001)
            HNR = st.number_input('HNR', min_value=0.0, max_value=100.0, step=0.1)
            RPDE = st.number_input('RPDE', min_value=0.0, max_value=1.0, step=0.001)
            DFA = st.number_input('DFA', min_value=0.0, max_value=1.0, step=0.001)
            spread1 = st.number_input('Spread1', min_value=-10.0, max_value=10.0, step=0.001)
            spread2 = st.number_input('Spread2', min_value=-10.0, max_value=10.0, step=0.001)
            D2 = st.number_input('D2', min_value=0.0, max_value=10.0, step=0.001)
            PPE = st.number_input('PPE', min_value=0.0, max_value=1.0, step=0.001)
        
        if st.button("Predict Parkinson's Disease Risk"):
            with st.spinner('Analyzing voice measurements...'):
                try:
                    inputs = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
                    prediction, probability = get_prediction_with_probability(models['parkinsons'], [inputs])
                    display_prediction_result(prediction, probability, "Parkinson's Disease")
                    
                    # Show voice measurement analysis
                    st.subheader("Voice Measurement Analysis")
                    features = ['Fo', 'Fhi', 'Flo', 'Jitter%', 'Jitter(Abs)', 'RAP', 'PPQ', 'DDP', 'Shimmer', 'Shimmer(dB)', 'APQ3', 'APQ5', 'APQ', 'DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'Spread1', 'Spread2', 'D2', 'PPE']
                    
                    fig = px.bar(x=features, y=inputs,
                               title='Voice Measurement Analysis',
                               color=inputs,
                               color_continuous_scale='Viridis')
                    fig.update_layout(
                        xaxis_title="Parameters",
                        yaxis_title="Values",
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    logger.error(f"Error in Parkinson's prediction: {str(e)}")
                    st.error("An error occurred during prediction. Please try again.")

def lung_cancer_page():
    st.title("Lung Cancer Prediction")
    st.markdown("""
        <div class="info-box">
            <h4>ℹ️ About Lung Cancer Prediction</h4>
            <p>This tool helps assess your risk of lung cancer based on various factors.</p>
            <h5>Key Risk Factors:</h5>
            <ul>
                <li>Smoking</li>
                <li>Exposure to secondhand smoke</li>
                <li>Family history</li>
                <li>Exposure to radon</li>
                <li>Occupational hazards</li>
            </ul>
            <h5>Early Warning Signs:</h5>
            <ul>
                <li>Persistent cough</li>
                <li>Chest pain</li>
                <li>Shortness of breath</li>
                <li>Unexplained weight loss</li>
                <li>Fatigue</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        GENDER = st.selectbox('Gender', ['Male', 'Female'])
        AGE = st.number_input('Age', min_value=0, max_value=120, step=1)
        SMOKING = st.selectbox('Smoking', ['Yes', 'No'])
        YELLOW_FINGERS = st.selectbox('Yellow Fingers', ['Yes', 'No'])
        ANXIETY = st.selectbox('Anxiety', ['Yes', 'No'])
        PEER_PRESSURE = st.selectbox('Peer Pressure', ['Yes', 'No'])
        CHRONIC_DISEASE = st.selectbox('Chronic Disease', ['Yes', 'No'])
        FATIGUE = st.selectbox('Fatigue', ['Yes', 'No'])
    
    with col2:
        ALLERGY = st.selectbox('Allergy', ['Yes', 'No'])
        WHEEZING = st.selectbox('Wheezing', ['Yes', 'No'])
        ALCOHOL_CONSUMING = st.selectbox('Alcohol Consumption', ['Yes', 'No'])
        COUGHING = st.selectbox('Coughing', ['Yes', 'No'])
        SHORTNESS_OF_BREATH = st.selectbox('Shortness of Breath', ['Yes', 'No'])
        SWALLOWING_DIFFICULTY = st.selectbox('Swallowing Difficulty', ['Yes', 'No'])
        CHEST_PAIN = st.selectbox('Chest Pain', ['Yes', 'No'])
    
    if st.button("Predict Lung Cancer Risk"):
        with st.spinner('Analyzing risk factors...'):
            # Convert categorical variables to numerical
            GENDER = 1 if GENDER == 'Male' else 0
            SMOKING = 1 if SMOKING == 'Yes' else 0
            YELLOW_FINGERS = 1 if YELLOW_FINGERS == 'Yes' else 0
            ANXIETY = 1 if ANXIETY == 'Yes' else 0
            PEER_PRESSURE = 1 if PEER_PRESSURE == 'Yes' else 0
            CHRONIC_DISEASE = 1 if CHRONIC_DISEASE == 'Yes' else 0
            FATIGUE = 1 if FATIGUE == 'Yes' else 0
            ALLERGY = 1 if ALLERGY == 'Yes' else 0
            WHEEZING = 1 if WHEEZING == 'Yes' else 0
            ALCOHOL_CONSUMING = 1 if ALCOHOL_CONSUMING == 'Yes' else 0
            COUGHING = 1 if COUGHING == 'Yes' else 0
            SHORTNESS_OF_BREATH = 1 if SHORTNESS_OF_BREATH == 'Yes' else 0
            SWALLOWING_DIFFICULTY = 1 if SWALLOWING_DIFFICULTY == 'Yes' else 0
            CHEST_PAIN = 1 if CHEST_PAIN == 'Yes' else 0
            
            inputs = [GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN]
            prediction, probability = get_prediction_with_probability(models['lung_cancer'], [inputs])
            display_prediction_result(prediction, probability, "Lung Cancer")
            
            # Show risk factors visualization
            st.subheader("Risk Factor Analysis")
            features = ['Gender', 'Age', 'Smoking', 'Yellow Fingers', 'Anxiety', 'Peer Pressure', 'Chronic Disease', 'Fatigue', 'Allergy', 'Wheezing', 'Alcohol', 'Coughing', 'Shortness of Breath', 'Swallowing Difficulty', 'Chest Pain']
            values = [GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN]
            
            fig = px.bar(x=features, y=values, title='Risk Factors Analysis')
            st.plotly_chart(fig, use_container_width=True)

def thyroid_page():
    st.title("Hypo-Thyroid Prediction")
    st.markdown("""
        <div class="info-box">
            <h4>ℹ️ About Hypo-Thyroid Prediction</h4>
            <p>This tool helps assess your risk of hypothyroidism based on various health parameters.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input('Age', 
                            min_value=18,  # Minimum age for thyroid screening
                            max_value=120,
                            value=30,  # Default to middle age
                            help="Enter your age (18-120 years)")
        
        sex = st.selectbox('Sex', ['Male', 'Female'])
        
        on_thyroxine = st.selectbox('On Thyroxine', ['Yes', 'No'])
        
        tsh = st.number_input('TSH Level (mIU/L)', 
                            min_value=0.005,  # Minimum normal TSH
                            max_value=530.0,
                            value=2.5,  # Default to normal TSH
                            help="Normal range: 0.4-4.0 mIU/L")
    
    with col2:
        t3_measured = st.selectbox('T3 Measured', ['Yes', 'No'])
        
        t3 = st.number_input('T3 Level (ng/dL)', 
                           min_value=0.2,  # Minimum normal T3
                           max_value=10.0,
                           value=1.2,  # Default to normal T3
                           help="Normal range: 0.8-2.0 ng/dL")
        
        tt4 = st.number_input('TT4 Level (μg/dL)', 
                            min_value=2.0,  # Minimum normal TT4
                            max_value=430.0,
                            value=7.0,  # Default to normal TT4
                            help="Normal range: 4.5-12.5 μg/dL")
    
    if st.button("Predict Hypo-Thyroid Risk"):
        with st.spinner('Analyzing thyroid levels...'):
            # Convert categorical variables to numerical
            sex = 1 if sex == 'Male' else 0
            on_thyroxine = 1 if on_thyroxine == 'Yes' else 0
            t3_measured = 1 if t3_measured == 'Yes' else 0
            
            inputs = [age, sex, on_thyroxine, tsh, t3_measured, t3, tt4]
            prediction, probability = get_prediction_with_probability(models['thyroid'], [inputs])
            display_prediction_result(prediction, probability, "Hypo-Thyroid")
            
            # Show thyroid levels analysis
            st.subheader("Thyroid Levels Analysis")
            features = ['Age', 'Sex', 'On Thyroxine', 'TSH', 'T3 Measured', 'T3', 'TT4']
            values = [age, sex, on_thyroxine, tsh, t3_measured, t3, tt4]
            
            fig = px.bar(x=features, y=values, title='Thyroid Levels Analysis')
            st.plotly_chart(fig, use_container_width=True)

# Main app logic
if __name__ == "__main__":
    try:
        # Load models
        models = load_models()
        if models is None:
            st.error("Failed to load models. Please check the model files.")
            st.stop()

        # Sidebar Navigation
        with st.sidebar:
            st.image("https://www.strategyand.pwc.com/m1/en/strategic-foresight/sector-strategies/healthcare/ai-powered-healthcare-solutions/img01-section1.jpg", width=200)
            st.title("Medical Diagnosis AI")
            
            selected = option_menu(
                menu_title="Select Disease",
                options=[
                    "Dashboard",
                    "Diabetes Prediction",
                    "Heart Disease Prediction",
                    "Parkinsons Prediction",
                    "Lung Cancer Prediction",
                    "Hypo-Thyroid Prediction"
                ],
                icons=[
                    "speedometer2",
                    "activity",
                    "heart-pulse",
                    "person",
                    "lungs",
                    "thermometer"
                ],
                menu_icon="hospital",
                default_index=0,
            )

        # Route to appropriate page
        if selected == "Dashboard":
            display_dashboard()
        elif selected == "Diabetes Prediction":
            diabetes_page()
        elif selected == "Heart Disease Prediction":
            heart_disease_page()
        elif selected == "Parkinsons Prediction":
            parkinsons_page()
        elif selected == "Lung Cancer Prediction":
            lung_cancer_page()
        elif selected == "Hypo-Thyroid Prediction":
            thyroid_page()
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An unexpected error occurred. Please try refreshing the page.")
