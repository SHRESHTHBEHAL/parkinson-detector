"""
Parkinson's Voice Detection - Modern Responsive UI

A completely redesigned Streamlit web application with modern, minimal UI
that adapts to all screen sizes while maintaining all existing functionality.

Built by Shreshth Behal
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from audio_feature_extraction import extract_features, predict_parkinsons
import io
import warnings
import os
import soundfile as sf
from datetime import datetime

# Suppress warnings for cleaner UI
warnings.filterwarnings('ignore')

# Configure page settings
st.set_page_config(
    page_title="Parkinson's Voice Detection",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Language support
LANGUAGES = {
    "English": {
        "title": "Parkinson's Voice Detection",
        "subtitle": "AI-Powered Early Screening Tool",
        "upload_title": "Upload Audio File",
        "upload_help": "Upload a .wav file for analysis",
        "demo_mode": "Try Demo Sample",
        "language": "Language",
        "analyze": "Analyze Voice",
        "results": "Analysis Results",
        "diagnosis": "Diagnosis",
        "confidence": "Confidence",
        "risk_assessment": "Risk Assessment",
        "audio_visualization": "Audio Visualization",
        "how_it_works": "How It Works",
        "recording_tips": "Recording Tips",
        "clear_results": "Clear Results",
        "demo_description": "Use our sample recording to test the app instantly",
        "healthy": "Likely Healthy",
        "parkinsons": "Possible Parkinson's Risk",
        "disclaimer": "Important Medical Disclaimer",
        "footer_credits": "Built by Shreshth Behal"
    },
    "Hindi": {
        "title": "पार्किंसन की आवाज़ पहचान",
        "subtitle": "AI-संचालित प्रारंभिक स्क्रीनिंग टूल",
        "upload_title": "ऑडियो फ़ाइल अपलोड करें",
        "upload_help": "विश्लेषण के लिए .wav फ़ाइल अपलोड करें",
        "demo_mode": "डेमो नमूना आज़माएं",
        "language": "भाषा",
        "analyze": "आवाज़ विश्लेषण करें",
        "results": "विश्लेषण परिणाम",
        "diagnosis": "निदान",
        "confidence": "विश्वसनीयता",
        "risk_assessment": "जोखिम मूल्यांकन",
        "audio_visualization": "ऑडियो दृश्य",
        "how_it_works": "यह कैसे काम करता है",
        "recording_tips": "रिकॉर्डिंग सुझाव",
        "clear_results": "परिणाम साफ़ करें",
        "demo_description": "ऐप का तुरंत परीक्षण करने के लिए हमारा नमूना रिकॉर्डिंग उपयोग करें",
        "healthy": "संभवतः स्वस्थ",
        "parkinsons": "संभावित पार्किंसन का जोखिम",
        "disclaimer": "महत्वपूर्ण चिकित्सा अस्वीकरण",
        "footer_credits": "श्रेष्ठ बहल द्वारा निर्मित"
    }
}

# Custom CSS for modern, responsive design
st.markdown("""
<style>
    /* Root Variables */
    :root {
        --primary-color: #00BFA6;
        --background-dark: #1a1a1a;
        --background-card: #2d2d2d;
        --background-hover: #3d3d3d;
        --text-primary: #ffffff;
        --text-secondary: #b0b0b0;
        --text-muted: #808080;
        --border-color: #404040;
        --shadow-light: 0 2px 8px rgba(0, 191, 166, 0.1);
        --shadow-medium: 0 4px 16px rgba(0, 191, 166, 0.15);
        --shadow-heavy: 0 8px 32px rgba(0, 191, 166, 0.2);
        --border-radius: 12px;
        --border-radius-large: 16px;
        --transition: all 0.3s ease;
    }

    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, var(--background-dark) 0%, #2d2d2d 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container */
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 1rem;
    }

    /* Top Bar */
    .top-bar {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background: rgba(26, 26, 26, 0.98);
        backdrop-filter: blur(10px);
        border-bottom: 2px solid var(--primary-color);
        padding: 2rem;
        z-index: 1000;
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
        min-height: 120px;
    }

    .title-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        max-width: 100%;
    }

    .app-title {
        color: var(--text-primary);
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
        text-shadow: 0 3px 6px rgba(0, 0, 0, 0.5);
        letter-spacing: 0.5px;
    }

    .app-subtitle {
        color: var(--text-secondary);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        line-height: 1.3;
        font-weight: 400;
        opacity: 0.9;
    }

    /* Content Spacer */
    .content-spacer {
        height: 140px;
    }

    /* Cards */
    .card {
        background: var(--background-card);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius-large);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-light);
        transition: var(--transition);
    }

    .card:hover {
        box-shadow: var(--shadow-medium);
        border-color: var(--primary-color);
    }

    .card-header {
        margin-bottom: 1.5rem;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 1rem;
    }

    .card-title {
        color: var(--text-primary);
        font-size: 1.3rem;
        font-weight: 600;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Upload Section */
    .upload-section {
        background: var(--background-card);
        border: 2px dashed var(--border-color);
        border-radius: var(--border-radius-large);
        padding: 2rem;
        text-align: center;
        transition: var(--transition);
        cursor: pointer;
    }

    .upload-section:hover {
        border-color: var(--primary-color);
        background: var(--background-hover);
    }

    .upload-icon {
        font-size: 3rem;
        color: var(--primary-color);
        margin-bottom: 1rem;
    }

    /* Results Section */
    .result-card {
        background: linear-gradient(135deg, var(--background-card) 0%, #2a2a2a 100%);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius-large);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-medium);
    }

    .result-healthy {
        border-left: 4px solid #4CAF50;
    }

    .result-parkinsons {
        border-left: 4px solid #f44336;
    }

    /* Metrics */
    .metric-container {
        background: var(--background-hover);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid var(--border-color);
        transition: var(--transition);
    }

    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-medium);
    }

    /* Progress Bar */
    .custom-progress {
        background: var(--background-hover);
        border-radius: var(--border-radius);
        height: 8px;
        margin: 1rem 0;
        overflow: hidden;
    }

    .custom-progress-bar {
        height: 100%;
        background: linear-gradient(90deg, var(--primary-color) 0%, #00E5CC 100%);
        border-radius: var(--border-radius);
        transition: width 0.8s ease;
    }

    /* Buttons */
    .stButton > button {
        background: var(--primary-color);
        color: white;
        border: none;
        border-radius: var(--border-radius);
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 500;
        transition: var(--transition);
        min-height: 48px;
        box-shadow: var(--shadow-light);
    }

    .stButton > button:hover {
        background: #00A896;
        transform: translateY(-2px);
        box-shadow: var(--shadow-medium);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Demo Button */
    .demo-button {
        background: linear-gradient(135deg, var(--primary-color) 0%, #00E5CC 100%);
        color: white;
        border: none;
        border-radius: var(--border-radius);
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: var(--transition);
        cursor: pointer;
        width: 100%;
        margin: 1rem 0;
    }

    .demo-button:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-heavy);
    }

    /* Risk Score */
    .risk-score {
        background: linear-gradient(135deg, var(--primary-color) 0%, #00E5CC 100%);
        color: white;
        border-radius: var(--border-radius-large);
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }

    .risk-score-high {
        background: linear-gradient(135deg, #f44336 0%, #e57373 100%);
    }

    .risk-score-medium {
        background: linear-gradient(135deg, #ff9800 0%, #ffb74d 100%);
    }

    .risk-score-low {
        background: linear-gradient(135deg, #4CAF50 0%, #81C784 100%);
    }

    /* Footer */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(26, 26, 26, 0.95);
        backdrop-filter: blur(10px);
        border-top: 1px solid var(--border-color);
        padding: 1rem 2rem;
        text-align: center;
        color: var(--text-muted);
        font-size: 0.9rem;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .main-container {
            padding: 0.5rem;
        }

        .top-bar {
            padding: 1.5rem 1rem;
            min-height: 100px;
        }

        .app-title {
            font-size: 1.8rem;
        }

        .app-subtitle {
            font-size: 1rem;
        }

        .content-spacer {
            height: 120px;
        }

        .card {
            padding: 1.5rem;
            margin: 0.5rem 0;
        }

        .card-title {
            font-size: 1.1rem;
        }

        .upload-section {
            padding: 1.5rem;
        }

        .upload-icon {
            font-size: 2rem;
        }

        .stButton > button {
            padding: 0.75rem 1.5rem;
            font-size: 0.9rem;
        }

        .demo-button {
            padding: 0.75rem 1.5rem;
            font-size: 1rem;
        }
    }

    @media (max-width: 480px) {
        .top-bar {
            padding: 0.5rem;
        }

        .card {
            padding: 1rem;
        }

        .upload-section {
            padding: 1rem;
        }

        .footer {
            padding: 0.75rem 1rem;
            font-size: 0.8rem;
        }
    }

    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--background-dark);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary-color);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #00A896;
    }

    /* Success/Error Messages */
    .stSuccess {
        background: rgba(76, 175, 80, 0.1);
        border: 1px solid #4CAF50;
        border-radius: var(--border-radius);
    }

    .stError {
        background: rgba(244, 67, 54, 0.1);
        border: 1px solid #f44336;
        border-radius: var(--border-radius);
    }
</style>
""", unsafe_allow_html=True)

def get_text(key, lang="English"):
    """Get translated text for the selected language."""
    return LANGUAGES.get(lang, LANGUAGES["English"]).get(key, key)

def load_model():
    """Load the trained model and scaler."""
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure model.pkl and scaler.pkl are in the same directory.")
        return None, None

def create_demo_audio():
    """Create a demo audio sample for testing."""
    try:
        duration = 3
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create realistic voice-like signal
        f0 = 150
        audio_data = 0.3 * np.sin(2 * np.pi * f0 * t)
        audio_data += 0.15 * np.sin(2 * np.pi * f0 * 2 * t)
        audio_data += 0.1 * np.sin(2 * np.pi * f0 * 3 * t)
        
        # Add modulation
        modulation = 0.5 * (1 + 0.3 * np.sin(2 * np.pi * 2 * t))
        audio_data *= modulation
        
        # Add noise
        noise = 0.05 * np.random.normal(0, 1, len(audio_data))
        audio_data += noise
        
        # Normalize
        audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
        
        return audio_data, sr
    except Exception as e:
        st.error(f"Error creating demo audio: {str(e)}")
        return None, None

def save_demo_audio():
    """Save demo audio to a temporary file."""
    try:
        audio_data, sr = create_demo_audio()
        if audio_data is not None:
            demo_filename = "demo_audio.wav"
            sf.write(demo_filename, audio_data, sr)
            return demo_filename
        return None
    except Exception as e:
        st.error(f"Error saving demo audio: {str(e)}")
        return None

def create_waveform(audio_data, sample_rate):
    """Create a waveform visualization."""
    try:
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Use librosa display for better visualization
        import librosa.display
        librosa.display.waveshow(audio_data, sr=sample_rate, ax=ax, color='#00BFA6', alpha=0.8)
        
        ax.set_xlabel('Time (seconds)', fontsize=12, color='white')
        ax.set_ylabel('Amplitude', fontsize=12, color='white')
        ax.set_title('Voice Waveform Analysis', fontsize=14, fontweight='bold', color='white')
        ax.grid(True, alpha=0.3, color='white')
        
        # Style the plot - fix transparency issues
        fig.patch.set_facecolor((0.1, 0.1, 0.1, 0.0))  # RGBA tuple instead of 'transparent'
        ax.set_facecolor((0.1, 0.1, 0.1, 0.0))  # RGBA tuple instead of 'transparent'
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.tick_params(colors='white')
        
        # Set axis colors
        for spine in ax.spines.values():
            spine.set_alpha(0.5)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating waveform: {str(e)}")
        return None

def create_spectrogram(audio_data, sample_rate):
    """Create a spectrogram visualization."""
    try:
        D = librosa.stft(audio_data)
        magnitude = np.abs(D)
        magnitude_db = librosa.amplitude_to_db(magnitude)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        im = librosa.display.specshow(magnitude_db, sr=sample_rate, x_axis='time', y_axis='hz', ax=ax, cmap='plasma')
        cbar = plt.colorbar(im, ax=ax, format='%+2.0f dB')
        cbar.set_label('Magnitude (dB)', rotation=270, labelpad=20, color='white')
        cbar.ax.tick_params(colors='white')
        
        ax.set_title('Audio Spectrogram Analysis', fontsize=14, fontweight='bold', color='white')
        ax.set_xlabel('Time (seconds)', fontsize=12, color='white')
        ax.set_ylabel('Frequency (Hz)', fontsize=12, color='white')
        ax.tick_params(colors='white')
        
        # Style the plot - fix transparency issues
        fig.patch.set_facecolor((0.1, 0.1, 0.1, 0.0))  # RGBA tuple instead of 'transparent'
        ax.set_facecolor((0.1, 0.1, 0.1, 0.0))  # RGBA tuple instead of 'transparent'
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating spectrogram: {str(e)}")
        return None

def get_risk_color(probability):
    """Get risk color based on probability."""
    if probability < 0.3:
        return "#4CAF50"  # Green for low risk
    elif probability < 0.7:
        return "#ff9800"  # Orange for medium risk
    else:
        return "#f44336"  # Red for high risk

def get_risk_level(probability):
    """Get risk level description."""
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.7:
        return "Moderate Risk"
    else:
        return "High Risk"

def main():
    """Main Streamlit application with modern responsive UI."""
    
    # Initialize session state
    if 'language' not in st.session_state:
        st.session_state['language'] = "English"
    if 'demo_loaded' not in st.session_state:
        st.session_state['demo_loaded'] = False
    
    # Fixed top bar
    st.markdown("""
    <div class="top-bar">
        <div class="title-container">
            <h1 class="app-title">🎤 Parkinson's Voice Detection</h1>
            <p class="app-subtitle">AI-Powered Early Screening Tool</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Content spacer
    st.markdown('<div class="content-spacer"></div>', unsafe_allow_html=True)
    
    # Load model
    model, scaler = load_model()
    if model is None:
        st.stop()
    
    # Main container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Responsive layout
    col1, col2 = st.columns([1, 2] if st.session_state.get('screen_width', 1200) > 768 else [1])
    
    with col1:
        # Upload Section
        st.markdown(f"""
        <div class="card fade-in-up">
            <div class="card-header">
                <h2 class="card-title">📁 {get_text('upload_title')}</h2>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Demo mode
        st.markdown(f"""
        <div class="card">
            <h3 class="card-title">🚀 {get_text('demo_mode')}</h3>
            <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">{get_text('demo_description')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(get_text('demo_mode'), key="demo_btn", use_container_width=True):
            with st.spinner("Loading demo sample..."):
                try:
                    demo_file = save_demo_audio()
                    if demo_file:
                        st.session_state['demo_file'] = demo_file
                        st.session_state['demo_loaded'] = True
                        st.success("Demo sample loaded successfully! Click 'Analyze Voice' to test.")
                        st.rerun()
                    else:
                        st.error("Failed to load demo sample")
                except Exception as e:
                    st.error(f"Demo error: {str(e)}")
        
        # File upload
        st.markdown(f"""
        <div class="card">
            <h3 class="card-title">📤 {get_text('upload_title')}</h3>
            <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">{get_text('upload_help')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        audio_file = st.file_uploader(
            "Upload your audio file",
            type=['wav'],
            help="Upload a clean voice recording in WAV format",
            label_visibility="collapsed"
        )
        
        # Collapsible sections
        with st.expander("🎙️ " + get_text('recording_tips'), expanded=False):
            st.markdown("""
            - **Speak clearly and naturally** (10-30 seconds)
            - **Use minimal background noise**
            - **Record in a quiet environment**
            - **Ensure good audio quality**
            - **Speak at normal volume**
            """)
        
        with st.expander("🔬 " + get_text('how_it_works'), expanded=False):
            st.markdown("""
            1. **Audio Preprocessing** - Resample to 16kHz, trim silence, normalize
            2. **Feature Extraction** - Extract 22 voice biomarkers including pitch, jitter, shimmer
            3. **AI Analysis** - RandomForest model evaluates voice patterns
            4. **Risk Assessment** - Calculate probability of Parkinson's indicators
            5. **Visualization** - Display waveforms and spectrograms for analysis
            """)
    
    with col2:
        # Handle demo mode or uploaded file
        if st.session_state.get('demo_loaded') and st.session_state.get('demo_file'):
            demo_file = st.session_state['demo_file']
            if os.path.exists(demo_file):
                try:
                    # Load demo audio
                    audio_data, sample_rate = librosa.load(demo_file, sr=16000)
                    st.success("🎯 Demo sample ready!")
                    st.info(f"📊 Duration: {len(audio_data)/sample_rate:.2f} seconds")
                    
                    # Analysis button
                    if st.button(get_text('analyze'), key="analyze_demo", type="primary", use_container_width=True):
                        with st.spinner("Analyzing demo audio..."):
                            try:
                                result = predict_parkinsons(demo_file)
                                st.session_state['analysis_result'] = result
                                st.session_state['audio_data'] = audio_data
                                st.session_state['sample_rate'] = sample_rate
                                st.rerun()
                            except Exception as e:
                                st.error(f"Analysis failed: {str(e)}")
                    
                    # Clear demo button
                    if st.button("🗑️ Clear Demo", key="clear_demo", use_container_width=True):
                        if os.path.exists(demo_file):
                            os.remove(demo_file)
                        keys_to_remove = ['demo_file', 'demo_loaded']
                        for key in keys_to_remove:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error loading demo: {str(e)}")
        
        elif audio_file is not None:
            try:
                # Process uploaded file
                audio_bytes = audio_file.read()
                temp_filename = f"temp_upload_{audio_file.name}"
                
                with open(temp_filename, 'wb') as f:
                    f.write(audio_bytes)
                
                audio_data, sample_rate = librosa.load(temp_filename, sr=16000)
                
                st.success("✅ Audio file loaded successfully!")
                st.info(f"📊 Duration: {len(audio_data)/sample_rate:.2f} seconds")
                
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                
                # Analysis button
                if st.button(get_text('analyze'), key="analyze_upload", type="primary", use_container_width=True):
                    with st.spinner("Processing audio and extracting features..."):
                        try:
                            temp_filename = f"temp_analysis_{audio_file.name}"
                            with open(temp_filename, 'wb') as f:
                                f.write(audio_bytes)
                            
                            result = predict_parkinsons(temp_filename)
                            
                            if os.path.exists(temp_filename):
                                os.remove(temp_filename)
                            
                            st.session_state['analysis_result'] = result
                            st.session_state['audio_data'] = audio_data
                            st.session_state['sample_rate'] = sample_rate
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Analysis failed: {str(e)}")
                            if os.path.exists(temp_filename):
                                os.remove(temp_filename)
            
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
        
        # Analysis Results
        if 'analysis_result' in st.session_state:
            result = st.session_state['analysis_result']
            audio_data = st.session_state['audio_data']
            sample_rate = st.session_state['sample_rate']
            
            parkinsons_prob = result['probability_parkinsons']
            healthy_prob = result['probability_healthy']
            confidence = result['confidence']
            
            # Results card
            result_class = "result-parkinsons" if parkinsons_prob > 0.5 else "result-healthy"
            
            st.markdown(f"""
            <div class="result-card {result_class} fade-in-up">
                <div class="card-header">
                    <h2 class="card-title">📊 {get_text('results')}</h2>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics row
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown(f"""
                <div class="metric-container">
                    <h4 style="color: var(--text-secondary); margin: 0 0 0.5rem 0; font-size: 0.9rem;">{get_text('diagnosis')}</h4>
                    <h3 style="color: var(--text-primary); margin: 0; font-size: 1.4rem; font-weight: 600;">
                        {'✅ ' + get_text('healthy') if parkinsons_prob <= 0.5 else '⚠️ ' + get_text('parkinsons')}
                    </h3>
                    <p style="color: var(--text-muted); margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                        {parkinsons_prob:.1%} Risk
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f"""
                <div class="metric-container">
                    <h4 style="color: var(--text-secondary); margin: 0 0 0.5rem 0; font-size: 0.9rem;">{get_text('confidence')}</h4>
                    <h3 style="color: var(--primary-color); margin: 0; font-size: 1.4rem; font-weight: 600;">
                        {confidence:.1%}
                    </h3>
                    <div class="custom-progress">
                        <div class="custom-progress-bar" style="width: {confidence * 100}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_c:
                risk_color = get_risk_color(parkinsons_prob)
                risk_level = get_risk_level(parkinsons_prob)
                
                st.markdown(f"""
                <div class="metric-container">
                    <h4 style="color: var(--text-secondary); margin: 0 0 0.5rem 0; font-size: 0.9rem;">Risk Level</h4>
                    <h3 style="color: {risk_color}; margin: 0; font-size: 1.4rem; font-weight: 600;">
                        {risk_level}
                    </h3>
                    <p style="color: var(--text-muted); margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                        {parkinsons_prob:.1%} Probability
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Large risk score display
            risk_class = "risk-score-high" if parkinsons_prob > 0.7 else ("risk-score-medium" if parkinsons_prob > 0.3 else "risk-score-low")
            
            st.markdown(f"""
            <div class="risk-score {risk_class} fade-in-up">
                <h2 style="margin: 0; font-size: 3rem; font-weight: 700;">{parkinsons_prob:.1%}</h2>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">Parkinson's Risk Score</p>
                <p style="margin: 0; font-size: 1rem; opacity: 0.8;">{risk_level}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed breakdown
            st.markdown("### 📈 Detailed Breakdown")
            prob_data = {
                get_text('healthy'): [f"{healthy_prob:.1%}"],
                get_text('parkinsons'): [f"{parkinsons_prob:.1%}"]
            }
            
            prob_df = pd.DataFrame(prob_data, index=['Probability'])
            st.dataframe(prob_df, use_container_width=True)
            
            # Visualization tabs
            st.markdown(f"### {get_text('audio_visualization')}")
            
            tab1, tab2 = st.tabs(["🎵 Waveform", "🌈 Spectrogram"])
            
            with tab1:
                waveform_fig = create_waveform(audio_data, sample_rate)
                if waveform_fig:
                    st.pyplot(waveform_fig, use_container_width=True)
                    plt.close(waveform_fig)
            
            with tab2:
                spectrogram_fig = create_spectrogram(audio_data, sample_rate)
                if spectrogram_fig:
                    st.pyplot(spectrogram_fig, use_container_width=True)
                    plt.close(spectrogram_fig)
            
            # Clear results
            if st.button(get_text('clear_results'), use_container_width=True):
                keys_to_remove = ['analysis_result', 'audio_data', 'sample_rate']
                for key in keys_to_remove:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("---")
    with st.expander("⚠️ " + get_text('disclaimer'), expanded=False):
        st.markdown(f"""
        **This is a screening tool only and is NOT a medical diagnosis.**
        
        • This AI system analyzes voice patterns to identify potential risk factors
        • Results should not replace professional medical consultation  
        • If you have concerns about Parkinson's disease, please consult a healthcare professional
        • Early detection and proper medical evaluation are essential for the best outcomes
        • This tool is intended for educational and research purposes
        """)
    
    # Fixed footer
    st.markdown(f"""
    <div class="footer">
        🛠️ Built by Shreshth Behal • {datetime.now().strftime('%Y-%m-%d %H:%M')} • Powered by Machine Learning
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()