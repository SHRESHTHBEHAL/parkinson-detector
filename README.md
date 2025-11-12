# 🎙️ AI for Early Detection of Parkinson's Disease from Voice

An AI-powered web application that analyzes voice recordings to detect early signs of Parkinson's disease using machine learning and audio signal processing.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-Interface-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Overview

This project uses advanced audio feature extraction and machine learning to analyze voice patterns that may indicate Parkinson's disease. The application provides a user-friendly interface for recording or uploading audio samples and receiving instant predictions.

## ✨ Features

- 🎤 **Real-time Audio Recording**: Record voice samples directly in the browser
- 📁 **Audio File Upload**: Support for various audio formats
- 🔬 **Advanced Feature Extraction**: Extracts 22+ vocal features including:
  - MDVP (Multi-Dimensional Voice Program) measures
  - Jitter and Shimmer variations
  - Harmonic-to-Noise Ratio (HNR)
  - Pitch and amplitude variations
  - RPDE (Recurrence Period Density Entropy)
  - DFA (Detrended Fluctuation Analysis)
  - PPE (Pitch Period Entropy)
- 🤖 **Machine Learning Prediction**: Random Forest classifier for accurate detection
- 📊 **Feature Visualization**: Visual analysis of extracted audio features
- 🎯 **Real-time Results**: Instant prediction with confidence scores

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/SHRESHTHBEHAL/parkinson-detector.git
cd parkinson-detector
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

Launch the Gradio web interface:
```bash
python app.py
```

The application will start and provide a local URL (typically `http://127.0.0.1:7860`) that you can open in your browser.

## 📁 Project Structure

```
parkinson-detector/
├── app.py                              # Main Gradio web application
├── audio_feature_extraction.py         # Audio processing and feature extraction
├── parkinsons_voice_detection.py       # Model training and evaluation
├── model.pkl                           # Trained Random Forest model
├── scaler.pkl                          # Feature scaler for normalization
├── demo_audio.wav                      # Sample audio file for testing
├── test_download.py                    # Dataset download utility
├── test_pipeline.py                    # Comprehensive testing suite
├── requirements.txt                    # Project dependencies
└── README.md                           # This file
```

## 🔧 Usage

### Web Interface

1. **Record Audio**: Click the microphone button to record a voice sample (sustained vowel sound recommended)
2. **Upload Audio**: Or upload a pre-recorded audio file
3. **Analyze**: Click "Analyze Audio" to process the recording
4. **View Results**: See the prediction result and extracted features

### Recording Tips

For best results when recording:
- Sustain the vowel sound "aaaaa" for 3-5 seconds
- Record in a quiet environment
- Speak at a normal, comfortable volume
- Keep a consistent distance from the microphone

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_pipeline.py
```

This will validate:
- Audio file processing
- Feature extraction accuracy
- Model prediction pipeline
- End-to-end system functionality

## 📊 Model Information

- **Algorithm**: Random Forest Classifier
- **Features**: 22 acoustic and vocal characteristics
- **Training Data**: UCI Parkinson's Disease Dataset
- **Evaluation**: Cross-validation with standard metrics

### Extracted Features

The system analyzes multiple voice characteristics:
- **MDVP:Fo(Hz)**: Average vocal fundamental frequency
- **MDVP:Fhi(Hz)**: Maximum vocal fundamental frequency
- **MDVP:Flo(Hz)**: Minimum vocal fundamental frequency
- **Jitter Variations**: Frequency variation measures
- **Shimmer Variations**: Amplitude variation measures
- **NHR & HNR**: Noise-to-harmonics ratios
- **RPDE**: Nonlinear dynamical complexity
- **DFA**: Signal fractal scaling exponent
- **PPE**: Pitch period entropy

## 📦 Dependencies

Core libraries:
- **gradio**: Web interface
- **librosa**: Audio processing
- **scikit-learn**: Machine learning
- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **matplotlib**: Visualization
- **joblib**: Model serialization
- **sounddevice**: Audio recording

See `requirements.txt` for complete list.

## ⚠️ Disclaimer

This application is for **educational and research purposes only**. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for proper medical evaluation and diagnosis.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Shreshth Behal**
- GitHub: [@SHRESHTHBEHAL](https://github.com/SHRESHTHBEHAL)

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Parkinson's Disease Dataset
- The open-source community for the amazing libraries used in this project
- Medical researchers advancing Parkinson's disease detection methods

## 📚 References

- Little, M. A., McSharry, P. E., Roberts, S. J., Costello, D. A., & Moroz, I. M. (2007). Exploiting nonlinear recurrence and fractal scaling properties for voice disorder detection. *BioMedical Engineering OnLine*, 6(1), 23.
- UCI Machine Learning Repository: [Parkinson's Disease Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)

## 🔮 Future Enhancements

- [ ] Support for multiple languages
- [ ] Mobile app development
- [ ] Integration with more advanced deep learning models
- [ ] Longitudinal tracking and monitoring
- [ ] Additional voice biomarkers
- [ ] Cloud deployment options

---

**Note**: If you find this project useful, please consider giving it a ⭐ on GitHub!
