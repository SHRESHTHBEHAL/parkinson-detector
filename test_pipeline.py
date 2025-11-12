"""
Complete MVP Pipeline Test Script

This script validates the entire Parkinson's Voice Detection pipeline:
1. Model loading
2. Feature extraction from audio files
3. Prediction pipeline
4. Output validation

Usage: python test_pipeline.py
"""

import os
import sys
import numpy as np
import joblib
import warnings
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import our modules
from audio_feature_extraction import extract_features, predict_parkinsons
import pandas as pd

def create_dummy_audio(filename, duration=3, sample_rate=16000, frequency=150):
    """
    Create a dummy WAV file for testing.
    
    Args:
        filename (str): Output filename
        duration (int): Duration in seconds
        sample_rate (int): Sample rate
        frequency (int): Frequency of the sine wave
    """
    try:
        import soundfile as sf
        
        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Add some noise to make it more realistic
        noise = 0.1 * np.random.normal(0, 1, len(audio_data))
        audio_data += noise
        
        # Save as WAV file
        sf.write(filename, audio_data, sample_rate)
        
        return True
        
    except ImportError:
        print("Warning: soundfile not available, creating placeholder file")
        # Create empty file as fallback
        with open(filename, 'wb') as f:
            f.write(b'RIFF' + b'\x00' * 100)  # Minimal WAV header
        return True
    except Exception as e:
        print(f"Error creating dummy audio: {e}")
        return False

def test_model_loading():
    """Test if the trained model and scaler can be loaded."""
    print("🔍 Testing Model Loading...")
    
    try:
        # Check if model files exist
        if not os.path.exists('model.pkl'):
            print("   ❌ model.pkl not found")
            return False
            
        if not os.path.exists('scaler.pkl'):
            print("   ❌ scaler.pkl not found")
            return False
        
        # Try loading the models
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        print(f"   ✅ Model loaded successfully")
        print(f"      - Model type: {type(model).__name__}")
        print(f"      - Scaler type: {type(scaler).__name__}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        return False

def test_feature_extraction(audio_files):
    """Test feature extraction from audio files."""
    print("\n🔍 Testing Feature Extraction...")
    
    successful_extractions = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        try:
            print(f"   Testing file {i}: {audio_file}")
            
            # Extract features
            features = extract_features(audio_file)
            
            # Validate output
            if features is None:
                print(f"   ❌ File {i}: Feature extraction returned None")
                continue
                
            if not isinstance(features, pd.DataFrame):
                print(f"   ❌ File {i}: Expected DataFrame, got {type(features)}")
                continue
            
            if features.empty:
                print(f"   ❌ File {i}: Feature extraction returned empty DataFrame")
                continue
            
            expected_features = 22  # Number of features expected
            actual_features = features.shape[1]
            
            if actual_features != expected_features:
                print(f"   ⚠️  File {i}: Expected {expected_features} features, got {actual_features}")
            else:
                print(f"   ✅ File {i}: Successfully extracted {actual_features} features")
            
            # Display some feature values for verification
            feature_names = list(features.columns)
            print(f"      Features: {', '.join(feature_names[:5])}...")
            
            successful_extractions += 1
            
        except Exception as e:
            print(f"   ❌ File {i}: Feature extraction failed - {e}")
    
    print(f"   📊 Feature extraction results: {successful_extractions}/{len(audio_files)} successful")
    return successful_extractions == len(audio_files)

def test_prediction_pipeline(audio_files):
    """Test the complete prediction pipeline."""
    print("\n🔍 Testing Prediction Pipeline...")
    
    successful_predictions = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        try:
            print(f"   Testing file {i}: {audio_file}")
            
            # Run prediction
            result = predict_parkinsons(audio_file)
            
            # Validate result structure
            required_keys = ['prediction', 'probability_parkinsons', 'probability_healthy', 'confidence']
            for key in required_keys:
                if key not in result:
                    print(f"   ❌ File {i}: Missing required key '{key}' in result")
                    break
            else:
                # Extract values
                prediction = result['prediction']
                parkinsons_prob = result['probability_parkinsons']
                healthy_prob = result['probability_healthy']
                confidence = result['confidence']
                
                # Validate values
                if not isinstance(prediction, str):
                    print(f"   ❌ File {i}: Expected string prediction, got {type(prediction)}")
                    continue
                
                if prediction not in ['Healthy', "Parkinson's"]:
                    # Also check for alternative formats
                    if prediction not in ['Likely Healthy', "Possible Parkinson's Risk", '0', '1']:
                        print(f"   ❌ File {i}: Unexpected prediction value: {prediction}")
                        continue
                
                # Check probability ranges
                if not (0 <= parkinsons_prob <= 1):
                    print(f"   ❌ File {i}: Invalid parkinsons probability: {parkinsons_prob}")
                    continue
                
                if not (0 <= healthy_prob <= 1):
                    print(f"   ❌ File {i}: Invalid healthy probability: {healthy_prob}")
                    continue
                
                if not (0 <= confidence <= 1):
                    print(f"   ❌ File {i}: Invalid confidence: {confidence}")
                    continue
                
                print(f"   ✅ File {i}: Prediction successful")
                print(f"      - Diagnosis: {prediction}")
                print(f"      - Parkinson's Probability: {parkinsons_prob:.3f}")
                print(f"      - Healthy Probability: {healthy_prob:.3f}")
                print(f"      - Confidence: {confidence:.3f}")
                
                successful_predictions += 1
                
        except Exception as e:
            print(f"   ❌ File {i}: Prediction failed - {e}")
    
    print(f"   📊 Prediction results: {successful_predictions}/{len(audio_files)} successful")
    return successful_predictions == len(audio_files)

def test_end_to_end_pipeline():
    """Test the complete pipeline with dummy data."""
    print("\n🔍 Testing End-to-End Pipeline...")
    
    # Create test audio files
    test_files = [
        'test_audio_1.wav',
        'test_audio_2.wav', 
        'test_audio_3.wav'
    ]
    
    try:
        print("   Creating dummy audio files...")
        
        # Create different types of dummy audio
        frequencies = [120, 180, 200]  # Different pitch frequencies
        durations = [2, 3, 4]  # Different durations
        
        for i, (filename, freq, duration) in enumerate(zip(test_files, frequencies, durations), 1):
            if not create_dummy_audio(filename, duration=duration, frequency=freq):
                print(f"   ❌ Failed to create {filename}")
                return False
            print(f"   ✅ Created {filename} ({duration}s, {freq}Hz)")
        
        # Test the complete pipeline
        print("\n   Running complete pipeline tests...")
        
        # Test feature extraction
        feature_success = test_feature_extraction(test_files)
        
        # Test prediction pipeline
        prediction_success = test_prediction_pipeline(test_files)
        
        # Clean up test files
        print("\n   Cleaning up test files...")
        for filename in test_files:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"   🗑️  Removed {filename}")
        
        # Overall result
        if feature_success and prediction_success:
            print("   ✅ End-to-end pipeline test passed")
            return True
        else:
            print("   ❌ End-to-end pipeline test failed")
            return False
            
    except Exception as e:
        print(f"   ❌ End-to-end pipeline test failed: {e}")
        return False

def main():
    """Main test function."""
    
    print("=" * 60)
    print("🧪 Parkinson's Voice Detection - MVP Pipeline Test")
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Track test results
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Model Loading
    if test_model_loading():
        tests_passed += 1
    
    # Test 2: Import validation
    print("\n🔍 Testing Module Imports...")
    try:
        import pandas as pd
        import librosa
        import streamlit
        print("   ✅ All required modules imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Module import failed: {e}")
    
    # Test 3: Feature extraction validation
    feature_files = ['validation_audio.wav']
    print("\n🔍 Testing Basic Feature Extraction...")
    try:
        create_dummy_audio('validation_audio.wav')
        features = extract_features('validation_audio.wav')
        if os.path.exists('validation_audio.wav'):
            os.remove('validation_audio.wav')
        
        if features is not None and isinstance(features, pd.DataFrame) and not features.empty:
            print("   ✅ Basic feature extraction validation passed")
            tests_passed += 1
        else:
            print("   ❌ Basic feature extraction validation failed")
    except Exception as e:
        print(f"   ❌ Basic feature extraction failed: {e}")
    
    # Test 4: End-to-end pipeline
    if test_end_to_end_pipeline():
        tests_passed += 1
    
    # Final results
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 SUCCESS: All systems functional!")
        print("✅ Model loading: PASSED")
        print("✅ Feature extraction: PASSED") 
        print("✅ Prediction pipeline: PASSED")
        print("✅ End-to-end validation: PASSED")
        print("\n🚀 The MVP is ready for deployment!")
        return True
    else:
        print("❌ FAILURE: Some tests failed")
        print("🔧 Please check the error messages above")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error during testing: {e}")
        sys.exit(1)