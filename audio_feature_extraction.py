"""
Audio Feature Extraction for Parkinson's Disease Detection

This script extracts voice features from audio files that are compatible
with the UCI Parkinson's dataset and trained RandomForest model.

Features extracted:
- Fundamental frequency (pitch): MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz)
- Jitter measures: MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP
- Shimmer measures: MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA
- Harmonic-to-noise ratio: NHR, HNR
- Other voice features: RPDE, D2, DFA, spread1, spread2, PPE
"""

import librosa
import numpy as np
import pandas as pd
import warnings
from scipy import stats
from scipy.signal import find_peaks
import os

def load_and_preprocess_audio(file_path, sr=16000):
    """
    Load and preprocess audio file with librosa.
    
    Args:
        file_path (str): Path to the audio file
        sr (int): Target sampling rate (default: 16000)
        
    Returns:
        tuple: (audio_signal, sampling_rate)
    """
    try:
        # Load audio file
        y, orig_sr = librosa.load(file_path, sr=sr)
        
        # Ensure audio is not too short
        if len(y) < sr:  # Less than 1 second
            warnings.warn(f"Short audio file detected ({len(y)/orig_sr:.2f}s). Consider using longer recordings.")
        
        # Trim silence from the beginning and end
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        # Normalize the signal
        y_normalized = librosa.util.normalize(y_trimmed)
        
        return y_normalized, orig_sr
        
    except Exception as e:
        raise ValueError(f"Error loading audio file {file_path}: {str(e)}")

def extract_fundamental_frequency(y, sr):
    """
    Extract fundamental frequency (pitch) features.
    
    Args:
        y (np.array): Audio signal
        sr (int): Sampling rate
        
    Returns:
        tuple: (fo_mean, fo_max, fo_min) in Hz
    """
    try:
        # Extract fundamental frequency (F0)
        fo = librosa.piptrack(y=y, sr=sr, threshold=0.1)
        
        # Get pitch values
        pitch_values = []
        for t in range(fo[1].shape[1]):
            index = fo[1][:, t].argmax()
            pitch = fo[0][index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) == 0:
            warnings.warn("No pitch detected, using default values")
            return 120.0, 200.0, 80.0  # Default values
        
        return np.mean(pitch_values), np.max(pitch_values), np.min(pitch_values)
        
    except Exception as e:
        warnings.warn(f"Error extracting pitch: {str(e)}")
        return 120.0, 200.0, 80.0

def extract_jitter_features(y, sr):
    """
    Extract jitter measures (cycle-to-cycle frequency variations).
    
    Args:
        y (np.array): Audio signal
        sr (int): Sampling rate
        
    Returns:
        tuple: (jitter_percent, jitter_abs, rap, ppq, ddp)
    """
    try:
        # Extract pitch using pyin (more reliable for jitter calculation)
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
        
        # Remove unvoiced frames
        f0_voiced = f0[voiced_flag]
        
        if len(f0_voiced) < 2:
            warnings.warn("Insufficient voiced frames for jitter calculation")
            return 0.01, 0.0001, 0.01, 0.01, 0.03
        
        # Calculate cycle-to-cycle differences
        diff = np.diff(f0_voiced)
        
        # Jitter percentage (relative jitter)
        jitter_percent = np.mean(np.abs(diff)) / np.mean(f0_voiced) * 100
        
        # Absolute jitter
        jitter_abs = np.mean(np.abs(diff))
        
        # RAP (Relative Average Perturbation) - average of differences over 3 cycles
        if len(diff) >= 3:
            rap = np.mean(np.abs([diff[i:i+3] for i in range(len(diff)-2)])) / np.mean(f0_voiced) * 100
        else:
            rap = jitter_percent
        
        # PPQ (Period Perturbation Quotient) - similar to RAP but normalized
        ppq = rap  # Simplified calculation
        
        # DDP (Difference of Differences of Periods) - average absolute difference of consecutive differences
        if len(diff) >= 2:
            ddp = np.mean(np.abs(np.diff(diff)))
        else:
            ddp = jitter_abs
        
        return jitter_percent, jitter_abs, rap, ppq, ddp
        
    except Exception as e:
        warnings.warn(f"Error calculating jitter: {str(e)}")
        return 0.01, 0.0001, 0.01, 0.01, 0.03

def extract_shimmer_features(y, sr):
    """
    Extract shimmer measures (cycle-to-cycle amplitude variations).
    
    Args:
        y (np.array): Audio signal
        sr (int): Sampling rate
        
    Returns:
        tuple: (shimmer, shimmer_db, apq3, apq5, mdvp_apq, dda)
    """
    try:
        # Extract amplitude envelope
        stft = librosa.stft(y)
        amplitude = np.abs(stft)
        
        # Get RMS energy for each frame
        rms = librosa.feature.rms(y=y)[0]
        
        if len(rms) < 2:
            warnings.warn("Insufficient frames for shimmer calculation")
            return 0.04, 0.4, 0.02, 0.02, 0.02, 0.1
        
        # Calculate cycle-to-cycle amplitude differences
        amp_diff = np.diff(rms)
        
        # Shimmer (relative shimmer)
        shimmer = np.mean(np.abs(amp_diff)) / np.mean(rms) * 100
        
        # Shimmer in dB
        shimmer_db = 20 * np.log10((np.mean(rms) + np.abs(np.mean(amp_diff))) / np.mean(rms))
        
        # APQ3 (Amplitude Perturbation Quotient) - average of differences over 3 cycles
        if len(amp_diff) >= 3:
            apq3 = np.mean(np.abs([amp_diff[i:i+3] for i in range(len(amp_diff)-2)])) / np.mean(rms) * 100
        else:
            apq3 = shimmer
        
        # APQ5 (Amplitude Perturbation Quotient) - over 5 cycles
        if len(amp_diff) >= 5:
            apq5 = np.mean(np.abs([amp_diff[i:i+5] for i in range(len(amp_diff)-4)])) / np.mean(rms) * 100
        else:
            apq5 = shimmer
        
        # MDVP APQ (similar to APQ3)
        mdvp_apq = apq3
        
        # DDA (Amplitude difference of differences)
        if len(amp_diff) >= 2:
            dda = np.mean(np.abs(np.diff(amp_diff)))
        else:
            dda = shimmer / 100
        
        return shimmer, shimmer_db, apq3, apq5, mdvp_apq, dda
        
    except Exception as e:
        warnings.warn(f"Error calculating shimmer: {str(e)}")
        return 0.04, 0.4, 0.02, 0.02, 0.02, 0.1

def extract_mfcc_features(y, sr):
    """
    Extract MFCC features and derive additional voice quality measures.
    
    Args:
        y (np.array): Audio signal
        sr (int): Sampling rate
        
    Returns:
        tuple: (hnr, nhr, rpde, d2, dfa, spread1, spread2, ppe)
    """
    try:
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        
        # Harmonic-to-noise ratio estimation
        # Using spectral tilt as a proxy for HNR
        S = np.abs(librosa.stft(y))
        harmonic_power = np.mean(S**2)
        noise_power = np.var(S)  # Simple noise estimation
        
        hnr = 10 * np.log10(harmonic_power / (noise_power + 1e-10))
        hnr = max(hnr, 0)  # Ensure non-negative
        
        # Noise-to-harmonic ratio (inverse of HNR)
        nhr = 1 / (hnr + 1e-10)
        
        # RPDE (Recurrence Period Density Entropy)
        # Simplified implementation
        if len(y) > 100:
            # Create a simple binary voice activity signal
            rms = librosa.feature.rms(y=y)[0]
            threshold = np.mean(rms) * 0.5
            vuv_signal = (rms > threshold).astype(int)
            
            # Calculate RPDE
            rpde = np.std(vuv_signal) / (np.mean(vuv_signal) + 1e-10)
        else:
            rpde = 0.1
        
        # D2 (Correlation dimension) - complexity measure
        # Using spectral centroid variance as proxy
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        d2 = np.var(spectral_centroids) / (np.mean(spectral_centroids) + 1e-10)
        
        # DFA (Detrended Fluctuation Analysis) - fractal scaling exponent
        # Simplified using signal variance
        dfa = np.std(y) / (np.mean(np.abs(y)) + 1e-10)
        
        # Spread1 and Spread2 (frequency spread measures)
        # Using spectral bandwidth and spectral rolloff variance
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        spread1 = np.std(spectral_bandwidth) / (np.mean(spectral_bandwidth) + 1e-10)
        spread2 = np.std(spectral_rolloff) / (np.mean(spectral_rolloff) + 1e-10)
        
        # PPE (Pitch Period Entropy) - entropy of pitch period distribution
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
        f0_voiced = f0[~np.isnan(f0)]
        
        if len(f0_voiced) > 10:
            # Create histogram of pitch periods
            pitch_periods = 1 / (f0_voiced + 1e-10)
            hist, _ = np.histogram(pitch_periods, bins=10)
            hist = hist / (np.sum(hist) + 1e-10)
            
            # Calculate entropy
            ppe = -np.sum(hist * np.log2(hist + 1e-10))
        else:
            ppe = 1.0
        
        return hnr, nhr, rpde, d2, dfa, spread1, spread2, ppe
        
    except Exception as e:
        warnings.warn(f"Error calculating MFCC-derived features: {str(e)}")
        return 10.0, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 1.0

def extract_features(file_path):
    """
    Extract all voice features from an audio file compatible with the Parkinson's detection model.
    
    Args:
        file_path (str): Path to the audio file (.wav format preferred)
        
    Returns:
        pd.DataFrame: DataFrame with extracted features matching the model input format
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        # Load and preprocess audio
        y, sr = load_and_preprocess_audio(file_path)
        
        # Extract features
        # Fundamental frequency
        fo_mean, fo_max, fo_min = extract_fundamental_frequency(y, sr)
        
        # Jitter measures
        jitter_pct, jitter_abs, rap, ppq, ddp = extract_jitter_features(y, sr)
        
        # Shimmer measures
        shimmer, shimmer_db, apq3, apq5, mdvp_apq, dda = extract_shimmer_features(y, sr)
        
        # Harmonic-to-noise ratio and other features
        hnr, nhr, rpde, d2, dfa, spread1, spread2, ppe = extract_mfcc_features(y, sr)
        
        # Create feature dictionary in EXACT order matching the UCI dataset
        features = {
            'MDVP:Fo(Hz)': fo_mean,
            'MDVP:Fhi(Hz)': fo_max,
            'MDVP:Flo(Hz)': fo_min,
            'MDVP:Jitter(%)': jitter_pct,
            'MDVP:Jitter(Abs)': jitter_abs,
            'MDVP:RAP': rap,
            'MDVP:PPQ': ppq,
            'Jitter:DDP': ddp,
            'MDVP:Shimmer': shimmer,
            'MDVP:Shimmer(dB)': shimmer_db,
            'Shimmer:APQ3': apq3,
            'Shimmer:APQ5': apq5,
            'MDVP:APQ': mdvp_apq,
            'Shimmer:DDA': dda,
            'NHR': nhr,
            'HNR': hnr,
            'RPDE': rpde,
            'DFA': dfa,
            'spread1': spread1,
            'spread2': spread2,
            'D2': d2,
            'PPE': ppe
        }
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        print(f"Successfully extracted features from {file_path}")
        print(f"Audio duration: {len(y)/sr:.2f} seconds")
        print(f"Sample rate: {sr} Hz")
        
        return features_df
        
    except Exception as e:
        raise RuntimeError(f"Error extracting features from {file_path}: {str(e)}")

def predict_parkinsons(file_path, model_path='model.pkl', scaler_path='scaler.pkl'):
    """
    Extract features from audio file and predict Parkinson's disease probability.
    
    Args:
        file_path (str): Path to the audio file
        model_path (str): Path to the trained model file
        scaler_path (str): Path to the scaler file
        
    Returns:
        dict: Prediction results with probability and confidence
    """
    try:
        import joblib
        
        # Load model and scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Extract features
        features_df = extract_features(file_path)
        
        # Scale features
        features_scaled = scaler.transform(features_df)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        result = {
            'prediction': 'Parkinson\'s' if prediction == 1 else 'Healthy',
            'probability_parkinsons': probability[1],
            'probability_healthy': probability[0],
            'confidence': max(probability),
            'features': features_df
        }
        
        print(f"\nPrediction Results:")
        print(f"Diagnosis: {result['prediction']}")
        print(f"Parkinson's Probability: {result['probability_parkinsons']:.3f}")
        print(f"Healthy Probability: {result['probability_healthy']:.3f}")
        print(f"Confidence: {result['confidence']:.3f}")
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"Error making prediction: {str(e)}")

# Example usage and testing
if __name__ == "__main__":
    print("Audio Feature Extraction for Parkinson's Detection")
    print("=" * 50)
    
    # Test with a sample file if available
    test_files = [
        "test_audio.wav",
        "sample.wav", 
        "voice_sample.wav"
    ]
    
    found_test_file = None
    for test_file in test_files:
        if os.path.exists(test_file):
            found_test_file = test_file
            break
    
    if found_test_file:
        try:
            print(f"Testing with file: {found_test_file}")
            result = predict_parkinsons(found_test_file)
            
        except Exception as e:
            print(f"Test failed: {e}")
    else:
        print("No test audio files found.")
        print("\nTo test the script:")
        print("1. Place a .wav audio file in the current directory")
        print("2. Run: extract_features('your_audio_file.wav')")
        print("3. Or run: predict_parkinsons('your_audio_file.wav')")
    
    print("\nFunctions available:")
    print("- extract_features(file_path): Extract features from audio file")
    print("- predict_parkinsons(file_path): Extract features and make prediction")
    print("- load_and_preprocess_audio(file_path): Load and preprocess audio")