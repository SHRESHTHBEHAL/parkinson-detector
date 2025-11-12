"""
AI Detection of Parkinson's from Voice - Hackathon MVP

This script downloads the UCI Parkinson's dataset, preprocesses it,
trains a RandomForest classifier, and saves the model for deployment.
"""

import pandas as pd
import numpy as np
import requests
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from io import StringIO

def download_dataset():
    """
    Download the UCI Parkinson's dataset from the official repository.
    
    Returns:
        pd.DataFrame: Loaded dataset as pandas DataFrame
    """
    print("Downloading UCI Parkinson's dataset...")
    
    # UCI Parkinson's dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    
    try:
        # Download the dataset
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Load the dataset into a pandas DataFrame
        # The dataset has headers, so we'll read it properly
        data = pd.read_csv(StringIO(response.text))
        
        print(f"Dataset downloaded successfully! Shape: {data.shape}")
        
        return data
        
    except requests.RequestException as e:
        print(f"Error downloading dataset: {e}")
        raise
    except Exception as e:
        print(f"Error processing dataset: {e}")
        raise

def preprocess_data(data):
    """
    Clean and preprocess the Parkinson's dataset.
    
    Args:
        data (pd.DataFrame): Raw dataset
        
    Returns:
        tuple: Processed features (X) and target (y)
    """
    print("Preprocessing data...")
    
    # Display basic info about the dataset
    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Target variable distribution:")
    print(data['status'].value_counts())
    
    # Remove the 'name' column as it's not a feature
    if 'name' in data.columns:
        data = data.drop('name', axis=1)
        print("Removed 'name' column")
    
    # Convert all columns to numeric (except 'status' for now)
    numeric_columns = [col for col in data.columns if col != 'status']
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    print("Converted feature columns to numeric")
    
    # Check for missing values after conversion
    missing_values = data.isnull().sum()
    if missing_values.any():
        print(f"Found missing values after conversion:\n{missing_values[missing_values > 0]}")
        # Handle missing values by dropping rows with NaN
        data = data.dropna()
        print(f"After removing missing values, shape: {data.shape}")
    else:
        print("No missing values found")
    
    # Separate features and target
    X = data.drop('status', axis=1)
    y = data['status']
    
    # Convert target to numeric as well
    y = pd.to_numeric(y, errors='coerce')
    
    # Remove any rows where target couldn't be converted
    valid_indices = ~y.isna()
    X = X[valid_indices]
    y = y[valid_indices]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Final target distribution:")
    print(y.value_counts().sort_index())
    
    return X, y

def train_model(X, y):
    """
    Train a RandomForest classifier on the preprocessed data.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        
    Returns:
        tuple: (trained_model, X_train, X_test, y_train, y_test, scaler)
    """
    print("Training RandomForest classifier...")
    
    # Split the data into training and testing sets
    # Remove stratification temporarily to handle edge cases
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Scale the features for better performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features normalized using StandardScaler")
    
    # Train RandomForest classifier
    # Using RandomForest as it handles mixed feature types well and provides good performance
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.fit(X_train_scaled, y_train)
    print("RandomForest model trained successfully")
    
    return model, X_train_scaled, X_test_scaled, y_train, y_test, scaler

def evaluate_model(model, X_test_scaled, y_test):
    """
    Evaluate the trained model and print performance metrics.
    
    Args:
        model: Trained classifier
        X_test_scaled (np.array): Scaled test features
        y_test (pd.Series): Test target values
    """
    print("\nEvaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'Parkinson\'s']))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Feature importance
    feature_names = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
        'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer',
        'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
        'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    ]
    
    # Get feature importance from RandomForest
    importance = model.feature_importances_
    feature_importance = list(zip(feature_names, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 Most Important Features:")
    for i, (feature, imp) in enumerate(feature_importance[:10]):
        print(f"{i+1:2d}. {feature:<20}: {imp:.4f}")

def save_model_and_scaler(model, scaler):
    """
    Save the trained model and scaler using joblib.
    
    Args:
        model: Trained classifier
        scaler: Fitted StandardScaler
    """
    print("\nSaving model and scaler...")
    
    # Save the trained model
    model_filename = "model.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved as {model_filename}")
    
    # Save the scaler for preprocessing new data
    scaler_filename = "scaler.pkl"
    joblib.dump(scaler, scaler_filename)
    print(f"Scaler saved as {scaler_filename}")

def main():
    """
    Main function to orchestrate the entire pipeline.
    """
    print("="*60)
    print("AI Detection of Parkinson's from Voice - MVP")
    print("="*60)
    
    try:
        # Step 1: Download and load the dataset
        data = download_dataset()
        
        # Step 2: Preprocess the data
        X, y = preprocess_data(data)
        
        # Step 3: Train the model
        model, X_train, X_test, y_train, y_test, scaler = train_model(X, y)
        
        # Step 4: Evaluate the model
        evaluate_model(model, X_test, y_test)
        
        # Step 5: Save the model and scaler
        save_model_and_scaler(model, scaler)
        
        print("\n" + "="*60)
        print("Pipeline completed successfully!")
        print("Model saved as 'model.pkl' and scaler as 'scaler.pkl'")
        print("="*60)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()