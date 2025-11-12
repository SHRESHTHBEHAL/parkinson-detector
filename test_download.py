"""
Test script to check dataset download
"""

import requests
import pandas as pd
from io import StringIO

def test_download():
    """Test downloading the UCI Parkinson's dataset"""
    print("Testing dataset download...")
    
    # UCI Parkinson's dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    
    try:
        print(f"Downloading from: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        print(f"Download successful! Status: {response.status_code}")
        print(f"Content length: {len(response.text)} characters")
        print(f"First few lines:")
        print(response.text[:500])
        
        # Test loading the data
        column_names = [
            'name', 'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
            'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer',
            'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
            'NHR', 'HNR', 'RPDE', 'D2', 'DFA', 'spread1', 'spread2', 'DDE', 'status'
        ]
        
        data = pd.read_csv(StringIO(response.text), names=column_names)
        print(f"\nDataset loaded successfully!")
        print(f"Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"Target distribution:")
        print(data['status'].value_counts())
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_download()