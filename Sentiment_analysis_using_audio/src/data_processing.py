import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np
import librosa

def load_data(data_path):
    # Load your dataset
    data = []
    labels = []
    for file_name in os.listdir(data_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(data_path, file_name)
            label = file_name.split('-')[0]  # Example of label extraction
            data.append(file_path)
            labels.append(label)
    
    return data, labels

def preprocess_data(data, labels, test_size=0.2):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test
