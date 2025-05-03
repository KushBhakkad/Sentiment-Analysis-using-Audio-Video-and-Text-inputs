import librosa
import numpy as np

def extract_features(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

def extract_features_from_files(file_paths, n_mfcc=40):
    features = []
    for file_path in file_paths:
        feature = extract_features(file_path, n_mfcc)
        features.append(feature)
    return np.array(features)
