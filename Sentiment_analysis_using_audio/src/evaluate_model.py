import numpy as np
from tensorflow.keras.models import load_model
from feature_extraction import extract_features_from_files
from data_processing import load_data, preprocess_data

def evaluate_model(data_path, num_classes):
    data, labels = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(data, labels)
    
    X_test = extract_features_from_files(X_test)
    y_test = to_categorical(y_test, num_classes=num_classes)
    
    model = load_model('models/sentiment_model.h5')
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    evaluate_model('path_to_your_data', num_classes=8)
