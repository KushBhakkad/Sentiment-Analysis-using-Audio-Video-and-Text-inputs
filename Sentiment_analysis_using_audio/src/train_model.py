import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from feature_extraction import extract_features_from_files
from data_processing import load_data, preprocess_data

def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(128, input_shape=(input_shape,), activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(data_path, num_classes):
    data, labels = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(data, labels)
    
    X_train = extract_features_from_files(X_train)
    X_test = extract_features_from_files(X_test)
    
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)
    
    model = create_model(X_train.shape[1], num_classes)
    
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
    
    model.save('models/sentiment_model.h5')

if __name__ == "__main__":
    train_model('path_to_your_data', num_classes=8)
