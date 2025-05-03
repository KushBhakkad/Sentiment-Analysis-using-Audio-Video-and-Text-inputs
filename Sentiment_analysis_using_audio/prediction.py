import os
import librosa
import numpy as np
import soundfile as sf
from tensorflow.keras.models import load_model
import pyaudio
import wave

# Load the trained model
model = load_model('models/sentiment_model.h5')

def extract_features(file_path, n_mfcc=40):
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

def predict_sentiment(audio_file):
    print(f"Predicting sentiment for file: {audio_file}")  # Debug print statement
    features = extract_features(audio_file)
    print(f"Extracted features: {features}")  # Debug print statement
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features)
    sentiment = np.argmax(prediction, axis=1)[0]
    return sentiment

def record_audio(output_file="recorded.wav", record_seconds=5, sample_rate=16000, chunk=1024):
    # Set up the pyaudio parameters
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=chunk)
    print("Recording...")

    frames = []
    for _ in range(0, int(sample_rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    print("Recording finished.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded data as a WAV file
    wf = wave.open(output_file, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    return output_file

def main():
    print("Select an option:")
    print("1. Upload an audio file")
    print("2. Record your own audio")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == '1':
        file_path = input("Enter the path to the audio file: ")
        print(f"Received file path: {file_path}")  # Debug print statement
        if not os.path.isfile(file_path):
            print("Invalid file path. Exiting.")
            return
    elif choice == '2':
        print("Recording audio...")
        file_path = record_audio()
        print(f"Recorded file path: {file_path}")  # Debug print statement
    else:
        print("Invalid choice. Exiting.")
        return
    
    if not os.path.isfile(file_path):
        print(f"File not found at path: {file_path}")
        return
    
    sentiment = predict_sentiment(file_path)
    sentiments_map = {
        0: "neutral",
        1: "calm",
        2: "happy",
        3: "sad",
        4: "angry",
        5: "fearful",
        6: "disgust",
        7: "surprised"
    }
    print(f"The predicted sentiment is: {sentiments_map[sentiment]}")
    
    # If recording, remove the temporary file
    if choice == '2' and os.path.isfile(file_path):
        os.remove(file_path)


if __name__ == "__main__":
    main()
