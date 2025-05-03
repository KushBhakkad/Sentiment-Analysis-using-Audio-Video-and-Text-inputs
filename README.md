# Sentiment-Analysis-using-Audio-Video-and-Text-inputs

This project consists of three different tools to Analyse Sentiment through different input modes: Audio, Video and Text. Technologies like Librosa, OpenCV, LSTM, CNN, etc. were used for the development. The Models are trained and tested on authentic data maintaining high performance and reliability.


## Sentiment-Analysis-Using-Audio

Sentiment Analysis Using Audio is a machine learning project designed to predict emotional sentiment from audio inputs. It uses audio feature extraction and a pre-trained neural network to classify emotions into one of eight categories.

**Features**
- Predicts sentiment from an uploaded audio file or recorded audio.
- Recognizes eight emotional sentiments:
  - Neutral
  - Calm
  - Happy
  - Sad
  - Angry
  - Fearful
  - Disgust
  - Surprised
- User-friendly interface through command-line interaction.
- Integrated audio recording functionality.

**Dataset**
- The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
- Dataset Source: https://zenodo.org/records/1188976

**Requirements**
- Python 3.8 or 3.9
- Required Python libraries (Install via pip):
  - pip install -r requirements.txt
  - For PyAudio installation:
    -Use the .whl file provided or install via pip:
      - pip install PyAudio-0.2.14-cp310-cp310-win_amd64.whl

**Command to run:**
- python prediction.py

**Output**

![Output](https://github.com/user-attachments/assets/b86e8e8c-5dc8-4075-8ecd-017e3c424131)

**Acknowledgements**
- TensorFlow for the deep learning framework.
- Librosa for audio processing.
- Speech Emotion Recognition researchers and contributors for inspiration and datasets.


## Sentiment-Analysis-Using-Video

This project demonstrates real-time sentiment analysis using video input from a webcam. By analyzing facial expressions, the model predicts emotions such as Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise. It leverages OpenCV for face detection and a pre-trained deep learning model for emotion classification.

**Features**

-Real-time emotion detection using webcam input.

-Accurate face detection with Haar cascades.

-Emotion classification using a trained Keras model.

**Dataset**

Facial Expression Recognition(FER)-2013 Dataset.

**Requirements**

-Python 3.7 or later

-Webcam for capturing video input

**Emotion Labels**

The model predicts the following emotions:

-Angry

-Disgust

-Fear

-Happy

-Neutral

-Sad

-Surprise

**Working**

![Happy](https://github.com/user-attachments/assets/764bdbb0-f6bb-4d18-ac2d-d38b35702185)![Neutral](https://github.com/user-attachments/assets/6d464427-8536-4bd4-bc00-b3c77e314a92)![Surprise](https://github.com/user-attachments/assets/99a6813f-0055-4bb2-bd4d-b4408c77ed3f)


## Sentiment-Analysis-Using-Text

A Python-based sentiment analysis project that predicts the sentiment of text as **positive**, **neutral**, or **negative**. This project leverages advanced Natural Language Processing (NLP) techniques and a deep learning model built with TensorFlow and Keras.

**Features**

- **Text Preprocessing**: Includes stopword removal, lemmatization, and tokenization using NLTK.
- **Deep Learning Model**: LSTM-based architecture for robust sentiment prediction.
- **User Interaction**: A command-line interface for real-time text sentiment prediction.
- **Customizable**: Easily extendable for other datasets or additional functionalities.

**Dataset** 
- Trained on Twitter sentiment data (Tweets.csv).

**Prerequisites**
1. Python 3.7+
2. Libraries: Install the required dependencies using the command:
   - pip install tensorflow nltk scikit-learn pandas

**Command to run**
- python predict_sentiment.py

**How It Works**
- **Data Preprocessing:**
  - Removes URLs, mentions, hashtags, and punctuation.
  - Converts text to lowercase.
  - Tokenizes and lemmatizes the text.
- **Model Architecture:**
  - Embedding layer for text vectorization.
  - LSTM layer for capturing context.
  - Fully connected dense layer with softmax activation for sentiment classification.
- **Prediction:**
  - Processes user input using the same tokenizer as the training process.
  - Predicts sentiment using the trained LSTM model.

**Output**

![Output](https://github.com/user-attachments/assets/d955f7c1-69fc-4073-8924-62dc21276cf5)