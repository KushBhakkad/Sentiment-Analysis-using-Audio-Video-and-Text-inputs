import sys
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('punkt')

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'\@w+|\#','', text) # Remove mentions and hashtags
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = text.lower() # Lowercase text
    tokens = word_tokenize(text)
    filtered_words = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(filtered_words)

# Load tokenizer and model
max_words = 5000
max_len = 200
tokenizer = Tokenizer(num_words=max_words, lower=True)
tokenizer.fit_on_texts(pd.read_csv('data/Tweets.csv')['text'].apply(preprocess_text).values)
model = load_model('models/sentiment_model.keras')

# Predict function
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded_sequence)
    sentiment = np.argmax(prediction, axis=1)[0]
    sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return sentiment_labels[sentiment]

# Main loop
while True:
    input_text = input("Enter text (or 'exit' to quit): ")
    if input_text.lower() == 'exit':
        break
    print(f"Predicted sentiment: {predict_sentiment(input_text)}")
