import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
import pickle

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Load dataset
df = pd.read_csv('data/Tweets.csv')

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@w+|\#','', text)  # Remove mentions and hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Lowercase text
    tokens = word_tokenize(text)
    filtered_words = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(filtered_words)

df['text'] = df['text'].apply(preprocess_text)

# Tokenization and Padding
max_words = 5000
max_len = 200
tokenizer = Tokenizer(num_words=max_words, lower=True)
tokenizer.fit_on_texts(df['text'].values)
word_index = tokenizer.word_index
X = tokenizer.texts_to_sequences(df['text'].values)
X = pad_sequences(X, maxlen=max_len)

# Encode target labels
y = pd.get_dummies(df['airline_sentiment']).values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model building
model = Sequential()
model.add(Embedding(max_words, 100, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Training
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# Save model and tokenizer
model.save('models/sentiment_model_final.keras')

# Save tokenizer for future predictions
with open('models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)