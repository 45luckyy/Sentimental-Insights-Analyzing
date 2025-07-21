import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Load the preprocessed dataset
try:
    df = pd.read_csv('Preprocessed_Reddit_Data2.csv')
except FileNotFoundError:
    print("Error: 'Preprocessed_Reddit_Data2.csv' not found. Please ensure the dataset is in the correct directory.")
    exit()

# Drop rows with missing values
df.dropna(inplace=True)

# FIX: Remap labels from [-1, 0, 1] to [0, 1, 2]
df['category'] = df['category'] + 1

# Split data into training and testing sets
X = df['clean_comment']
y = df['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize the text
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

# Convert text to sequences
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Pad the sequences
X_train_padded = pad_sequences(X_train_sequences, maxlen=120, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=120, padding='post', truncating='post')

# Load GloVe pre-trained embeddings
def load_glove_embeddings(glove_file):
    print("Loading GloVe embeddings...")
    embeddings_index = {}
    try:
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        print("GloVe embeddings loaded.")
        return embeddings_index
    except FileNotFoundError:
        print(f"Error: GloVe file not found at {glove_file}. Please download it and place it in the correct directory.")
        return None

# Create the embedding matrix
def create_embedding_matrix(embeddings_index, word_index, embedding_dim=100):
    if embeddings_index is None:
        return None
    print("Creating embedding matrix...")
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print("Embedding matrix created.")
    return embedding_matrix

glove_file = 'glove.6B.100d.txt'
embeddings_index = load_glove_embeddings(glove_file)
embedding_matrix = create_embedding_matrix(embeddings_index, word_index)

# Build the model
model = Sequential([
    Embedding(len(word_index) + 1, 100, weights=[embedding_matrix], input_length=120, trainable=False),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    X_train_padded, y_train,
    epochs=20,
    validation_data=(X_test_padded, y_test),
    callbacks=[early_stopping]
)

# --- CHANGE: Save the model in the .h5 format ---
model.save('sentiment_model.h5')
print("Model training complete and saved as sentiment_model.h5")

# Save the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Tokenizer saved as tokenizer.pickle")

# Evaluate the model
loss, accuracy = model.evaluate(X_test_padded, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')