import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional # <-- IMPORT Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import os

# --- 1. Define Parameters ---
NUM_WORDS = 10000       # Vocabulary size: top 10,000 most frequent words
MAX_LEN = 500           # Max length of a review (in words)
EMBEDDING_DIM = 32      # Dimension of the word embedding vectors
BATCH_SIZE = 64
EPOCHS = 8              # <-- CHANGED: Train for longer

# --- 2. Load and Preprocess Data ---
print("Loading and preprocessing data...")
# Load the IMDB dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=NUM_WORDS)

# Pad sequences to ensure all reviews have the same length
X_train = pad_sequences(X_train, maxlen=MAX_LEN)
X_test = pad_sequences(X_test, maxlen=MAX_LEN)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# --- 3. Build the LSTM Model ---
print("Building the Bidirectional LSTM model...")
model = Sequential([
    # Embedding layer: Turns positive integers (word indices) into dense vectors
    Embedding(input_dim=NUM_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
    
    # --- CHANGED: Wrapped LSTM in Bidirectional ---
    # This layer reads the sequence forwards and backwards for better context
    Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)),
    
    # Output layer: A single neuron with a sigmoid activation for binary classification (0 or 1)
    Dense(1, activation='sigmoid')
])

# --- 4. Compile the Model ---
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print(model.summary())

# --- 5. Train the Model ---
print("\nTraining the model...")
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test)
)

# --- 6. Evaluate the Model ---
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# --- 7. Save the Model ---
print("Saving the trained model...")
# Create a directory if it doesn't exist
if not os.path.exists('saved_model'):
    os.makedirs('saved_model')
model.save('saved_model/sentiment_lstm_model.h5')
print("Model saved successfully in 'saved_model' directory.")

# --- 8. Plot Training History ---
def plot_history(history_data):
    acc = history_data.history['accuracy']
    val_acc = history_data.history['val_accuracy']
    loss = history_data.history['loss']
    val_loss = history_data.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs_range, val_acc, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'bo-', label='Training Loss')
    plt.plot(epochs_range, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)

