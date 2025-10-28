import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# --- 0. Define Model Parameters ---
# This MUST match the parameters used in main.py
VOCABULARY_SIZE = 10000
MAX_LEN = 500

# --- 1. Load the Saved Model and Word Index ---
print("Loading the saved model...")
model = load_model('saved_model/sentiment_lstm_model.h5')
print("Model loaded.")

# Load the original word index from IMDB dataset
word_to_id = imdb.get_word_index()
# The first few indices are reserved for special characters
# 0: padding, 1: start of sequence, 2: unknown word
word_to_id = {k:(v+3) for k, v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

# Invert the dictionary to map IDs back to words (for potential debugging)
id_to_word = {value: key for key, value in word_to_id.items()}

# --- 2. Preprocessing Function for New Reviews ---
def preprocess_text(text):
    """
    Cleans and converts a single text review into a sequence of word IDs.
    """
    # Remove HTML tags and non-alphabetic characters
    text = re.sub('<[^>]*>', '', text)
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    
    # Tokenize and convert to word IDs
    words = text.split()
    review_ids = [word_to_id.get(word, 2) for word in words] # Use 2 for unknown words
    
    # --- THIS IS THE CRITICAL FIX ---
    # Filter out any word IDs that are outside the model's vocabulary
    review_ids = [i if i < VOCABULARY_SIZE else 2 for i in review_ids]
    # --- END OF FIX ---
    
    # Pad the sequence
    padded_review = pad_sequences([review_ids], maxlen=MAX_LEN) # Use the same max_len as training
    return padded_review

# --- 3. Prediction Function ---
def predict_sentiment(review_text):
    """
    Takes a string review, preprocesses it, and predicts the sentiment.
    """
    # Preprocess the input review
    processed_review = preprocess_text(review_text)
    
    # Make a prediction (added verbose=0 to hide '1/1' output)
    prediction_score = model.predict(processed_review, verbose=0)[0][0]
    
    # Interpret the result
    if prediction_score >= 0.5:
        sentiment = "Positive"
    else:
        sentiment = "Negative"
        
    print(f"\nReview: '{review_text}'")
    print(f"Prediction Score: {prediction_score:.4f}")
    print(f"Predicted Sentiment: {sentiment}")

# --- 4. Interactive Prediction Loop ---
if __name__ == "__main__":
    print("\n--- Movie Sentiment Analyzer ---")
    print("Type 'exit' or 'quit' to stop.")
    
    while True:
        # Ask the user to enter a review
        user_review = input("\nEnter a movie review: ")
        
        # Check if the user wants to exit
        if user_review.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
            
        # If the input is not empty, predict its sentiment
        if user_review:
            predict_sentiment(user_review)
        else:
            print("Please enter some text.")

