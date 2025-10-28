import json
from tensorflow.keras.datasets import imdb

print("Loading word index from Keras...")
word_to_id = imdb.get_word_index()

# Keras reserves 0, 1, 2 for padding, start, and unknown
word_to_id = {k:(v+3) for k, v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

print(f"Saving word index to word_index.json...")
# This creates the new file
with open('word_index.json', 'w') as f:
    json.dump(word_to_id, f)

print("Done! 'word_index.json' has been created.")
print("You can now run the web server by typing: python app.py")

