import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import os
import json
import requests
from flask import Flask, request, jsonify, render_template, send_file
from fpdf import FPDF
from io import BytesIO

# --- 1. PASTE YOUR OMDb API KEY (THE STRING) HERE (INSIDE THE QUOTES) ---
# --- Get your key from http://www.omdbapi.com/apikey.aspx ---
OMDb_API_KEY = "243ca78d" 
# -----------------------------------------------------------

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

app = Flask(__name__)

# --- Model Parameters ---
VOCABULARY_SIZE = 10000
MAX_LEN = 500

# --- FIX: Use absolute paths to load model and word index ---
# This ensures the server can find the files regardless of the working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'saved_model', 'sentiment_lstm_model.h5')
WORD_INDEX_PATH = os.path.join(BASE_DIR, 'word_index.json')

print(f"Loading Keras model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("Model loaded.")

print(f"Loading word index from: {WORD_INDEX_PATH}")
with open(WORD_INDEX_PATH) as f:
    word_to_id = json.load(f)
print("Word index loaded.")

# --- Preprocessing Function ---
def preprocess_text(text):
    text = re.sub('<[^>]*>', '', text)
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    review_ids = [word_to_id.get(word, 2) for word in words]
    review_ids = [i if i < VOCABULARY_SIZE else 2 for i in review_ids]
    return pad_sequences([review_ids], maxlen=MAX_LEN)

# --- Web Routes ---

@app.route('/')
def home():
    return render_template('index.html')

# --- NEW ROUTE FOR ABOUT US PAGE ---
@app.route('/about')
def about_page():
    return render_template('about.html')
# -----------------------------------

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        review_text = data['review']
        
        processed_review = preprocess_text(review_text)
        prediction_score = model.predict(processed_review, verbose=0)[0][0]
        sentiment = "Positive" if prediction_score >= 0.5 else "Negative"
        
        return jsonify({
            'sentiment': sentiment,
            'score': float(prediction_score)
        })
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Error during prediction'}), 500

# --- THIS ENTIRE FUNCTION IS UPDATED FOR OMDb ---
@app.route('/search_movie', methods=['GET'])
def search_movie():
    query = request.args.get('query')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    # Check if the API key is set
    if not OMDb_API_KEY or OMDb_API_KEY == "PASTE_YOUR_OMDB_KEY_STRING_HERE":
        print("ERROR: OMDb API Key is not set in app.py")
        return jsonify({'error': 'Server is not configured for movie search.'}), 500
        
    try:
        # Build the new OMDb URL
        # We use s= for search and type=movie to filter out TV shows
        url = f"http://www.omdbapi.com/?apikey={OMDb_API_KEY}&s={query}&type=movie"
        
        response = requests.get(url)
        response.raise_for_status() # This will catch HTTP errors
        data = response.json()
        
        movies = []
        # OMDb returns "True" or "False" (as strings)
        if data.get('Response') == 'True':
            # The list of movies is under the "Search" key
            for movie in data.get('Search', [])[:5]:
                movies.append({
                    'title': movie.get('Title'),
                    'year': movie.get('Year'),
                    # OMDb provides the FULL poster URL.
                    # We will send this full URL to the frontend.
                    'poster_path': movie.get('Poster') 
                })
        
        # This will correctly return an empty list if no movies are found
        return jsonify(movies)
        
    except requests.exceptions.RequestException as e:
        print(f"OMDb API Error: {e}")
        return jsonify({'error': 'Could not contact movie database.'}), 500
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({'error': 'An unexpected server error occurred.'}), 500
# --- END OF UPDATED FUNCTION ---


# --- ROUTE: For PDF Generation (WITH FIX) ---
@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    try:
        data = request.get_json(force=True)
        # --- FIX 1: Sanitize all incoming text to replace non-latin-1 chars ---
        movie_name = data.get('movie_name', 'Movie Analysis').encode('latin-1', 'replace').decode('latin-1')
        poster_url = data.get('poster_url')
        sentiment = data.get('sentiment') # We handle this one special below
        confidence = data.get('confidence', '').encode('latin-1', 'replace').decode('latin-1')
        raw_score = data.get('raw_score', '').encode('latin-1', 'replace').decode('latin-1')
        review_text = data.get('review_text', '').encode('latin-1', 'replace').decode('latin-1')

        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 15)
                self.set_text_color(220, 220, 220)
                self.cell(0, 10, 'Sentiment Analysis Report', 0, 1, 'C')
                self.ln(5)

            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.set_text_color(128)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

        pdf = PDF('P', 'mm', 'A4')
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_fill_color(30, 30, 30)
        pdf.set_text_color(230, 230, 230)
        
        pdf.set_font('Arial', 'B', 24)
        pdf.multi_cell(0, 12, movie_name, 0, 'C') # Use sanitized name
        pdf.ln(10)

        # Poster URL logic is now simpler, as the frontend sends the full URL
        if poster_url and 'http' in poster_url and poster_url != 'N/A':
            try:
                response = requests.get(poster_url)
                response.raise_for_status()
                img_bytes = BytesIO(response.content)
                
                # --- THIS IS THE FIX ---
                # Determine image type from URL instead of hard-coding 'JPG'
                img_type = ''
                if poster_url.lower().endswith('.png'):
                    img_type = 'PNG'
                elif poster_url.lower().endswith('.jpg') or poster_url.lower().endswith('.jpeg'):
                    img_type = 'JPG'
                
                # If we can't tell, default to JPG and hope for the best
                if not img_type:
                    print(f"Could not determine image type from URL: {poster_url}. Defaulting to JPG.")
                    img_type = 'JPG'
                # --- END OF FIX ---

                y = pdf.get_y() 
                # Use the new dynamic img_type variable
                pdf.image(img_bytes, x=55, y=y, w=100, type=img_type) 
                pdf.set_y(y + 155) # Move cursor down past the poster
                
            except Exception as e:
                print(f"Error downloading or processing poster: {e}")
                pdf.set_font('Arial', 'I', 10)
                pdf.set_text_color(255, 100, 100)
                pdf.cell(0, 10, '(Could not load movie poster)', 0, 1, 'C')
                pdf.ln(5)
        else:
             pdf.ln(10) 

        pdf.set_font('Arial', 'B', 18)
        
        # --- FIX 2: Remove emojis for the PDF ---
        if sentiment == 'Positive':
            pdf.set_text_color(46, 204, 113) # Green
            pdf.cell(0, 10, 'Sentiment: Positive', 0, 1, 'C') # Emoji removed
        else:
            pdf.set_text_color(231, 76, 60) # Red
            pdf.cell(0, 10, 'Sentiment: Negative', 0, 1, 'C') # Emoji removed
        
        pdf.ln(5)
        pdf.set_font('Arial', '', 14)
        pdf.set_text_color(230, 230, 230)
        pdf.cell(0, 10, confidence, 0, 1, 'C') # Use sanitized string
        pdf.cell(0, 10, raw_score, 0, 1, 'C') # Use sanitized string
        pdf.ln(10)

        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Original Review:', 0, 1, 'L')
        pdf.set_font('Arial', '', 12)
        pdf.set_fill_color(40, 40, 40)
        pdf.multi_cell(0, 8, review_text, border=1, fill=True) # Use sanitized string
        pdf.ln(5)

        # --- FIX 3: Change output method ---
        # Get the PDF as a string (dest='S')
        pdf_output_str = pdf.output(dest='S')
        # Encode the string to bytes, ignoring any errors (like emojis)
        pdf_output_bytes = pdf_output_str.encode('latin-1', 'ignore')
        # Pass the bytes to BytesIO
        pdf_bytes = BytesIO(pdf_output_bytes)
        
        return send_file(
            pdf_bytes,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"{movie_name.replace(' ', '_')}_Analysis.pdf"
        )
    except Exception as e:
        print(f"PDF Generation Error: {e}")
        # Send a specific error message back to the frontend
        return jsonify({'error': f'PDF Generation Error: {e}'}), 500

if __name__ == '__main__':
    print("\nStarting Flask server... Go to http://127.0.0.1:5000 in your browser.")
    app.run(debug=True)