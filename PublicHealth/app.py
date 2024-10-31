from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
import os

app = Flask(__name__)
CORS(app)

# Download necessary NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load pre-trained Word2Vec model
model_path = 'GoogleNews-vectors-negative300.bin'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Load sentences from Excel file
excel_file = 'Untitled2.xlsx'
df = pd.read_excel(excel_file)

# Check if required columns are in the DataFrame
if 'sentences' not in df.columns or 'severity' not in df.columns or 'disease' not in df.columns:
    raise ValueError("Excel file must contain 'sentences', 'severity', and 'disease' columns.")

sentences = df['sentences'].tolist()
severities = df['severity'].tolist()
diseases = df['disease'].tolist()

def sentence_to_vec(sentence, model):
    words = word_tokenize(sentence.lower())
    word_vecs = [model[word] for word in words if word in model]
    if len(word_vecs) == 0:
        return np.zeros(model.vector_size)
    return np.mean(word_vecs, axis=0)

def find_top_similar_sentences(target_sentence, sentences, model, top_n=10):
    target_vec = sentence_to_vec(target_sentence, model)
    sentence_vecs = [sentence_to_vec(sent, model) for sent in sentences]

    similarities = cosine_similarity([target_vec], sentence_vecs)[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]  # Get top N indices

    results = []
    for index in top_indices:
        results.append({
            'sentence': sentences[index],
            'similarity_score': float(similarities[index]),
            'severity': severities[index],
            'disease': diseases[index]
        })

    return results

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/find_similar', methods=['POST'])
def find_similar():
    data = request.get_json()

    # Validate input data
    if not data or 'sentence' not in data or 'user' not in data:
        return jsonify({'error': 'Invalid input data'}), 400

    target_sentence = data['sentence']

    # Find the most similar sentence with its details
    similar_sentences = find_top_similar_sentences(target_sentence, sentences, model, top_n=1)

    if similar_sentences:
        top_match = similar_sentences[0]  # Get the best match

        response = {
            'disease': top_match['disease'],
            'severity': top_match['severity'],
            'similarity_score': round(top_match['similarity_score'] * 100, 2)  # Convert to percentage with two decimals
        }
    else:
        response = {
            'error': 'No similar sentences found'
        }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)


