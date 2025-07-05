from flask import Flask, render_template, request, jsonify
import pickle
import os
import pandas as pd
from src.preprocessing import TextPreprocessor
from src.training import train_and_save_models
from src.data_generator import generate_synthetic_dataset

app = Flask(__name__)

# Global variables
classifier = None
preprocessor = None
vectorizer = None
sentiment_labels = ['supportive', 'neutral', 'distress']

def load_or_train_models():
    global classifier, preprocessor, vectorizer
    models_dir = 'models'
    data_path = os.path.join('data', 'synthetic_mental_health_posts.csv')
    try:
        with open(os.path.join(models_dir, 'best_model.pkl'), 'rb') as f:
            classifier = pickle.load(f)
        with open(os.path.join(models_dir, 'preprocessor.pkl'), 'rb') as f:
            preprocessor = pickle.load(f)
        with open(os.path.join(models_dir, 'vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        print('Models loaded successfully!')
        return True
    except Exception as e:
        print(f'Models not found or error: {e}. Training new models...')
        classifier, preprocessor, vectorizer = train_and_save_models(data_path, models_dir)
        return True

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        processed_text = preprocessor.preprocess(text)
        text_tfidf = vectorizer.transform([processed_text])
        prediction = classifier.predict(text_tfidf)[0]
        probabilities = classifier.predict_proba(text_tfidf)[0]
        predicted_sentiment = sentiment_labels[prediction]
        response = {
            'text': text,
            'processed_text': processed_text,
            'predicted_sentiment': predicted_sentiment,
            'confidence': float(max(probabilities)),
            'probabilities': {
                'supportive': float(probabilities[0]),
                'neutral': float(probabilities[1]),
                'distress': float(probabilities[2])
            }
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': classifier is not None})

if __name__ == '__main__':
    if not load_or_train_models():
        print('Failed to load or train models.')
        exit(1)
    print('Mental Health Sentiment Classifier Web App')
    print('=' * 50)
    print('Server starting on http://localhost:5000')
    app.run(debug=True, host='0.0.0.0', port=5000) 