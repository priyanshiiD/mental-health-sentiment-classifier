import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download all required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")

class TextPreprocessor:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("Warning: Stopwords not available, using empty set")
            self.stop_words = set()
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        if pd.isna(text) or text == '':
            return ''
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text):
        try:
            return word_tokenize(text)
        except LookupError:
            # Fallback to simple word splitting if NLTK tokenizer fails
            return text.split()

    def remove_stopwords(self, tokens):
        return [token for token in tokens if token not in self.stop_words]

    def lemmatize(self, tokens):
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def preprocess(self, text):
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return ' '.join(tokens)

    def preprocess_series(self, series):
        return series.apply(self.preprocess) 