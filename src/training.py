import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from src.preprocessing import TextPreprocessor
from src.data_generator import get_mental_health_dataset

def train_and_save_models(data_path, models_dir):
    # Load or generate data
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        df = get_mental_health_dataset(data_path)
    
    preprocessor = TextPreprocessor()
    df['processed_text'] = preprocessor.preprocess_series(df['text'])
    df = df[df['processed_text'].str.len() > 0].reset_index(drop=True)
    
    print(f"Training on {len(df)} samples")
    print(f"Class distribution: {df['sentiment'].value_counts().to_dict()}")
    
    X = df['processed_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2, max_df=0.95)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_tfidf, y_train)
    y_pred = lr.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Logistic Regression Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Save models
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'best_model.pkl'), 'wb') as f:
        pickle.dump(lr, f)
    with open(os.path.join(models_dir, 'preprocessor.pkl'), 'wb') as f:
        pickle.dump(preprocessor, f)
    with open(os.path.join(models_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"Models saved to {models_dir}")
    return lr, preprocessor, vectorizer

if __name__ == "__main__":
    train_and_save_models("../data/synthetic_mental_health_posts.csv", "../models") 