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
from src.visualization import plot_label_distribution, plot_confusion_matrix, plot_model_comparison

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
    
    # Visualize data distribution
    plot_label_distribution(df, save_path='static/sentiment_distribution.png')
    
    X = df['processed_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2, max_df=0.95)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train both models
    models = {}
    
    # Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_tfidf, y_train)
    lr_pred = lr.predict(X_test_tfidf)
    lr_acc = accuracy_score(y_test, lr_pred)
    models['logistic_regression'] = lr
    
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, lr_pred))
    
    # Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)
    nb_pred = nb.predict(X_test_tfidf)
    nb_acc = accuracy_score(y_test, nb_pred)
    models['naive_bayes'] = nb
    
    print(f"\nNaive Bayes Accuracy: {nb_acc:.4f}")
    print("Naive Bayes Classification Report:")
    print(classification_report(y_test, nb_pred))
    
    # Visualize model comparison
    plot_model_comparison(lr_acc, nb_acc, save_path='static/model_comparison.png')
    
    # Choose the best model
    if lr_acc > nb_acc:
        best_model = lr
        best_model_name = 'logistic_regression'
        print(f"\nLogistic Regression performs better ({lr_acc:.4f} vs {nb_acc:.4f})")
    else:
        best_model = nb
        best_model_name = 'naive_bayes'
        print(f"\nNaive Bayes performs better ({nb_acc:.4f} vs {lr_acc:.4f})")
    
    # Visualize confusion matrix for best model
    sentiment_labels = ['supportive', 'neutral', 'distress']
    if best_model_name == 'logistic_regression':
        plot_confusion_matrix(y_test, lr_pred, sentiment_labels, save_path='static/confusion_matrix.png')
    else:
        plot_confusion_matrix(y_test, nb_pred, sentiment_labels, save_path='static/confusion_matrix.png')
    
    # Save models
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'best_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    with open(os.path.join(models_dir, 'preprocessor.pkl'), 'wb') as f:
        pickle.dump(preprocessor, f)
    with open(os.path.join(models_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save all models for comparison
    with open(os.path.join(models_dir, 'all_models.pkl'), 'wb') as f:
        pickle.dump(models, f)
    
    print(f"Models saved to {models_dir}")
    print(f"Best model ({best_model_name}) saved as 'best_model.pkl'")
    
    return best_model, preprocessor, vectorizer

if __name__ == "__main__":
    train_and_save_models("../data/synthetic_mental_health_posts.csv", "../models") 