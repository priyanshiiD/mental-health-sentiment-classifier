import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import os

def plot_label_distribution(df, save_path=None):
    """Plot sentiment distribution using seaborn"""
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='sentiment', hue='sentiment', legend=False, palette='Set2')
    plt.title('Mental Health Sentiment Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Sentiment Categories', fontsize=12)
    plt.ylabel('Number of Posts', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to {save_path}")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels, save_path=None):
    """Plot confusion matrix using matplotlib"""
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, 
        display_labels=labels, 
        cmap='Blues',
        values_format='d'
    )
    plt.title('Model Performance - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.show()

def plot_model_comparison(lr_acc, nb_acc, save_path=None):
    """Compare Logistic Regression vs Naive Bayes performance"""
    models = ['Logistic Regression', 'Naive Bayes']
    accuracies = [lr_acc, nb_acc]
    colors = ['#2E8B57', '#4682B4']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=colors)
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to {save_path}")
    plt.show()

def plot_model_performance(accuracy, precision, recall, f1_score, save_path=None):
    """Plot model performance metrics"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1_score]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['#2E8B57', '#4682B4', '#D2691E', '#DC143C'])
    plt.title('Model Performance Metrics', fontsize=16, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance plot saved to {save_path}")
    plt.show() 