import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

def plot_label_distribution(df, save_path=None):
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x='sentiment', hue='sentiment', legend=False, palette='Set2')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels, save_path=None):
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show() 