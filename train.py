import os
from src.data_generator import generate_synthetic_dataset
from src.training import train_and_save_models
from src.visualization import plot_label_distribution
import pandas as pd

def main():
    data_path = os.path.join('data', 'synthetic_mental_health_posts.csv')
    models_dir = 'models'
    # Generate data if not exists
    if not os.path.exists(data_path):
        print('Generating synthetic dataset...')
        generate_synthetic_dataset(data_path)
    # Visualize data
    df = pd.read_csv(data_path)
    plot_label_distribution(df)
    # Train and save models
    train_and_save_models(data_path, models_dir)

if __name__ == "__main__":
    main()
