import pandas as pd
import os
import requests
import json

def download_mental_health_dataset():
    """Download mental health dataset from Hugging Face"""
    try:
        # Using a mental health dataset from Hugging Face
        url = "https://huggingface.co/datasets/mental-health-dataset/mental-health-text-classification/raw/main/mental_health_dataset.csv"
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            df = pd.read_csv(url)
            return df
        else:
            print("Could not download dataset, using synthetic data")
            return None
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

def generate_synthetic_dataset(save_path=None):
    """Generate synthetic dataset with more diverse examples"""
    supportive_posts = [
        "You're doing great! Remember that it's okay to take things one step at a time.",
        "I believe in you and your strength to get through this difficult time.",
        "You're not alone in this journey. There are people who care about you.",
        "It's completely normal to feel this way. You're showing great courage by reaching out.",
        "You have the power within you to overcome these challenges. Keep going!",
        "Your feelings are valid and important. Don't be too hard on yourself.",
        "It's okay to ask for help when you need it. That's a sign of strength.",
        "You're making progress, even if it doesn't feel like it right now.",
        "Remember to be kind to yourself today. You deserve compassion.",
        "You're stronger than you think. This difficult time will pass.",
        "Hang in there, better days are ahead.",
        "You're not a failure, you're learning and growing.",
        "It's brave of you to share your feelings.",
        "You matter and your life has value.",
        "There's help available, you don't have to go through this alone.",
        # Simple/positive words
        "good", "happy", "fine", "okay", "well", "great", "awesome", "fantastic", "excellent", "positive", "hopeful", "calm", "relaxed", "peaceful", "content", "joyful", "cheerful", "smiling", "confident", "motivated", "supported"
    ]
    
    neutral_posts = [
        "Depression affects approximately 280 million people worldwide.",
        "Anxiety disorders are the most common mental health condition in the US.",
        "Regular exercise can help improve mood and reduce symptoms of depression.",
        "Sleep hygiene is important for maintaining good mental health.",
        "Cognitive Behavioral Therapy (CBT) is an effective treatment for many mental health conditions.",
        "Mindfulness meditation has been shown to reduce stress and anxiety.",
        "Social support networks are crucial for mental health recovery.",
        "Professional therapy can provide tools and strategies for managing mental health.",
        "Medication can be an effective treatment option for some mental health conditions.",
        "Mental health awareness has increased significantly in recent years.",
        "Therapy sessions typically last 50-60 minutes.",
        "Antidepressants can take 4-6 weeks to show full effects.",
        "Exercise releases endorphins which improve mood.",
        "Sleep deprivation can worsen mental health symptoms.",
        "Social isolation is a risk factor for depression.",
        # Simple/neutral words
        "information", "fact", "data", "statistics", "number", "average", "report", "study", "research", "neutral", "statement", "general", "update", "news", "article", "summary", "overview", "routine", "schedule", "plan", "note"
    ]
    
    distress_posts = [
        "I can't take this anymore. Everything feels hopeless and pointless.",
        "I'm so alone and no one understands what I'm going through.",
        "I don't see any point in continuing. Life is just too painful.",
        "I feel like I'm drowning in my own thoughts and emotions.",
        "I can't stop thinking about ending it all. I'm so tired of fighting.",
        "I'm a burden to everyone around me. They'd be better off without me.",
        "I can't get out of bed anymore. Everything feels overwhelming.",
        "I'm having thoughts of self-harm and I don't know how to stop them.",
        "I feel like I'm losing my mind. Nothing makes sense anymore.",
        "I'm so scared and I don't know what to do. I need help but I'm afraid to ask.",
        "I want to disappear. I hate existing.",
        "I feel like I should just give up on everything.",
        "Nobody would miss me if I was gone.",
        "I'm worthless and don't deserve to live.",
        "I can't stop thinking about hurting myself.",
        "I wish I could just disappear from this world.",
        "I'm tired of being alive.",
        "I feel like ending it all.",
        "I don't want to exist anymore.",
        "I hate myself and my life.",
        "I'm a failure and everyone knows it.",
        "I can't take the pain anymore.",
        "I want to die.",
        "I'm better off dead.",
        "I don't deserve to be happy.",
        "I'm a waste of space.",
        "I should just kill myself.",
        "I'm broken beyond repair.",
        "I can't handle this pain anymore.",
        "I'm done with life."
    ]
    
    data = []
    for post in supportive_posts:
        data.append({'text': post, 'sentiment': 'supportive', 'label': 0})
    for post in neutral_posts:
        data.append({'text': post, 'sentiment': 'neutral', 'label': 1})
    for post in distress_posts:
        data.append({'text': post, 'sentiment': 'distress', 'label': 2})
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
    
    return df

def get_mental_health_dataset(save_path=None):
    """Get mental health dataset - try real data first, fallback to synthetic"""
    # Try to download real dataset
    real_df = download_mental_health_dataset()
    
    if real_df is not None and len(real_df) > 0:
        print(f"Using real mental health dataset with {len(real_df)} samples")
        # Map the real dataset to our format
        if 'text' in real_df.columns and 'label' in real_df.columns:
            # Assuming the real dataset has similar labels
            df = real_df[['text', 'label']].copy()
            df['sentiment'] = df['label'].map({0: 'supportive', 1: 'neutral', 2: 'distress'})
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                df.to_csv(save_path, index=False)
            return df
    
    # Fallback to synthetic data
    print("Using synthetic mental health dataset")
    return generate_synthetic_dataset(save_path)

if __name__ == "__main__":
    df = get_mental_health_dataset("../data/synthetic_mental_health_posts.csv")
    print(f"Saved {len(df)} samples to ../data/synthetic_mental_health_posts.csv") 