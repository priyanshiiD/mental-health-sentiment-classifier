import pandas as pd
import os

def generate_synthetic_dataset(save_path=None):
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
        "You're stronger than you think. This difficult time will pass."
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
        "Mental health awareness has increased significantly in recent years."
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
        "I'm so scared and I don't know what to do. I need help but I'm afraid to ask."
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

if __name__ == "__main__":
    df = generate_synthetic_dataset("../data/synthetic_mental_health_posts.csv")
    print(f"Saved {len(df)} samples to ../data/synthetic_mental_health_posts.csv") 