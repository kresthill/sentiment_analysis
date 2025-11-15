import pandas as pd
import numpy as np

def create_hybrid_dataset():
    """
    Create a hybrid dataset:
    - Use your REAL 250 angry tweets
    - Generate sample tweets for other emotions
    """
    
    print("="*70)
    print("CREATING HYBRID DATASET")
    print("="*70)
    
    # Load your real angry tweets
    print("\nLoading real angry tweets...")
    angry_df = pd.read_csv('../data/raw/upset.csv')
    angry_df = angry_df[['text', 'id', 'tweet_id']].copy()
    angry_df['sentiment'] = 'angry'
    angry_df['id'] = 3
    print(f"✓ Loaded {len(angry_df)} real angry tweets")
    
    # Create diverse sample tweets for other emotions
    print("\nGenerating sample tweets for other emotions...")
    
    happy_samples = [
        "I'm so excited about this amazing opportunity! Best feeling ever!",
        "Absolutely thrilled with how everything turned out today!",
        "Feeling blessed and grateful for all the wonderful people in my life",
        "This makes me so happy I can't stop smiling!",
        "What a fantastic day! Everything went perfectly!",
        "So delighted to share this incredible news with everyone",
        "Overjoyed and ecstatic about this wonderful surprise!",
        "My heart is full of joy and happiness right now",
        "This is the best thing that's happened to me all year!",
        "Celebrating this amazing achievement with pure joy!",
        "Feeling on top of the world today! Life is beautiful!",
        "So much love and positivity surrounding me right now",
        "Incredibly happy and content with where I am in life",
        "This wonderful moment deserves all the celebration!",
        "Pure happiness and excitement flowing through me!",
        "Grateful for this amazing journey and all its blessings",
        "What an incredible experience that filled me with joy!",
        "So happy I could cry! This means everything to me!",
        "Blessed beyond measure with this fantastic opportunity",
        "The happiness I feel right now is indescribable!",
        "Absolutely loving every moment of this wonderful day!",
        "My heart is singing with joy and gratitude today",
        "This amazing news has made my entire week perfect!",
        "Feeling so cheerful and optimistic about everything!",
        "What a beautiful day filled with love and happiness!"
    ]
    
    sad_samples = [
        "Feeling really down today, everything seems so difficult",
        "My heart is heavy with sadness and disappointment",
        "Can't shake this feeling of overwhelming sadness",
        "So disappointed with how things turned out",
        "Everything feels hopeless and empty right now",
        "Deeply hurt by what happened, can't stop thinking about it",
        "The sadness is consuming me today",
        "Feeling lost and alone in this difficult time",
        "My heart aches with this unbearable sadness",
        "Everything reminds me of what I've lost",
        "So much pain and sorrow weighing on my heart",
        "Can't find any joy in anything anymore",
        "Feeling completely heartbroken and devastated",
        "The loneliness is crushing me today",
        "Wish I could escape this overwhelming sadness",
        "Everything feels meaningless and sad right now",
        "Drowning in tears and can't seem to stop",
        "This emptiness inside is so painful",
        "Missing what used to be, feeling so blue",
        "The sadness just won't go away no matter what",
        "Feeling defeated and discouraged by everything",
        "My soul feels heavy with this persistent sadness",
        "Can't remember the last time I felt truly happy",
        "This melancholy mood has taken over completely",
        "Feeling so down and depressed about the situation"
    ]
    
    fearful_samples = [
        "I'm really scared about what might happen next",
        "Feeling so anxious and worried about everything",
        "Terrified of the possible outcomes of this situation",
        "This constant fear is overwhelming me",
        "Can't sleep because I'm so worried and afraid",
        "My anxiety is through the roof right now",
        "So nervous about what the future holds",
        "Fear and panic are consuming my thoughts",
        "This uncertainty is making me incredibly anxious",
        "I'm frightened by all these unknowns",
        "The worry is eating away at me constantly",
        "Feeling paralyzed by fear and anxiety",
        "So scared that something terrible will happen",
        "My heart races with fear every time I think about it",
        "This nervous energy won't leave me alone",
        "Terrified of making the wrong decision",
        "Anxiety and fear are my constant companions now",
        "Can't shake this feeling of impending doom",
        "So worried about everything going wrong",
        "The fear of failure is overwhelming me",
        "Feeling trapped by my own anxious thoughts",
        "This dread and worry are consuming me",
        "Scared out of my mind about the possibilities",
        "My anxiety makes everything seem impossible",
        "Living in constant fear and apprehension"
    ]
    
    # Expand samples to match angry tweets count
    multiplier = len(angry_df) // 25 + 1
    
    happy_tweets = happy_samples * multiplier
    sad_tweets = sad_samples * multiplier
    fearful_tweets = fearful_samples * multiplier
    
    # Trim to match sizes
    happy_tweets = happy_tweets[:len(angry_df)]
    sad_tweets = sad_tweets[:len(angry_df)]
    fearful_tweets = fearful_tweets[:len(angry_df)]
    
    # Create DataFrames
    happy_df = pd.DataFrame({
        'text': happy_tweets,
        'id': 1,
        'tweet_id': [f'happy_sample_{i}' for i in range(len(happy_tweets))],
        'sentiment': 'happy'
    })
    
    sad_df = pd.DataFrame({
        'text': sad_tweets,
        'id': 2,
        'tweet_id': [f'sad_sample_{i}' for i in range(len(sad_tweets))],
        'sentiment': 'sad'
    })
    
    fearful_df = pd.DataFrame({
        'text': fearful_tweets,
        'id': 4,
        'tweet_id': [f'fearful_sample_{i}' for i in range(len(fearful_tweets))],
        'sentiment': 'fearful'
    })
    
    print(f"✓ Generated {len(happy_df)} happy samples")
    print(f"✓ Generated {len(sad_df)} sad samples")
    print(f"✓ Generated {len(fearful_df)} fearful samples")
    
    # Combine all datasets
    combined_df = pd.concat([happy_df, sad_df, angry_df, fearful_df], ignore_index=True)
    
    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to processed folder
    output_path = '../data/processed/multi_sentiment_dataset.csv'
    combined_df.to_csv(output_path, index=False, encoding='utf-8')
    
    print("\n" + "="*70)
    print("DATASET CREATED SUCCESSFULLY!")
    print("="*70)
    print(f"\nTotal tweets: {len(combined_df)}")
    print("\nBreakdown by sentiment:")
    print(combined_df['sentiment'].value_counts())
    print(f"\n✓ Saved as: {output_path}")
    print("\nThis dataset contains:")
    print("  • 250 REAL angry tweets from Twitter")
    print("  • 250 sample happy tweets (for training)")
    print("  • 250 sample sad tweets (for training)")
    print("  • 250 sample fearful tweets (for training)")
    print("\nYou can now train your multi-class classifier!")
    
    return combined_df

if __name__ == "__main__":
    df = create_hybrid_dataset()