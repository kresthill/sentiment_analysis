"""
PRACTICAL MULTI-SENTIMENT DATA COLLECTION STRATEGY

Due to Twitter API rate limits, here's a realistic collection plan:
"""

import pandas as pd
import time
from datetime import datetime


class SentimentCollectionScheduler:
    """Plan and schedule sentiment data collection"""
    
    def __init__(self):
        self.collection_schedule = {
            'Day 1': {
                'sentiment': 'happy',
                'target': 500,
                'queries': ['happy', 'excited', 'joyful'],
                'time_slots': ['Morning', 'Afternoon', 'Evening']
            },
            'Day 2': {
                'sentiment': 'sad',
                'target': 500,
                'queries': ['sad', 'depressed', 'disappointed'],
                'time_slots': ['Morning', 'Afternoon', 'Evening']
            },
            'Day 3': {
                'sentiment': 'angry',
                'target': 500,
                'queries': ['angry', 'furious', 'frustrated'],
                'time_slots': ['Morning', 'Afternoon', 'Evening']
            },
            'Day 4': {
                'sentiment': 'fearful',
                'target': 500,
                'queries': ['scared', 'afraid', 'anxious'],
                'time_slots': ['Morning', 'Afternoon', 'Evening']
            }
        }
    
    def print_collection_plan(self):
        """Print the data collection plan"""
        print("\n" + "="*70)
        print("MULTI-SENTIMENT DATA COLLECTION PLAN")
        print("="*70)
        print("\nDue to Twitter API rate limits, spread collection over 4 days:")
        print("Each day, collect data in 3 sessions to avoid rate limits.\n")
        
        for day, info in self.collection_schedule.items():
            print(f"\n{day}: {info['sentiment'].upper()}")
            print(f"  Target: {info['target']} tweets")
            print(f"  Queries: {', '.join(info['queries'])}")
            print(f"  Time Slots: {', '.join(info['time_slots'])}")
            print(f"  Per Session: ~{info['target'] // len(info['time_slots'])} tweets")


def create_sample_dataset():
    """Create a small sample dataset for immediate testing"""
    print("\n" + "="*70)
    print("CREATING SAMPLE DATASET FOR TESTING")
    print("="*70)
    
    # Sample tweets for each sentiment
    samples = {
        'happy': [
            "I'm so excited about this amazing opportunity!",
            "Best day ever! Everything is going perfectly!",
            "Feeling blessed and grateful for all the love",
            "This makes me so happy! Can't stop smiling!",
            "Absolutely delighted with the results today!"
        ] * 20,  # 100 samples
        
        'sad': [
            "Feeling really down today, everything seems wrong",
            "So disappointed with how things turned out",
            "My heart is broken, can't stop crying",
            "Everything feels hopeless and sad right now",
            "Deeply hurt by what happened yesterday"
        ] * 20,  # 100 samples
        
        'angry': [
            "I'm so frustrated with this terrible service!",
            "This makes me absolutely furious!",
            "Can't believe how angry this situation makes me",
            "Extremely irritated by this incompetence",
            "This is unacceptable! I'm so mad right now!"
        ] * 20,  # 100 samples
        
        'fearful': [
            "I'm really scared about what might happen",
            "Feeling so anxious and worried about everything",
            "Terrified of the possible outcomes here",
            "This situation makes me very nervous and afraid",
            "Constant fear and panic about the future"
        ] * 20  # 100 samples
    }
    
    # Create sentiment ID mapping
    sentiment_ids = {'happy': 1, 'sad': 2, 'angry': 3, 'fearful': 4}
    
    # Build dataframe
    all_data = []
    for sentiment, tweets in samples.items():
        for tweet in tweets:
            all_data.append({
                'text': tweet,
                'id': sentiment_ids[sentiment],
                'sentiment': sentiment,
                'tweet_id': f'sample_{len(all_data)}'
            })
    
    df = pd.DataFrame(all_data)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    df.to_csv('sample_multi_sentiment.csv', index=False, encoding='utf-8')
    
    print(f"\n✓ Created sample dataset with {len(df)} tweets")
    print("\nBreakdown by sentiment:")
    print(df['sentiment'].value_counts())
    print("\n✓ Saved as: sample_multi_sentiment.csv")
    print("\nYou can use this to test your model while collecting real data!")
    
    return df


def merge_datasets():
    """Merge your existing angry tweets with new sentiments"""
    print("\n" + "="*70)
    print("MERGING EXISTING DATA WITH NEW COLLECTIONS")
    print("="*70)
    
    datasets = []
    
    # Load existing angry tweets
    try:
        angry_df = pd.read_csv('../upset.csv')
        angry_df['sentiment'] = 'angry'
        angry_df['id'] = 3  # Angry ID
        print(f"✓ Loaded {len(angry_df)} angry tweets")
        datasets.append(angry_df)
    except FileNotFoundError:
        print("✗ upset.csv not found")
    
    # Load new sentiment files if they exist
    for sentiment, sent_id in [('happy', 1), ('sad', 2), ('fearful', 4)]:
        filename = f'{sentiment}_tweets.csv'
        try:
            df = pd.read_csv(filename)
            print(f"✓ Loaded {len(df)} {sentiment} tweets")
            datasets.append(df)
        except FileNotFoundError:
            print(f"  {filename} not found - will collect later")
    
    if len(datasets) > 1:
        # Merge all datasets
        combined_df = pd.concat(datasets, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['text'])
        
        combined_df.to_csv('all_sentiments_combined.csv', index=False, encoding='utf-8')
        
        print(f"\n✓ Combined dataset created: {len(combined_df)} total tweets")
        print("\nBreakdown:")
        print(combined_df['sentiment'].value_counts())
        print("\n✓ Saved as: all_sentiments_combined.csv")
        
        return combined_df
    else:
        print("\nNeed to collect more sentiment categories first!")
        return None


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MULTI-SENTIMENT COLLECTION HELPER")
    print("="*70)
    
    print("\nChoose an option:")
    print("1. View 4-day collection plan")
    print("2. Create sample dataset for immediate testing")
    print("3. Merge existing datasets")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    if choice == '1':
        scheduler = SentimentCollectionScheduler()
        scheduler.print_collection_plan()
    elif choice == '2':
        df = create_sample_dataset()
    elif choice == '3':
        df = merge_datasets()
    else:
        print("Invalid choice!")