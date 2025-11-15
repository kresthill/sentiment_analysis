import os
from dotenv import load_dotenv
import tweepy
import time
import pandas as pd


load_dotenv()

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# Authenticate with API v2
client = tweepy.Client(
    bearer_token=BEARER_TOKEN,
    consumer_key=API_KEY,
    consumer_secret=API_SECRET,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET,
    wait_on_rate_limit=True
)


class MultiSentimentCollector:
    """Collect tweets for multiple sentiment categories"""
    
    def __init__(self):
        self.client = client
        
        # Define sentiment categories and their search queries
        self.sentiments = {
            'happy': {
                'id': 1,
                'queries': [
                    'happy -is:retweet lang:en',
                    'excited -is:retweet lang:en',
                    'joyful -is:retweet lang:en',
                    'delighted -is:retweet lang:en',
                    'cheerful -is:retweet lang:en'
                ],
                'keywords': ['happy', 'excited', 'joyful', 'delighted', 'cheerful', 
                           'wonderful', 'amazing', 'fantastic', 'love', 'blessed']
            },
            'sad': {
                'id': 2,
                'queries': [
                    'sad -is:retweet lang:en',
                    'depressed -is:retweet lang:en',
                    'heartbroken -is:retweet lang:en',
                    'disappointed -is:retweet lang:en',
                    'unhappy -is:retweet lang:en'
                ],
                'keywords': ['sad', 'depressed', 'heartbroken', 'disappointed', 
                           'unhappy', 'miserable', 'lonely', 'hurt', 'crying']
            },
            'angry': {
                'id': 3,
                'queries': [
                    'angry -is:retweet lang:en',
                    'furious -is:retweet lang:en',
                    'frustrated -is:retweet lang:en',
                    'mad -is:retweet lang:en',
                    'irritated -is:retweet lang:en'
                ],
                'keywords': ['angry', 'furious', 'frustrated', 'mad', 'irritated',
                           'annoyed', 'enraged', 'outraged', 'livid']
            },
            'fearful': {
                'id': 4,
                'queries': [
                    'scared -is:retweet lang:en',
                    'afraid -is:retweet lang:en',
                    'terrified -is:retweet lang:en',
                    'anxious -is:retweet lang:en',
                    'worried -is:retweet lang:en'
                ],
                'keywords': ['scared', 'afraid', 'terrified', 'anxious', 'worried',
                           'fearful', 'nervous', 'frightened', 'panic']
            }
        }
    
    def collect_sentiment_data(self, sentiment_name, tweets_per_query=100, 
                               max_total=500):
        """
        Collect tweets for a specific sentiment
        
        Parameters:
        -----------
        sentiment_name : str
            Name of sentiment ('happy', 'sad', 'angry', 'fearful')
        tweets_per_query : int
            Number of tweets to collect per query
        max_total : int
            Maximum total tweets for this sentiment
        """
        if sentiment_name not in self.sentiments:
            raise ValueError(f"Invalid sentiment. Choose from: {list(self.sentiments.keys())}")
        
        sentiment_info = self.sentiments[sentiment_name]
        sentiment_id = sentiment_info['id']
        queries = sentiment_info['queries']
        
        all_texts = []
        all_ids = []
        all_tweet_ids = []
        
        print(f"\n{'='*60}")
        print(f"Collecting {sentiment_name.upper()} tweets")
        print(f"Target: {max_total} tweets")
        print('='*60)
        
        for query_idx, query in enumerate(queries, 1):
            if len(all_texts) >= max_total:
                break
            
            print(f"\nQuery {query_idx}/{len(queries)}: '{query}'")
            
            try:
                tweets_collected = 0
                
                for response in tweepy.Paginator(
                    self.client.search_recent_tweets,
                    query=query,
                    max_results=100,  # Maximum per request
                    tweet_fields=['lang', 'created_at'],
                    limit=tweets_per_query // 100 + 1
                ):
                    if response.data:
                        for tweet in response.data:
                            if len(all_texts) >= max_total:
                                break
                            
                            all_texts.append(tweet.text)
                            all_ids.append(sentiment_id)
                            all_tweet_ids.append(tweet.id)
                            tweets_collected += 1
                            
                            if tweets_collected % 50 == 0:
                                print(f"  Collected: {tweets_collected} tweets from this query")
                    
                    if len(all_texts) >= max_total:
                        break
                
                print(f"  Total from this query: {tweets_collected} tweets")
                print(f"  Running total: {len(all_texts)}/{max_total}")
                
            except tweepy.TweepyException as e:
                print(f"  Error with query '{query}': {e}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': all_texts,
            'id': all_ids,
            'tweet_id': all_tweet_ids,
            'sentiment': sentiment_name
        })
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['text'])
        
        print(f"\n{'='*60}")
        print(f"Collection Summary for {sentiment_name.upper()}")
        print('='*60)
        print(f"Total tweets collected: {len(df)}")
        print(f"Unique tweets: {len(df)}")
        
        # Save to CSV
        filename = f'{sentiment_name}_tweets.csv'
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Saved to: {filename}")
        
        return df
    
    def collect_all_sentiments(self, tweets_per_sentiment=500):
        """Collect tweets for all sentiment categories"""
        all_dataframes = []
        
        print("\n" + "="*60)
        print("MULTI-SENTIMENT DATA COLLECTION")
        print("="*60)
        print(f"Collecting {tweets_per_sentiment} tweets per sentiment category")
        print(f"Total target: {tweets_per_sentiment * len(self.sentiments)} tweets")
        
        for sentiment_name in self.sentiments.keys():
            try:
                df = self.collect_sentiment_data(
                    sentiment_name, 
                    tweets_per_query=100,
                    max_total=tweets_per_sentiment
                )
                all_dataframes.append(df)
                
                # Brief pause between sentiment categories
                print("\nWaiting 10 seconds before next category...")
                time.sleep(10)
                
            except Exception as e:
                print(f"Error collecting {sentiment_name} tweets: {e}")
                continue
        
        # Combine all dataframes
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            
            # Save combined dataset
            combined_df.to_csv('all_sentiments.csv', index=False, encoding='utf-8')
            
            print("\n" + "="*60)
            print("FINAL COLLECTION SUMMARY")
            print("="*60)
            print(f"Total tweets collected: {len(combined_df)}")
            print("\nBreakdown by sentiment:")
            print(combined_df['sentiment'].value_counts())
            print("\nDataset saved as: all_sentiments.csv")
            
            return combined_df
        else:
            print("No data collected!")
            return None


if __name__ == "__main__":
    # Test authentication first
    try:
        me = client.get_me()
        print(f"✓ Authentication successful! Logged in as: {me.data.username}\n")
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        exit()
    
    # Initialize collector
    collector = MultiSentimentCollector()
    
    # Option 1: Collect all sentiments at once (may take time and hit rate limits)
    print("Choose collection mode:")
    print("1. Collect all sentiments (500 tweets each) - May take several hours")
    print("2. Collect one sentiment at a time")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '1':
        df = collector.collect_all_sentiments(tweets_per_sentiment=500)
    elif choice == '2':
        print("\nAvailable sentiments:")
        for i, sentiment in enumerate(collector.sentiments.keys(), 1):
            print(f"{i}. {sentiment}")
        
        sentiment_choice = input("\nEnter sentiment name: ").strip().lower()
        tweets_count = int(input("How many tweets to collect? (max 500 recommended): "))
        
        df = collector.collect_sentiment_data(sentiment_choice, max_total=tweets_count)
    else:
        print("Invalid choice!")