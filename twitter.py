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
BEARER_TOKEN = os.getenv("BEARER_TOKEN")  # You'll need to add this to your .env file

# Authenticate with API v2 using Bearer Token
client = tweepy.Client(
    bearer_token=BEARER_TOKEN,
    consumer_key=API_KEY,
    consumer_secret=API_SECRET,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET,
    wait_on_rate_limit=True
)

# Test authentication
try:
    me = client.get_me()
    print(f"Authentication successful! Logged in as: {me.data.username}")
except Exception as e:
    print(f"Authentication failed: {e}")
    exit()

# Search query
searchquery = "angry -is:retweet lang:en"  # Excludes retweets and filters for English

# Configuration
total_number = 1500 # total number of tweets to collect
max_results = 100    # tweets per request (10-100 for API v2)

text = []
tweet_ids = []
count = 0

print(f"Starting to collect up to {total_number} tweets with query: '{searchquery}'")
print("Note: Free tier has limitations on tweet volume and search recency\n")

try:
    # Use Paginator for API v2
    for response in tweepy.Paginator(
        client.search_recent_tweets,
        query=searchquery,
        max_results=max_results,
        tweet_fields=['lang', 'created_at'],
        limit=total_number // max_results + 1
    ):
        if response.data:
            for tweet in response.data:
                if len(text) >= total_number:
                    break
                
                text.append(tweet.text)
                tweet_ids.append(tweet.id)
                count += 1
                
                if count % 100 == 0:
                    print(f"Collected {count} tweets...")
        
        if len(text) >= total_number:
            break
    
except tweepy.TweepyException as e:
    print(f"Error occurred: {e}")
except KeyboardInterrupt:
    print("\nCollection interrupted by user.")

print(f"\nCollection complete! Total tweets collected: {len(text)}")

# Create dataframe
if text:
    d = {
        "text": text, 
        "id": [1] * len(text),  # 1 is angry
        "tweet_id": tweet_ids
    }
    df = pd.DataFrame(data=d)
    
    df.to_csv('upset.csv', header=True, index=False, encoding='utf-8')
    print(f"Saved {len(text)} tweets to upset.csv")
else:
    print("No tweets collected. Please check your API access level.")