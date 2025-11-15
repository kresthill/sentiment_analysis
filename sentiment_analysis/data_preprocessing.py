import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Add this after the import statements
def download_nltk_data():
    """Download all required NLTK data"""
    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {resource}: {e}")

# Call it immediately
download_nltk_data()

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class TweetPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_tweet(self, text):
        """Clean individual tweet"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (keep the word, remove #)
        text = re.sub(r'#', '', text)
        
        # Remove RT (retweet indicator)
        text = re.sub(r'\brt\b', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text"""
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def preprocess(self, text):
        """Full preprocessing pipeline"""
        text = self.clean_tweet(text)
        text = self.tokenize_and_lemmatize(text)
        return text


def load_and_preprocess_data(filepath):
    """Load and preprocess tweet data"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, encoding='utf-8')
    
    print(f"Loaded {len(df)} tweets")
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst few tweets:")
    print(df.head())
    
    # Initialize preprocessor
    preprocessor = TweetPreprocessor()
    
    # Add cleaned text column
    print("\nPreprocessing tweets...")
    df['cleaned_text'] = df['text'].apply(preprocessor.preprocess)
    
    # Remove empty tweets after cleaning
    df = df[df['cleaned_text'].str.strip() != '']
    
    print(f"\nAfter preprocessing: {len(df)} tweets remaining")
    
    # Save preprocessed data
    output_file = filepath.replace('.csv', '_preprocessed.csv')
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Preprocessed data saved to: {output_file}")
    
    return df


if __name__ == "__main__":
    # Load and preprocess the data
    df = load_and_preprocess_data('../upset.csv')
    
    # Display sample of original vs cleaned
    print("\n" + "="*80)
    print("Sample: Original vs Cleaned Text")
    print("="*80)
    for i in range(5):
        print(f"\nOriginal: {df.iloc[i]['text']}")
        print(f"Cleaned:  {df.iloc[i]['cleaned_text']}")