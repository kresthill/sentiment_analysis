import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import pickle

class FeatureExtractor:
    def __init__(self, method='tfidf', max_features=5000):
        """
        Initialize feature extractor
        
        Parameters:
        -----------
        method : str, default='tfidf'
            Feature extraction method: 'tfidf' or 'bow' (bag of words)
        max_features : int, default=5000
            Maximum number of features to extract
        """
        self.method = method
        self.max_features = max_features
        
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),  # Unigrams and bigrams
                min_df=2,  # Minimum document frequency
                max_df=0.8  # Maximum document frequency
            )
        elif method == 'bow':
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
        else:
            raise ValueError("Method must be 'tfidf' or 'bow'")
    
    def fit_transform(self, texts):
        """Fit vectorizer and transform texts"""
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        """Transform texts using fitted vectorizer"""
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """Get feature names"""
        return self.vectorizer.get_feature_names_out()
    
    def save_vectorizer(self, filepath):
        """Save vectorizer to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"Vectorizer saved to {filepath}")
    
    def load_vectorizer(self, filepath):
        """Load vectorizer from file"""
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)
        print(f"Vectorizer loaded from {filepath}")


def prepare_features(df, text_column='cleaned_text', label_column='id', 
                     test_size=0.2, method='tfidf'):
    """
    Prepare features for model training
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    text_column : str
        Column containing cleaned text
    label_column : str
        Column containing labels
    test_size : float
        Proportion of test set
    method : str
        Feature extraction method
    
    Returns:
    --------
    X_train, X_test, y_train, y_test, feature_extractor
    """
    print(f"Preparing features using {method.upper()} method...")
    
    # Extract texts and labels
    texts = df[text_column].values
    labels = df[label_column].values
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor(method=method)
    
    # Split data
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    print(f"Training set size: {len(X_train_text)}")
    print(f"Test set size: {len(X_test_text)}")
    
    # Extract features
    X_train = feature_extractor.fit_transform(X_train_text)
    X_test = feature_extractor.transform(X_test_text)
    
    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Save vectorizer
    feature_extractor.save_vectorizer('vectorizer.pkl')
    
    return X_train, X_test, y_train, y_test, feature_extractor


if __name__ == "__main__":
    # Load preprocessed data
    df = pd.read_csv('../upset_preprocessed.csv')
    
    # Prepare features
    X_train, X_test, y_train, y_test, feature_extractor = prepare_features(df)
    
    # Display top features
    feature_names = feature_extractor.get_feature_names()
    print(f"\nTop 20 features:")
    print(feature_names[:20])