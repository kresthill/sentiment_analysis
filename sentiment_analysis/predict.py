import pickle
from data_preprocessing import TweetPreprocessor


class SentimentPredictor:
    def __init__(self, model_path, vectorizer_path):
        """Load trained model and vectorizer"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        self.preprocessor = TweetPreprocessor()
        
        # Sentiment labels
        self.sentiment_labels = {
            1: 'Angry',
            # Add more labels as you collect more data
            # 2: 'Sad',
            # 3: 'Happy',
            # 4: 'Fearful'
        }
    
    def predict_sentiment(self, text):
        """Predict sentiment of a single text"""
        # Preprocess
        cleaned_text = self.preprocessor.preprocess(text)
        
        # Vectorize
        features = self.vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.model.predict(features)[0]
        
        # Get probability if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)[0]
            confidence = max(probabilities)
        else:
            confidence = None
        
        sentiment = self.sentiment_labels.get(prediction, 'Unknown')
        
        return {
            'text': text,
            'cleaned_text': cleaned_text,
            'sentiment': sentiment,
            'sentiment_id': prediction,
            'confidence': confidence
        }
    
    def predict_batch(self, texts):
        """Predict sentiments for multiple texts"""
        results = []
        for text in texts:
            result = self.predict_sentiment(text)
            results.append(result)
        return results


if __name__ == "__main__":
    # Load the best model (replace with your chosen model)
    predictor = SentimentPredictor(
        model_path='model_logistic_regression.pkl',
        vectorizer_path='vectorizer.pkl'
    )
    
    # Test predictions
    test_tweets = [
        "I'm so frustrated with this terrible service!",
        "This makes me absolutely furious!",
        "I hate how this company treats customers",
        "Extremely disappointed and angry",
        "This is unacceptable behavior!"
    ]
    
    print("="*80)
    print("SENTIMENT PREDICTIONS")
    print("="*80)
    
    for tweet in test_tweets:
        result = predictor.predict_sentiment(tweet)
        print(f"\nTweet: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        if result['confidence']:
            print(f"Confidence: {result['confidence']:.2%}")