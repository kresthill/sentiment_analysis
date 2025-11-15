import pickle
from data_preprocessing import TweetPreprocessor


class MultiClassSentimentPredictor:
    def __init__(self, model_path, vectorizer_path):
        """Load trained multi-class model and vectorizer"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        self.preprocessor = TweetPreprocessor()
        
        # Sentiment mapping
        self.sentiment_map = {
            1: '😊 Happy',
            2: '😢 Sad',
            3: '😠 Angry',
            4: '😨 Fearful'
        }
    
    def predict(self, text):
        """Predict sentiment of text"""
        # Preprocess
        cleaned = self.preprocessor.preprocess(text)
        
        # Vectorize
        features = self.vectorizer.transform([cleaned])
        
        # Predict
        prediction = self.model.predict(features)[0]
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(features)[0]
            confidence = max(probs)
            
            # Get all sentiment probabilities
            all_probs = {
                self.sentiment_map[i+1]: prob 
                for i, prob in enumerate(probs)
            }
        else:
            confidence = None
            all_probs = None
        
        return {
            'text': text,
            'sentiment': self.sentiment_map[prediction],
            'sentiment_id': prediction,
            'confidence': confidence,
            'all_probabilities': all_probs
        }
    
    def predict_batch(self, texts):
        """Predict sentiments for multiple texts"""
        return [self.predict(text) for text in texts]


def main():
    """Test the multi-class sentiment predictor"""
    
    # Load best model (Logistic Regression usually performs best)
    print("="*70)
    print("LOADING MULTI-CLASS SENTIMENT PREDICTOR")
    print("="*70)
    
    try:
        predictor = MultiClassSentimentPredictor(
            model_path='../models/multiclass_model_logistic_regression.pkl',
            vectorizer_path='../models/multiclass_vectorizer.pkl'
        )
        print("✓ Model and vectorizer loaded successfully!\n")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nPlease run 'python train_multiclass_model.py' first to train the models.")
        return
    
    # Test tweets covering all emotions
    test_tweets = [
        # Happy tweets
        "I'm so excited and happy about this wonderful news!",
        "What an amazing day! Best feeling ever!",
        "Feeling blessed and grateful for everything!",
        
        # Sad tweets
        "Feeling really sad and disappointed today",
        "Everything feels hopeless and depressing",
        "My heart is broken and I can't stop crying",
        
        # Angry tweets
        "This makes me absolutely furious and angry!",
        "Can't believe how frustrating this situation is",
        "I'm so irritated with this terrible service!",
        
        # Fearful tweets
        "I'm terrified and scared about what might happen",
        "So anxious and worried about the future",
        "This situation makes me very nervous and afraid"
    ]
    
    print("="*70)
    print("MULTI-CLASS SENTIMENT PREDICTIONS")
    print("="*70)
    
    for i, tweet in enumerate(test_tweets, 1):
        result = predictor.predict(tweet)
        print(f"\n[{i}] Tweet: {result['text']}")
        print(f"    Predicted Sentiment: {result['sentiment']}")
        
        if result['confidence']:
            print(f"    Confidence: {result['confidence']:.2%}")
            print("    All Probabilities:")
            for sent, prob in sorted(result['all_probabilities'].items(), 
                                     key=lambda x: x[1], reverse=True):
                print(f"      {sent}: {prob:.2%}")
        print("-" * 70)
    
    # Interactive mode
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("Enter your own tweets to analyze (or 'quit' to exit)")
    print("-" * 70)
    
    while True:
        user_input = input("\nYour tweet: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using the sentiment analyzer! 👋")
            break
        
        if not user_input:
            print("Please enter a valid tweet.")
            continue
        
        result = predictor.predict(user_input)
        print(f"\n🎯 Predicted Sentiment: {result['sentiment']}")
        
        if result['confidence']:
            print(f"   Confidence: {result['confidence']:.2%}")
            print("   Breakdown:")
            for sent, prob in sorted(result['all_probabilities'].items(), 
                                     key=lambda x: x[1], reverse=True):
                bar_length = int(prob * 30)
                bar = '█' * bar_length + '░' * (30 - bar_length)
                print(f"   {sent}: {bar} {prob:.1%}")


if __name__ == "__main__":
    main()