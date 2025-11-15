# Twitter Sentiment Analysis Project

A comprehensive multi-class sentiment analysis system for classifying tweets into four emotional categories: Happy, Sad, Angry, and Fearful.

## 📁 Project Structure

```
twitter_project/
├── .env                                # API credentials (not tracked in git)
├── .gitignore                         # Git ignore rules
├── README.md                          # This file
│
├── data/                              # All data files
│   ├── raw/                          # Raw collected tweets
│   │   ├── upset.csv                # Real angry tweets from Twitter
│   │   ├── happy_tweets.csv         # Happy tweets (to be collected)
│   │   ├── sad_tweets.csv           # Sad tweets (to be collected)
│   │   └── fearful_tweets.csv       # Fearful tweets (to be collected)
│   ├── processed/                   # Processed datasets
│   │   ├── multi_sentiment_dataset.csv
│   │   └── multi_sentiment_preprocessed.csv
│   └── unsmile_words.csv           # Reference data
│
├── models/                           # Trained models
│   ├── multiclass_vectorizer.pkl
│   ├── multiclass_model_naive_bayes.pkl
│   ├── multiclass_model_logistic_regression.pkl
│   └── multiclass_model_svm.pkl
│
├── results/                          # Training results and visualizations
│   ├── confusion_matrix_*.png
│   └── multiclass_model_comparison.png
│
├── sentiment_analysis/               # Main analysis package
│   ├── data_preprocessing.py        # Text preprocessing utilities
│   ├── feature_extraction.py        # Feature extraction (TF-IDF)
│   ├── train_model.py              # Single-class model training
│   ├── train_multiclass_model.py   # Multi-class model training
│   ├── predict.py                  # Single-class predictions
│   ├── predict_multiclass.py       # Multi-class predictions
│   ├── manual_collection_guide.py  # Data collection guide
│   ├── create_hybrid_dataset.py    # Create training dataset
│   └── collect_multi_sentiment.py  # Twitter data collection
│
├── notebook/                         # Jupyter notebooks
│   └── tweepy.ipynb                # Twitter API exploration
│
├── venvtweet/                       # Virtual environment
└── twitter.py                       # Original collection script
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or navigate to project
cd ~/OneDrive/Desktop/twitter_project

# Activate virtual environment
source venvtweet/Scripts/activate  # Git Bash
# OR
.\venvtweet\Scripts\activate       # PowerShell

# Install dependencies
pip install pandas numpy scikit-learn nltk textblob matplotlib seaborn tweepy python-dotenv
```

### 2. Create Multi-Sentiment Dataset

```bash
cd sentiment_analysis
python create_hybrid_dataset.py
```

This creates a balanced dataset with:
- 250 **real** angry tweets from Twitter
- 250 sample happy tweets
- 250 sample sad tweets  
- 250 sample fearful tweets

### 3. Train Models

```bash
python train_multiclass_model.py
```

This will:
- Preprocess all tweets
- Extract TF-IDF features
- Train 3 models (Naive Bayes, Logistic Regression, SVM)
- Save models to `models/` folder
- Save results to `results/` folder

### 4. Make Predictions

```bash
python predict_multiclass.py
```

Features:
- Test with predefined tweets
- Interactive mode for custom input
- Confidence scores and probability distributions

## 📊 Dataset Information

### Current Status
- **Total Tweets**: 1,000
- **Happy**: 250 (sample)
- **Sad**: 250 (sample)
- **Angry**: 250 (real from Twitter)
- **Fearful**: 250 (sample)

### Sentiment Labels
- **1**: Happy 😊
- **2**: Sad 😢
- **3**: Angry 😠
- **4**: Fearful 😨

## 🔧 Key Components

### Data Preprocessing (`data_preprocessing.py`)
- URL removal
- Mention removal (@username)
- Hashtag cleaning
- Special character removal
- Tokenization
- Stopword removal
- Lemmatization

### Feature Extraction (`feature_extraction.py`)
- TF-IDF vectorization
- Unigrams and bigrams
- Maximum 5,000 features
- Document frequency filtering

### Models
1. **Naive Bayes**: Fast baseline model
2. **Logistic Regression**: Best overall performance
3. **Support Vector Machine**: High accuracy

## 📈 Expected Performance

Based on the hybrid dataset:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | ~75% | ~75% | ~75% | ~75% |
| Logistic Regression | ~80-85% | ~81% | ~80% | ~80% |
| SVM | ~78-82% | ~79% | ~78% | ~78% |

## 🎯 Usage Examples

### Python Script

```python
from sentiment_analysis.predict_multiclass import MultiClassSentimentPredictor

# Load model
predictor = MultiClassSentimentPredictor(
    model_path='models/multiclass_model_logistic_regression.pkl',
    vectorizer_path='models/multiclass_vectorizer.pkl'
)

# Predict
result = predictor.predict("I'm so happy today!")
print(result['sentiment'])  # 😊 Happy
print(result['confidence'])  # 0.87
```

### Command Line

```bash
cd sentiment_analysis
python predict_multiclass.py
# Then enter tweets interactively
```

## 📝 Collecting Real Data

### Option 1: Twitter API (Slow but Real)

```bash
cd sentiment_analysis
python collect_multi_sentiment.py
```

Note: Rate limits apply. Collect ~100 tweets per emotion per day.

### Option 2: Manual Collection Schedule

**Day 1**: Collect happy tweets
**Day 2**: Collect sad tweets  
**Day 3**: Collect angry tweets (done ✓)
**Day 4**: Collect fearful tweets

See `manual_collection_guide.py` for detailed instructions.

## 🛠️ Troubleshooting

### Issue: ModuleNotFoundError

```bash
# Make sure you're in the correct directory
cd sentiment_analysis

# Check imports use relative paths
# Should work from sentiment_analysis folder
```

### Issue: File Not Found

```bash
# Check you're in sentiment_analysis folder
pwd  # Should end with /sentiment_analysis

# Or use absolute paths
python ~/OneDrive/Desktop/twitter_project/sentiment_analysis/train_multiclass_model.py
```

### Issue: Empty Predictions

```bash
# Train the model first
python train_multiclass_model.py

# Then make predictions
python predict_multiclass.py
```

## 🔮 Future Improvements

1. **Collect More Real Data**
   - Replace sample tweets with real Twitter data
   - Target: 500-1000 tweets per sentiment

2. **Add More Emotions**
   - Surprise
   - Disgust
   - Neutral

3. **Deep Learning Models**
   - LSTM/GRU networks
   - BERT-based models
   - Transfer learning

4. **Web Interface**
   - Flask/Streamlit dashboard
   - Real-time analysis
   - Visualization tools

5. **API Deployment**
   - REST API
   - Docker containerization
   - Cloud deployment

## 📚 Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
nltk>=3.6
textblob>=0.15.3
matplotlib>=3.4.0
seaborn>=0.11.0
tweepy>=4.0.0
python-dotenv>=0.19.0
```

## 👤 Author

**Kresthill**
- Data & ML/AI Engineer at Divinity Sound
- Specialization: Advanced Data Science & ML/AI
- Education: 10Alytics, Imperial College London

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- Twitter API for data access
- scikit-learn for ML algorithms
- NLTK for NLP preprocessing

---

**Last Updated**: November 13, 2025