import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from data_preprocessing import TweetPreprocessor


def load_and_prepare_data(filepath='../data/processed/multi_sentiment_dataset.csv'):
    """Load and preprocess multi-sentiment data"""
    print("="*70)
    print("LOADING MULTI-SENTIMENT DATASET")
    print("="*70)
    
    df = pd.read_csv(filepath)
    print(f"\n✓ Loaded {len(df)} tweets")
    print("\nSentiment distribution:")
    print(df['sentiment'].value_counts())
    print(df['id'].value_counts())
    
    # Preprocess texts
    print("\nPreprocessing tweets...")
    preprocessor = TweetPreprocessor()
    df['cleaned_text'] = df['text'].apply(preprocessor.preprocess)
    
    # Remove empty tweets
    df = df[df['cleaned_text'].str.strip() != '']
    
    print(f"✓ After preprocessing: {len(df)} tweets")
    
    # Save preprocessed data
    preprocessed_path = '../data/processed/multi_sentiment_preprocessed.csv'
    df.to_csv(preprocessed_path, index=False, encoding='utf-8')
    print(f"✓ Preprocessed data saved: {preprocessed_path}")
    
    return df


def extract_features(df, test_size=0.2):
    """Extract TF-IDF features for multi-class classification"""
    print("\n" + "="*70)
    print("FEATURE EXTRACTION")
    print("="*70)
    
    X = df['cleaned_text'].values
    y = df['id'].values  # Sentiment IDs: 1=happy, 2=sad, 3=angry, 4=fearful
    
    # Split data
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train_text)} tweets")
    print(f"Test set: {len(X_test_text)} tweets")
    
    # Initialize vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    
    # Transform
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    
    print(f"\nFeature matrix shape: {X_train.shape}")
    
    # Save vectorizer to models folder
    vectorizer_path = '../models/multiclass_vectorizer.pkl'
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"✓ Vectorizer saved: {vectorizer_path}")
    
    return X_train, X_test, y_train, y_test, vectorizer


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train multiple models for multi-class classification"""
    
    sentiment_names = {1: 'Happy', 2: 'Sad', 3: 'Angry', 4: 'Fearful'}
    
    models = {
        'Naive Bayes': MultinomialNB(alpha=1.0),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr'),
        'SVM': LinearSVC(max_iter=1000, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print("\n" + "="*70)
        print(f"TRAINING: {name}")
        print("="*70)
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\nAccuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=list(sentiment_names.values()),
            zero_division=0
        ))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, list(sentiment_names.values()), name)
        
        # Save model to models folder
        model_filename = f'../models/multiclass_model_{name.lower().replace(" ", "_")}.pkl'
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Model saved as: {model_filename}")
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    # Plot comparison
    plot_model_comparison(results)
    
    return results


def plot_confusion_matrix(cm, classes, model_name):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=classes, yticklabels=classes
    )
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    filename = f'../results/confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {filename}")
    plt.close()


def plot_model_comparison(results):
    """Plot comparison of model performances"""
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        ax.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Multi-Class Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/multiclass_model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Model comparison saved: ../results/multiclass_model_comparison.png")
    plt.close()


if __name__ == "__main__":
    # Load data
    df = load_and_prepare_data()
    
    # Extract features
    X_train, X_test, y_train, y_test, vectorizer = extract_features(df)
    
    # Train and evaluate
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    for model, metrics in results.items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
    
    print("\n" + "="*70)
    print("ALL FILES SAVED TO:")
    print("="*70)
    print("  Models: twitter_project/models/")
    print("  Results: twitter_project/results/")
    print("  Data: twitter_project/data/processed/")