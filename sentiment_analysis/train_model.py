import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from feature_extraction import prepare_features


class SentimentClassifier:
    def __init__(self, model_type='naive_bayes'):
        """
        Initialize sentiment classifier
        
        Parameters:
        -----------
        model_type : str
            Type of model: 'naive_bayes', 'logistic_regression', 'svm', 'random_forest'
        """
        self.model_type = model_type
        
        if model_type == 'naive_bayes':
            self.model = MultinomialNB(alpha=1.0)
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'svm':
            self.model = LinearSVC(max_iter=1000, random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("Invalid model type")
    
    def train(self, X_train, y_train):
        """Train the model"""
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        print("Training completed!")
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            return None
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print("\n" + "="*50)
        print(f"Model: {self.model_type}")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {self.model_type}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{self.model_type}.png', dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved as confusion_matrix_{self.model_type}.png")
        plt.close()
    
    def save_model(self, filepath):
        """Save model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {filepath}")


def compare_models(X_train, X_test, y_train, y_test):
    """Train and compare multiple models"""
    models = ['naive_bayes', 'logistic_regression', 'svm', 'random_forest']
    results = {}
    
    for model_type in models:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper().replace('_', ' ')}")
        print('='*60)
        
        classifier = SentimentClassifier(model_type=model_type)
        classifier.train(X_train, y_train)
        metrics = classifier.evaluate(X_test, y_test)
        
        results[model_type] = metrics
        
        # Save the model
        classifier.save_model(f'model_{model_type}.pkl')
    
    # Plot comparison
    plot_model_comparison(results)
    
    return results


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
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=15)
    ax.legend()
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("\nModel comparison chart saved as model_comparison.png")
    plt.close()


if __name__ == "__main__":
    # Load preprocessed data
    df = pd.read_csv('../upset_preprocessed.csv')
    
    # Prepare features
    X_train, X_test, y_train, y_test, feature_extractor = prepare_features(df)
    
    # Compare all models
    results = compare_models(X_train, X_test, y_train, y_test)
    
    # Display summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    for model, metrics in results.items():
        print(f"\n{model.upper().replace('_', ' ')}:")
        for metric, value in metrics.items():
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")