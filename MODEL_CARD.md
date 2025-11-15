# Model Card: Twitter Multi-Class Sentiment Analysis

**Version:** 1.0 | **Date:** November 13, 2025 | **Author:** Kresthill  
**Organization:** OGLek Intel | **License:** Educational/Research Use

---

## Model Overview

**Purpose:** Multi-class emotion classification for English tweets  
**Task:** Classify tweets into Happy (😊), Sad (😢), Angry (😠), or Fearful (😨)  
**Best Performance:** 82.5% accuracy, 5-10ms inference  
**Framework:** scikit-learn, NLTK

### Available Models

| Model | Accuracy | Speed | Use Case |
|-------|----------|-------|----------|
| **Logistic Regression** ⭐ | 82.5% | 5-10ms | Recommended for production |
| Linear SVM | 80.5% | 5-15ms | High accuracy alternative |
| Naive Bayes | 75.5% | 2-5ms | Fast inference, resource-constrained |

---

## Training Data

**Dataset Size:** 796 tweets (199 per emotion)  
**Split:** 80% train (636) / 20% test (160)  
**Balance:** Perfectly balanced across 4 classes

**Critical Limitation:** Only Angry class uses real Twitter data. Happy, Sad, and Fearful use template-generated samples.

**Emotion Definitions:**
- **Happy:** Joy, excitement, gratitude, satisfaction
- **Sad:** Depression, disappointment, heartbreak, loneliness
- **Angry:** Frustration, rage, irritation, outrage
- **Fearful:** Anxiety, worry, terror, nervousness

---

## Technical Architecture

### Preprocessing Pipeline
1. Lowercase → Remove URLs/mentions → Remove special chars
2. Tokenize → Remove stopwords → Lemmatize (WordNet)
3. Filter tokens (min 3 chars)

### Feature Extraction
- **Method:** TF-IDF vectorization
- **Features:** 5,000 (unigrams + bigrams)
- **Config:** Min DF=2, Max DF=0.8, L2 norm

### Model Configuration
- **Logistic Regression:** One-vs-Rest, L2 regularization, lbfgs solver
- **No hyperparameter tuning** (small dataset)

---

## Performance Metrics

### Overall (Logistic Regression)
- **Accuracy:** 82.5% | **Precision:** 82.7% | **Recall:** 82.5% | **F1:** 82.5%

### Per-Emotion Performance

| Emotion | F1-Score | Notes |
|---------|----------|-------|
| Angry 😠 | 87.0% | Best (real Twitter data) |
| Fearful 😨 | 82.1% | Good |
| Happy 😊 | 81.9% | Good |
| Sad 😢 | 78.8% | Moderate |

### Confidence Distribution
- **>70%:** 60% of predictions (reliable)
- **50-70%:** 30% of predictions (moderate)
- **<50%:** 10% of predictions (review recommended)

---

## Intended Use

### ✅ Appropriate Uses
- Social media monitoring (brand sentiment, crisis detection)
- Customer feedback categorization
- Content moderation support
- Academic research and education

### ❌ Prohibited Uses
- Clinical/mental health diagnosis
- High-stakes decisions (hiring, legal, financial)
- Individual profiling or surveillance
- Non-English content
- Detecting sarcasm/irony

---

## Limitations

### Technical Constraints
1. **75% synthetic data** → Poor generalization to real-world expressions
2. **Small dataset (796 tweets)** → Limited model capacity
3. **English-only** → No multilingual support
4. **TF-IDF features** → No semantic understanding or context
5. **Single-label only** → Cannot detect mixed emotions
6. **Vocabulary-bound** → Fails on slang, neologisms, out-of-vocabulary words

### Known Biases
- **Platform bias:** Trained on Twitter; may not work on formal text
- **Temporal bias:** November 2025 data; language evolves
- **Sampling bias:** Angry tweets from keyword search only
- **Cultural bias:** Western emotion taxonomy
- **No demographic data** → Fairness across groups untested

### Common Errors
- **Sad ↔ Fearful confusion** (15% of errors) - overlapping vocabulary
- **Angry ↔ Sad confusion** (12% of errors) - frustration vs disappointment
- **Sarcasm fails** - "Great, another perfect day!" → Happy (wrong)
- **Short texts** - Insufficient features for accurate classification

---

## Ethical Considerations

**Privacy:** Uses public Twitter data; no personal identifiers stored  
**Misuse Risk:** Could enable emotional manipulation or surveillance  
**Mental Health:** NOT suitable for clinical assessment; may trivialize distress  
**Fairness:** Untested for demographic bias; may disadvantage neurodivergent communication

**Mitigation:** Clear usage guidelines, confidence scores, human oversight recommended

---

## System Requirements

**Minimum:** Python 3.8+, 2GB RAM, 500MB disk  
**Recommended:** Python 3.10+, 4GB RAM, multi-core CPU

**Key Dependencies:**
```
scikit-learn>=1.0.0, nltk>=3.6, pandas>=1.3.0
numpy>=1.21.0, matplotlib>=3.4.0, seaborn>=0.11.0
```

**Model Files:**
- `multiclass_vectorizer.pkl` (5-15 MB)
- `multiclass_model_logistic_regression.pkl` (1-5 MB)

**Speed:** 5-10ms per tweet, 100-500 tweets/second batch processing

---

## Future Improvements

### Priority Actions (1-3 months)
1. **Collect real data** for Happy, Sad, Fearful (500+ tweets each) → +10-15% accuracy
2. **Add features:** Sentiment lexicons, emoji analysis → +5-8% accuracy
3. **Hyperparameter tuning:** Grid search with cross-validation → +3-5% accuracy

### Medium-Term (3-6 months)
- Implement deep learning (BERT, LSTM) → +15-20% accuracy
- Enable multi-label classification
- Deploy as REST API

### Long-Term (6-12 months)
- Multilingual support (Spanish, French, German)
- Real-time streaming dashboard
- Advanced explainability (LIME/SHAP)

---

## Responsible AI

**Transparency:** Full documentation, open methodology, clear limitations  
**Accountability:** Version control, contact info, human oversight recommended  
**Privacy:** GDPR-compatible, public data only, no user tracking  
**Safety:** Not approved for clinical use; requires human review for sensitive applications

---

## Maintenance

**Retraining Schedule:**
- **Quarterly:** Update with new Twitter data
- **Annually:** Architecture review
- **As-needed:** When accuracy drops >5%

**Monitoring:**
- Track performance drift
- Detect vocabulary/language shifts
- Regular fairness audits

---

## Citation

```bibtex
@software{kresthill_twitter_sentiment_2025,
  author = {Kresthill},
  title = {Twitter Multi-Class Sentiment Analysis System},
  year = {2025},
  version = {1.0},
}
```

## License

**Educational/Research Use Only**

**Allowed:** ✅ Academic research, Education, Personal projects, Benchmarking  
**Prohibited:** ❌ Commercial use, Clinical applications, High-stakes decisions

**Disclaimer:** Provided "as-is" without warranties. Users assume all risks and must evaluate fitness for their specific use case.

---

**Document Version:** 1.0 | **Last Updated:** November 13, 2025  
**Status:** Active Development | **Feedback:** Welcome

*Following Mitchell et al. (2019) Model Card framework*
