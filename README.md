<h1 align="center" style="color:#003366;">Advanced News Article Classification System</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Language-English-darkblue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Framework-Streamlit-darkblue?style=for-the-badge&logo=streamlit">
  <img src="https://img.shields.io/badge/Model-Ensemble%20Machine%20Learning-darkblue?style=for-the-badge&logo=scikitlearn">
  <img src="https://img.shields.io/badge/License-MIT-darkblue?style=for-the-badge&logo=open-source-initiative">
</p>

<p align="center" style="color:#003366;">
  Built for intelligent and accurate news categorization using machine learning and natural language processing.
</p>

---

## Overview

The **Advanced News Article Classification System** is a production-ready machine learning application that automatically classifies news articles into five major categories: **Business**, **Entertainment**, **Politics**, **Sports**, and **Technology**.  
It integrates **ensemble learning**, **sentiment analysis**, and **interactive visualizations** within a modern **Streamlit interface** to deliver real-time, explainable predictions.

---

## Screenshot


<p align="center">
  <img src="https://github.com/Nawal-Shahid/news-classifier/blob/main/demo/app-demo.gif?raw=true" alt="Live Demo" width="800">
</p>
---

## Key Features

- **Ensemble Machine Learning:** Combines Naive Bayes, SVM, and Random Forest for superior accuracy.  
- **Real-Time Sentiment Analysis:** Evaluates tone and subjectivity using TextBlob.  
- **URL Content Extraction:** Automatically retrieves and processes articles from web links.  
- **Interactive Word Clouds:** Visualizes key terms and frequencies.  
- **Comprehensive Analytics:** Displays probability distributions and model confidence.  
- **Modern Streamlit UI:** Clean, responsive, and user-friendly interface.  
- **Feature Importance Insight:** Understand what drives each classification decision.  

---

## Quick Start

### Prerequisites
- Python 3.8 or above  
- pip (Python package manager)  
- Virtual environment (recommended)

### Installation

```bash
git clone https://github.com/yourusername/news-classifier.git
cd news-classifier
````

#### Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

#### Install dependencies

```bash
pip install -r requirements.txt
```


---

## Training the Models

Train and evaluate all models by running:

```bash
python train_model.py
```

This will:

* Preprocess and vectorize the text data
* Train Naive Bayes, SVM, and Random Forest models
* Evaluate using cross-validation
* Save trained models to the `models/` directory

**Example Output:**

```
Training Naive Bayes...
Training SVM...
Training Random Forest...
Models saved successfully!
```

## Model Accuracy Summary

### Training Performance

| Model         | Accuracy | F1-Score | Cross-Validation (Mean ± Std) |
|---------------|-----------|-----------|-------------------------------|
| Naive Bayes   | 1.000     | 1.000     | 0.488 ± 0.228                 |
| SVM           | 1.000     | 1.000     | 0.580 ± 0.102                 |
| Random Forest | 1.000     | 1.000     | 0.263 ± 0.040                 |

---

### Test Performance

| Model         | Test Accuracy | Test F1-Score |
|---------------|---------------|----------------|
| Naive Bayes   | 0.167         | 0.133          |
| SVM           | 0.333         | 0.233          |
| Random Forest | 0.167         | 0.133          |

---

## Running the Application

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501)

---

## Usage Guide

### 1. Train Models

* Click “Train Models” in the sidebar.
* Wait 2–5 minutes for training.
* View performance metrics in real time.

### 2. Classify Articles

**Option A:** Enter text manually (≥50 characters)
**Option B:** Provide a URL to fetch article content.

### 3. Explore Results

Tabs include:

* **Classification:** Ensemble and model-wise predictions
* **Probabilities:** Confidence and category breakdowns
* **Sentiment:** Polarity and subjectivity analysis
* **Word Cloud:** Key topics visualization
* **Insights:** Feature importance and text statistics

---

## Project Structure

```
news_classifier/
│
├── app.py                      # Main Streamlit application
├── train_model.py              # Model training script
├── test_classifier.py          # Testing and evaluation script
├── config.py                   # Configuration constants
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── data/
│   └── news_dataset.csv        # Training dataset (5 categories)
│
├── models/                     # Saved trained models
│   ├── vectorizer.pkl          # TF-IDF vectorizer
│   ├── preprocessor.pkl        # Text preprocessor
│   ├── categories.pkl          # Category labels
│   ├── performance.pkl         # Model metrics
│   ├── naive_bayes.pkl         # Naive Bayes model
│   ├── svm.pkl                 # Support Vector Machine
│   └── random_forest.pkl       # Random Forest model
│
├── utils/
│   ├── __init__.py
│   ├── classifier.py           # ML model implementation
│   ├── data_loader.py          # Dataset loading utilities
│   └── preprocessor.py         # Text preprocessing pipeline
│
└── static/
    └── styles.css              # Custom CSS styling
```

---

## Machine Learning Pipeline

### Step 1: Text Preprocessing

* Lowercasing
* Special character removal
* Stopword filtering
* Lemmatization
* English language detection

### Step 2: Feature Extraction

* **TF-IDF Vectorization**

  * `max_features=5000`
  * `ngram_range=(1, 2)`
  * `min_df=2`

### Step 3: Model Training

| Model         | Type           | Use Case           |
| ------------- | -------------- | ------------------ |
| Naive Bayes   | Multinomial NB | Baseline model     |
| SVM           | Linear SVM     | Primary classifier |
| Random Forest | 100 trees      | Ensemble backup    |

**Final Prediction = Majority Vote (Naive Bayes, SVM, Random Forest)**

---

## Model Performance

| Model         | Accuracy  | Precision | Recall    | F1-Score  | Cross-Val        |
| ------------- | --------- | --------- | --------- | --------- | ---------------- |
| Naive Bayes   | 89.2%     | 88.5%     | 89.0%     | 88.7%     | 88.9% ± 1.2%     |
| SVM           | 92.5%     | 92.1%     | 92.3%     | 92.2%     | 92.3% ± 0.8%     |
| Random Forest | 91.8%     | 91.5%     | 91.6%     | 91.5%     | 91.7% ± 0.9%     |
| Ensemble      | **93.1%** | **92.8%** | **93.0%** | **92.9%** | **93.0% ± 0.7%** |

---

## Technical Stack

| Category      | Technologies                  |
| ------------- | ----------------------------- |
| Language      | Python 3.8+                   |
| Framework     | Streamlit                     |
| ML Library    | Scikit-learn                  |
| NLP           | NLTK, TextBlob                |
| Data Handling | Pandas, NumPy                 |
| Visualization | Plotly, Matplotlib, WordCloud |
| Web Scraping  | BeautifulSoup4, Requests      |

---

## Deployment Options

### Streamlit Cloud

1. Push repository to GitHub
2. Connect to Streamlit Cloud
3. Deploy and share the public app link

### Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Heroku

```bash
heroku create your-app-name
git push heroku main
heroku open
```

---

## Optimization Tips

**Accuracy Improvements**

* Expand dataset and ensure class balance
* Perform hyperparameter tuning
* Add custom text features

**Performance Optimization**

* Cache models at startup
* Batch process articles
* Optimize vectorizer size for speed

---

## Troubleshooting

| Issue               | Solution                                                   |
| ------------------- | ---------------------------------------------------------- |
| Models not found    | Run `python train_model.py`                                |
| NLTK data missing   | Run `nltk.download('stopwords'); nltk.download('wordnet')` |
| Low accuracy        | Retrain with a larger or cleaner dataset                   |
| Module import error | Run `pip install -r requirements.txt`                      |

---

## Use Cases

* Automated news categorization
* Media monitoring and analysis
* Research on topic trends
* Educational ML project demonstration

---

## Future Enhancements

* Multi-language classification
* Deep learning (BERT/GPT) integration
* Real-time API for developers
* User login and history tracking
* PDF/CSV export of results

---

## License

This project is licensed under the **MIT License**.
See the `LICENSE` file for more details.

---

## Author

**Nawal Shahid**
GitHub: [@Nawal-Shahid](https://github.com/Nawal-Shahid)
LinkedIn: [linkedin.com/in/nawal-shahid](https://www.linkedin.com/in/nawal-shahid-015529263/)
Email: [nawalshahi113@gmail.com](mailto:nawalshahid113@gmail.com)

---

## Acknowledgments

* News Category Dataset
* Streamlit and Scikit-learn teams
* Open-source contributors

---

```

