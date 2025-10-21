import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Path configurations - Update this path to match your dataset location
DATA_PATH = BASE_DIR / "data" / "bbc_news.csv"  # or whatever your file is named

# Common BBC dataset file names to try
BBC_DATASET_NAMES = [
    "bbc_news.csv",
    "bbc_articles.csv",
    "bbc_dataset.csv",
    "news_dataset.csv",
    "BBC_News.csv"
]

# Model configurations
MODEL_CONFIG = {
    'vectorizer': {
        'max_features': 5000,
        'ngram_range': (1, 2),
        'min_df': 2,
        'max_df': 0.8
    },
    'naive_bayes': {
        'alpha': 0.1
    },
    'svm': {
        'kernel': 'linear',
        'C': 1.0,
        'probability': True,
        'random_state': 42
    },
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    }
}

# Categories (standard BBC categories)
CATEGORIES = ['business', 'entertainment', 'politics', 'sport', 'tech']

# Model directory
MODELS_DIR = BASE_DIR / "models" / "trained_models"

# App settings
MAX_TEXT_LENGTH = 10000
MIN_TEXT_LENGTH = 50
REQUEST_TIMEOUT = 15