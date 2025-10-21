import re
import nltk
import spacy
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import warnings

warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback if spaCy model not available
            self.nlp = None

    def detect_language(self, text):
        """Detect text language"""
        try:
            return detect(text)
        except (LangDetectException, Exception):
            return 'en'  # Default to English

    def is_english(self, text):
        """Check if text is English"""
        return self.detect_language(text) == 'en'

    def clean_text(self, text):
        """Comprehensive text cleaning"""
        if not text or not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z\s\.\,\!\?]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def tokenize_and_lemmatize(self, text):
        """Advanced tokenization and lemmatization"""
        # Clean text first
        text = self.clean_text(text)

        # Tokenize
        tokens = word_tokenize(text)

        # Lemmatize and filter
        processed_tokens = []
        for token in tokens:
            if len(token) > 2 and token not in self.stop_words and token.isalpha():
                lemma = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemma)

        return ' '.join(processed_tokens)

    def preprocess_batch(self, texts):
        """Preprocess a batch of texts"""
        return [self.tokenize_and_lemmatize(text) for text in texts]

    def preprocess_single(self, text):
        """Preprocess single text"""
        return self.tokenize_and_lemmatize(text)