import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
import os

warnings.filterwarnings('ignore')

from config import MODEL_CONFIG, MODELS_DIR, CATEGORIES
from .preprocessor import TextPreprocessor


class NewsClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(**MODEL_CONFIG['vectorizer'])
        self.models = {
            'Naive Bayes': MultinomialNB(**MODEL_CONFIG['naive_bayes']),
            'SVM': SVC(**MODEL_CONFIG['svm']),
            'Random Forest': RandomForestClassifier(**MODEL_CONFIG['random_forest'])
        }
        self.preprocessor = TextPreprocessor()
        self.is_trained = False
        self.performance_metrics = {}
        self.categories = CATEGORIES
        
        # Auto-load models on initialization
        self._auto_load_models()

    def _auto_load_models(self):
        """Automatically load models if they exist, without raising errors"""
        try:
            # Check if model files exist
            required_files = [
                'vectorizer.pkl', 'preprocessor.pkl', 'categories.pkl',
                'naive_bayes.pkl', 'svm.pkl', 'random_forest.pkl'
            ]
            
            all_files_exist = all((MODELS_DIR / file).exists() for file in required_files)
            
            if all_files_exist:
                success = self.load_models(silent=True)
                if success:
                    print("✅ Models auto-loaded successfully on startup!")
                else:
                    print("⚠️ Models exist but couldn't be loaded. Training required.")
            else:
                print("ℹ️ No pre-trained models found. Training required.")
                
        except Exception as e:
            print(f"⚠️ Auto-load attempt failed: {e}")

    def train(self, texts, labels):
        """Train all models with comprehensive evaluation"""
        print("Preprocessing texts...")
        processed_texts = self.preprocessor.preprocess_batch(texts)

        print("Vectorizing texts...")
        X = self.vectorizer.fit_transform(processed_texts)

        print("Training models...")
        self.model_performance = {}
        self.cross_val_scores = {}

        # Cross-validation setup
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in self.models.items():
            print(f"Training {name}...")

            # Cross-validation
            cv_scores = cross_val_score(model, X, labels, cv=cv, scoring='f1_macro')
            self.cross_val_scores[name] = cv_scores

            # Full training
            model.fit(X, labels)

            # Training performance
            predictions = model.predict(X)
            accuracy = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average='macro')

            self.model_performance[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores)
            }

        self.is_trained = True
        self.categories = CATEGORIES

        # Save models automatically after training
        self.save_models()

        return self.model_performance

    def predict(self, text):
        """Predict category for new text with confidence scores"""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction. Please train or load models first.")

        processed_text = self.preprocessor.preprocess_single(text)
        X = self.vectorizer.transform([processed_text])

        predictions = {}
        probabilities = {}
        confidence_scores = {}

        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                pred = model.predict(X)[0]
            else:
                # For SVM without probability, use decision function
                pred = model.predict(X)[0]
                decision_values = model.decision_function(X)[0]
                # Convert decision values to probabilities using softmax
                proba = np.exp(decision_values) / np.sum(np.exp(decision_values))
            
            predictions[name] = pred
            probabilities[name] = dict(zip(self.categories, proba))
            confidence_scores[name] = np.max(proba)

        # Ensemble prediction (majority vote)
        ensemble_pred = max(set(predictions.values()), key=list(predictions.values()).count)

        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence_scores': confidence_scores,
            'ensemble_prediction': ensemble_pred,
            'processed_text': processed_text
        }

    def save_models(self):
        """Save all trained models and vectorizer"""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.vectorizer, MODELS_DIR / 'vectorizer.pkl')
        joblib.dump(self.preprocessor, MODELS_DIR / 'preprocessor.pkl')
        joblib.dump(self.categories, MODELS_DIR / 'categories.pkl')
        joblib.dump(self.model_performance, MODELS_DIR / 'performance.pkl')

        for name, model in self.models.items():
            filename = name.lower().replace(' ', '_') + '.pkl'
            joblib.dump(model, MODELS_DIR / filename)

        print("✅ Models saved successfully!")

    def load_models(self, silent=False):
        """Load pre-trained models"""
        try:
            self.vectorizer = joblib.load(MODELS_DIR / 'vectorizer.pkl')
            self.preprocessor = joblib.load(MODELS_DIR / 'preprocessor.pkl')
            self.categories = joblib.load(MODELS_DIR / 'categories.pkl')
            
            # Load performance metrics if exists
            performance_file = MODELS_DIR / 'performance.pkl'
            if performance_file.exists():
                self.model_performance = joblib.load(performance_file)

            for name in self.models.keys():
                filename = name.lower().replace(' ', '_') + '.pkl'
                self.models[name] = joblib.load(MODELS_DIR / filename)

            self.is_trained = True
            if not silent:
                print("✅ Models loaded successfully!")
            return True
            
        except FileNotFoundError as e:
            if not silent:
                print(f"❌ Model files not found: {e}")
            return False
        except Exception as e:
            if not silent:
                print(f"❌ Error loading models: {e}")
            return False

    def get_feature_importance(self, top_n=15):
        """Get most important features for each category"""
        if not self.is_trained or not hasattr(self.models['Naive Bayes'], 'feature_log_prob_'):
            return None

        feature_names = self.vectorizer.get_feature_names_out()
        importance_data = []

        for i, category in enumerate(self.categories):
            # For Naive Bayes, use log probabilities
            log_probs = self.models['Naive Bayes'].feature_log_prob_[i]
            top_indices = log_probs.argsort()[-top_n:][::-1]

            for idx in top_indices:
                importance_data.append({
                    'Category': category,
                    'Feature': feature_names[idx],
                    'Importance': log_probs[idx]
                })

        return pd.DataFrame(importance_data)

    def evaluate_model(self, test_texts, test_labels):
        """Evaluate model on test data"""
        if not self.is_trained:
            raise ValueError("Models must be trained before evaluation")

        processed_texts = self.preprocessor.preprocess_batch(test_texts)
        X_test = self.vectorizer.transform(processed_texts)

        evaluation_results = {}

        for name, model in self.models.items():
            predictions = model.predict(X_test)
            accuracy = accuracy_score(test_labels, predictions)
            f1 = f1_score(test_labels, predictions, average='macro')

            evaluation_results[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'classification_report': classification_report(test_labels, predictions, output_dict=True)
            }

        return evaluation_results