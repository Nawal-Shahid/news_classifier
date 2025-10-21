import joblib
import numpy as np
from pathlib import Path
from config import MODELS_DIR


class UnsupervisedClassifier:
    """Classifier using K-Means clustering for unlabeled data"""
    
    def __init__(self):
        self.kmeans = None
        self.vectorizer = None
        self.cluster_labels = None
        self.preprocessor = None
        self.is_trained = False
    
    def load_models(self):
        """Load pre-trained unsupervised models"""
        try:
            self.vectorizer = joblib.load(MODELS_DIR / 'unsupervised_vectorizer.pkl')
            self.kmeans = joblib.load(MODELS_DIR / 'kmeans_model.pkl')
            self.cluster_labels = joblib.load(MODELS_DIR / 'cluster_labels.pkl')
            self.preprocessor = joblib.load(MODELS_DIR / 'preprocessor.pkl')
            self.is_trained = True
            print("✅ Unsupervised models loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def predict(self, text):
        """Predict category for new text"""
        if not self.is_trained:
            raise ValueError("Models must be loaded before prediction")
        
        # Preprocess
        processed_text = self.preprocessor.preprocess_single(text)
        
        # Vectorize
        X = self.vectorizer.transform([processed_text])
        
        # Predict cluster
        cluster_id = self.kmeans.predict(X)[0]
        
        # Get distances to all cluster centers (for confidence)
        distances = self.kmeans.transform(X)[0]
        
        # Convert distances to probabilities (inverse distance)
        # Smaller distance = higher probability
        inv_distances = 1 / (distances + 1e-10)
        probabilities = inv_distances / inv_distances.sum()
        
        # Get predicted category
        predicted_category = self.cluster_labels[cluster_id]
        
        # Create probability dict for all categories
        probability_dict = {}
        for cluster_id, label in self.cluster_labels.items():
            probability_dict[label] = probabilities[cluster_id]
        
        # Get confidence (probability of assigned cluster)
        confidence = probabilities[cluster_id]
        
        return {
            'prediction': predicted_category,
            'cluster_id': int(cluster_id),
            'confidence': float(confidence),
            'probabilities': probability_dict,
            'processed_text': processed_text
        }
    
    def get_cluster_info(self):
        """Get information about clusters"""
        if not self.is_trained:
            return None
        
        return {
            'n_clusters': len(self.cluster_labels),
            'labels': self.cluster_labels
        }
