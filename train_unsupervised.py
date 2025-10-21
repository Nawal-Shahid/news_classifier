import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import joblib
from pathlib import Path
import time

from utils.preprocessor import TextPreprocessor
from config import DATA_PATH, MODELS_DIR

def load_unlabeled_data():
    """Load data without category labels"""
    print("📂 Loading unlabeled dataset...")
    
    try:
        df = pd.read_csv(DATA_PATH, encoding='utf-8')
        print(f"✅ Loaded {len(df)} articles")
        print(f"📋 Columns: {df.columns.tolist()}")
        
        # Use title or description as text
        if 'description' in df.columns:
            df['text'] = df['description'].fillna('')
        elif 'title' in df.columns:
            df['text'] = df['title'].fillna('')
        else:
            raise ValueError("No text column found")
        
        # Combine title and description if both exist
        if 'title' in df.columns and 'description' in df.columns:
            df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
        
        # Clean
        df = df[df['text'].str.len() > 50]
        print(f"✅ Using {len(df)} articles after filtering")
        
        return df['text'].tolist()
    
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def train_clustering_model(texts, n_clusters=5):
    """Train K-Means clustering model"""
    print(f"\n🔄 Training K-Means clustering with {n_clusters} clusters...")
    
    # Preprocess
    print("Preprocessing texts...")
    preprocessor = TextPreprocessor()
    processed_texts = preprocessor.preprocess_batch(texts[:5000])  # Limit for speed
    
    # Vectorize
    print("Vectorizing texts...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.8)
    X = vectorizer.fit_transform(processed_texts)
    
    # Cluster
    print(f"Clustering into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    clusters = kmeans.fit_predict(X)
    
    # Get top terms per cluster
    print("\n📊 Cluster Analysis:")
    print("=" * 60)
    
    cluster_labels = {}
    feature_names = vectorizer.get_feature_names_out()
    
    for i in range(n_clusters):
        # Get top terms for this cluster
        center = kmeans.cluster_centers_[i]
        top_indices = center.argsort()[-10:][::-1]
        top_terms = [feature_names[idx] for idx in top_indices]
        
        # Count articles in cluster
        count = np.sum(clusters == i)
        
        print(f"\nCluster {i} ({count} articles):")
        print(f"  Top terms: {', '.join(top_terms[:5])}")
        
        # Suggest label based on top terms
        suggested_label = suggest_cluster_label(top_terms)
        cluster_labels[i] = suggested_label
        print(f"  Suggested category: {suggested_label}")
    
    # Save models
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, MODELS_DIR / 'unsupervised_vectorizer.pkl')
    joblib.dump(kmeans, MODELS_DIR / 'kmeans_model.pkl')
    joblib.dump(cluster_labels, MODELS_DIR / 'cluster_labels.pkl')
    joblib.dump(preprocessor, MODELS_DIR / 'preprocessor.pkl')
    
    print(f"\n✅ Models saved to {MODELS_DIR}")
    
    return kmeans, vectorizer, cluster_labels, clusters


def suggest_cluster_label(top_terms):
    """Suggest a category label based on top terms"""
    terms_str = ' '.join(top_terms).lower()
    
    # Define keyword patterns for categories
    patterns = {
        'business': ['market', 'stock', 'economy', 'company', 'business', 'financial', 'trade', 'bank', 'profit', 'revenue'],
        'technology': ['tech', 'software', 'computer', 'digital', 'internet', 'app', 'data', 'cyber', 'ai', 'innovation'],
        'sports': ['game', 'team', 'player', 'match', 'win', 'sport', 'football', 'cricket', 'championship', 'coach'],
        'politics': ['government', 'minister', 'parliament', 'election', 'political', 'policy', 'vote', 'president', 'law', 'court'],
        'entertainment': ['film', 'movie', 'music', 'actor', 'show', 'star', 'album', 'concert', 'award', 'celebrity']
    }
    
    # Score each category
    scores = {}
    for category, keywords in patterns.items():
        score = sum(1 for keyword in keywords if keyword in terms_str)
        scores[category] = score
    
    # Return category with highest score
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    else:
        return 'general'


def analyze_sample_articles(texts, clusters, cluster_labels, n_samples=3):
    """Show sample articles from each cluster"""
    print("\n📰 Sample Articles per Cluster:")
    print("=" * 60)
    
    for cluster_id in range(len(set(clusters))):
        cluster_indices = np.where(clusters == cluster_id)[0]
        sample_indices = np.random.choice(cluster_indices, min(n_samples, len(cluster_indices)), replace=False)
        
        print(f"\n{cluster_labels[cluster_id].upper()} (Cluster {cluster_id}):")
        for idx in sample_indices:
            text_preview = texts[idx][:100] + "..." if len(texts[idx]) > 100 else texts[idx]
            print(f"  • {text_preview}")


def main():
    print("🚀 Starting Unsupervised News Classification Training...")
    print("=" * 60)
    
    # Load data
    texts = load_unlabeled_data()
    
    if texts is None or len(texts) == 0:
        print("❌ No data available for training")
        return
    
    # Limit to first 5000 for reasonable training time
    if len(texts) > 5000:
        print(f"⚠️ Using first 5000 articles (out of {len(texts)}) for training speed")
        texts = texts[:5000]
    
    start_time = time.time()
    
    # Train clustering
    kmeans, vectorizer, cluster_labels, clusters = train_clustering_model(texts, n_clusters=5)
    
    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time:.2f} seconds")
    
    # Show sample articles
    analyze_sample_articles(texts, clusters, cluster_labels)
    
    print("\n" + "=" * 60)
    print("🎉 Unsupervised training complete!")
    print("💡 You can now use the app to classify new articles")
    print("⚠️ Note: Cluster labels are automatically suggested and may not be perfect")


if __name__ == "__main__":
    main()
