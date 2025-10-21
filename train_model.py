import pandas as pd
from sklearn.model_selection import train_test_split
from utils.data_loader import DataLoader
from utils.classifier import NewsClassifier
from config import MODELS_DIR
import time


def main():
    print("🚀 Starting News Classification Model Training...")

    # Load data
    data_loader = DataLoader()
    texts, labels = data_loader.get_training_data()

    print(f"📊 Loaded {len(texts)} articles for training")
    print(f"📈 Category distribution: {pd.Series(labels).value_counts().to_dict()}")

    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"🎯 Training set: {len(X_train)} articles")
    print(f"🧪 Test set: {len(X_test)} articles")

    # Initialize and train classifier
    classifier = NewsClassifier()

    start_time = time.time()
    performance = classifier.train(X_train, y_train)
    training_time = time.time() - start_time

    print(f"✅ Training completed in {training_time:.2f} seconds")

    # Display performance metrics
    print("\n📊 Model Performance Summary:")
    print("=" * 50)
    for model_name, metrics in performance.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  F1-Score: {metrics['f1_score']:.3f}")
        print(f"  Cross-val: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")

    # Evaluate on test set
    print("\n🧪 Test Set Evaluation:")
    print("=" * 50)
    test_results = classifier.evaluate_model(X_test, y_test)

    for model_name, metrics in test_results.items():
        print(f"\n{model_name}:")
        print(f"  Test Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Test F1-Score: {metrics['f1_score']:.3f}")

    print(f"\n🎉 Models saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()