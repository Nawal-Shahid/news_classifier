from utils.unsupervised_classifier import UnsupervisedClassifier

# Initialize and load model
classifier = UnsupervisedClassifier()
classifier.load_models()

# Test articles
test_articles = [
    "Stock markets soared today as tech companies reported record profits and strong earnings growth.",
    "The football team won the championship after a thrilling final match that went into overtime.",
    "New smartphone features advanced camera technology and artificial intelligence capabilities.",
    "Parliament debated new healthcare legislation amid concerns from opposition parties.",
    "The blockbuster movie broke box office records in its opening weekend worldwide."
]

print("\n" + "="*60)
print("🧪 Testing Unsupervised News Classifier")
print("="*60)

for i, article in enumerate(test_articles, 1):
    print(f"\n📰 Article {i}:")
    print(f"Text: {article[:80]}...")
    
    result = classifier.predict(article)
    
    print(f"✅ Predicted Category: {result['prediction'].upper()}")
    print(f"📊 Confidence: {result['confidence']:.1%}")
    print(f"🎯 Cluster ID: {result['cluster_id']}")
    
    # Show top 3 probable categories
    sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
    print("Top 3 probabilities:")
    for cat, prob in sorted_probs:
        print(f"  • {cat}: {prob:.1%}")

print("\n" + "="*60)
print("✅ Testing complete!")
