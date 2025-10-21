import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import time
import json
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from config import MAX_TEXT_LENGTH, MIN_TEXT_LENGTH
from utils.classifier import NewsClassifier
from utils.data_loader import DataLoader
from utils.preprocessor import TextPreprocessor

# Page configuration
st.set_page_config(
    page_title="News Article Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load custom CSS
def load_css():
    try:
        with open('static/styles.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except:
        pass


class NewsClassifierApp:
    def __init__(self):
        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.data_loader = DataLoader()
        
        # Initialize classifier but don't load models yet
        self.classifier = NewsClassifier()
        self.models_loaded = False

    def setup_page(self):
        """Setup the main page layout and styling"""
        load_css()

        # Main header
        st.markdown('<h1 class="main-header">📰 Advanced News Article Classification System</h1>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <p style='font-size: 1.2rem; color: #666;'>
                Classify news articles into Business, Entertainment, Politics, Sports, and Technology categories
                using advanced machine learning models.
            </p>
        </div>
        """, unsafe_allow_html=True)

    def setup_sidebar(self):
        """Setup sidebar with controls"""
        with st.sidebar:
            st.header("🔧 Control Panel")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Train Models", use_container_width=True, type="primary"):
                    self.train_models()

            with col2:
                if st.button("Load Models", use_container_width=True):
                    self.load_existing_models()

            st.markdown("---")

            # Model status
            st.subheader("Model Status")
            if self.models_loaded:
                st.success("✅ Models Loaded")

                # Display model performance
                if hasattr(self.classifier, 'model_performance') and self.classifier.model_performance:
                    for model, metrics in self.classifier.model_performance.items():
                        with st.expander(f"{model} Performance"):
                            st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
                            st.metric("F1-Score", f"{metrics['f1_score']:.1%}")
                            st.metric("Cross-val", f"{metrics['cv_mean']:.1%} ± {metrics['cv_std']:.1%}")
            else:
                st.warning("⚠️ Models Not Loaded")
                st.info("Please train or load models to start classification")

            st.markdown("---")
            st.subheader("ℹ️ About")
            st.markdown("""
            This system uses multiple machine learning algorithms:
            - **Naive Bayes**: Fast and interpretable
            - **SVM**: High accuracy with good generalization
            - **Random Forest**: Robust ensemble method
            """)

    def train_models(self):
        """Train classification models"""
        with st.spinner('🔄 Training models with comprehensive dataset...'):
            try:
                texts, labels = self.data_loader.get_training_data()
                performance = self.classifier.train(texts, labels)
                self.models_loaded = True
                st.session_state.models_loaded = True
                st.session_state.performance = performance
                st.success("✅ Models trained and saved successfully!")

                # Show training summary
                with st.expander("📊 Training Summary", expanded=True):
                    self.show_training_summary(performance)

            except Exception as e:
                st.error(f"❌ Error training models: {str(e)}")
                st.info("💡 Make sure your dataset file exists and is properly formatted")

    def load_existing_models(self):
        """Load pre-trained models"""
        with st.spinner('🔄 Loading pre-trained models...'):
            try:
                success = self.classifier.load_models()
                if success:
                    self.models_loaded = True
                    st.session_state.models_loaded = True
                    st.session_state.performance = getattr(self.classifier, 'model_performance', {})
                    st.success("✅ Models loaded successfully!")
                else:
                    st.error("❌ No trained models found. Please train models first.")
                    st.info("💡 Click 'Train Models' to create new models")
            except Exception as e:
                st.error(f"❌ Error loading models: {str(e)}")

    def show_training_summary(self, performance):
        """Display training performance summary"""
        perf_df = pd.DataFrame(performance).T
        perf_df = perf_df[['accuracy', 'f1_score', 'cv_mean']]
        perf_df.columns = ['Accuracy', 'F1-Score', 'Cross-Val Score']

        col1, col2 = st.columns([1, 1])

        with col1:
            st.dataframe(perf_df.style.format("{:.3f}"))

        with col2:
            fig = px.bar(
                perf_df.reset_index().melt(id_vars=['index']),
                x='index',
                y='value',
                color='variable',
                barmode='group',
                title='Model Performance Comparison',
                labels={'index': 'Model', 'value': 'Score', 'variable': 'Metric'}
            )
            st.plotly_chart(fig, use_container_width=True)

    def extract_article_content(self, url):
        """Extract article content from URL with error handling"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Try to find main article content
            article = soup.find('article')
            if article:
                text = article.get_text()
            else:
                # Fallback: get all paragraph text
                paragraphs = soup.find_all('p')
                text = ' '.join([p.get_text() for p in paragraphs])

            # Clean and limit text
            text = ' '.join(text.split())[:MAX_TEXT_LENGTH]

            if len(text) < MIN_TEXT_LENGTH:
                raise ValueError("Extracted text is too short")

            return text

        except requests.exceptions.HTTPError as e:
            st.error(f"❌ HTTP Error: Could not fetch content from URL")
            return None
        except Exception as e:
            st.warning(f"⚠️ Could not fetch article content from URL. Please paste the text directly.")
            return None

    def analyze_sentiment(self, text):
        """Analyze text sentiment"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if polarity > 0.1:
            sentiment = "Positive"
            color = "green"
        elif polarity < -0.1:
            sentiment = "Negative"
            color = "red"
        else:
            sentiment = "Neutral"
            color = "blue"

        return sentiment, polarity, subjectivity, color

    def create_wordcloud(self, text):
        """Generate word cloud from text"""
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Word Cloud - Most Frequent Terms', fontsize=16, pad=20)
        return fig

    def validate_input(self, text):
        """Validate input text"""
        if not text or len(text.strip()) < MIN_TEXT_LENGTH:
            st.error(f"❌ Please enter at least {MIN_TEXT_LENGTH} characters of text")
            return False

        if not self.preprocessor.is_english(text):
            st.warning("⚠️ Non-English text detected. Results may be less accurate.")

        return True

    def run_analysis(self, article_text):
        """Run complete analysis on article text"""
        with st.spinner("🔍 Analyzing article content..."):
            try:
                results = self.classifier.predict(article_text)
                sentiment, polarity, subjectivity, color = self.analyze_sentiment(article_text)
                return results, sentiment, polarity, subjectivity, color
            except ValueError as e:
                if "Models must be trained" in str(e):
                    st.error("❌ Models are not trained or loaded. Please train or load models first.")
                    return None, None, None, None, None
                else:
                    raise e

    def display_results(self, results, sentiment, polarity, subjectivity, color, article_text):
        """Display analysis results"""
        if results is None:
            return
            
        st.success("✅ Analysis Complete!")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Classification", "Probabilities", "Sentiment",
            "Word Cloud", "Insights"
        ])

        with tab1:
            self.display_classification_results(results)
        with tab2:
            self.display_probability_analysis(results)
        with tab3:
            self.display_sentiment_analysis(sentiment, polarity, subjectivity, color)
        with tab4:
            self.display_wordcloud(article_text)
        with tab5:
            self.display_insights(results, article_text)

    def display_classification_results(self, results):
        """Display classification results"""
        st.subheader("Model Predictions")
        st.markdown(f"### Ensemble Prediction: **{results['ensemble_prediction'].title()}**")

        cols = st.columns(len(results['predictions']))
        for idx, (model, prediction) in enumerate(results['predictions'].items()):
            with cols[idx]:
                confidence = results['confidence_scores'][model]
                st.metric(f"{model}", f"{prediction.title()}", delta=f"{confidence:.1%} confidence")

    def display_probability_analysis(self, results):
        """Display probability analysis"""
        st.subheader("Classification Probabilities")

        for model, probabilities in results['probabilities'].items():
            with st.expander(f"{model} Probability Distribution"):
                prob_df = pd.DataFrame({
                    'Category': list(probabilities.keys()),
                    'Probability': list(probabilities.values())
                }).sort_values('Probability', ascending=False)

                for _, row in prob_df.iterrows():
                    col1, col2, col3 = st.columns([2, 5, 1])
                    with col1:
                        st.write(f"**{row['Category'].title()}**")
                    with col2:
                        st.progress(float(row['Probability']))
                    with col3:
                        st.write(f"{row['Probability']:.1%}")

    def display_sentiment_analysis(self, sentiment, polarity, subjectivity, color):
        """Display sentiment analysis"""
        st.subheader("Sentiment Analysis")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Sentiment", sentiment)
        with col2:
            st.metric("Polarity Score", f"{polarity:.3f}")
        with col3:
            st.metric("Subjectivity", f"{subjectivity:.3f}")

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=polarity,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Sentiment Polarity"},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': color},
                'steps': [
                    {'range': [-1, -0.1], 'color': "lightcoral"},
                    {'range': [-0.1, 0.1], 'color': "lightyellow"},
                    {'range': [0.1, 1], 'color': "lightgreen"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    def display_wordcloud(self, article_text):
        """Display word cloud"""
        st.subheader("Word Cloud Analysis")
        fig = self.create_wordcloud(article_text)
        st.pyplot(fig)

    def display_insights(self, results, article_text):
        """Display additional insights"""
        st.subheader("Advanced Insights")

        try:
            importance_df = self.classifier.get_feature_importance(top_n=10)
            if importance_df is not None:
                st.subheader("🔍 Key Features by Category")
                fig = px.bar(importance_df, x='Importance', y='Feature', color='Category',
                            orientation='h', title='Most Important Features for Classification')
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info("Feature importance analysis requires model training with specific attributes.")

        st.subheader("📏 Text Statistics")
        col1, col2, col3, col4 = st.columns(4)

        word_count = len(article_text.split())
        char_count = len(article_text)
        sentence_count = len(article_text.split('.'))
        avg_word_length = char_count / max(word_count, 1)

        with col1:
            st.metric("Word Count", word_count)
        with col2:
            st.metric("Character Count", char_count)
        with col3:
            st.metric("Sentences", sentence_count)
        with col4:
            st.metric("Avg Word Length", f"{avg_word_length:.1f}")

    def main_interface(self):
        """Main application interface"""
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Input Article")

            input_method = st.radio("Choose input method:", ["Enter Text", "Enter URL"], horizontal=True)
            article_text = ""

            if input_method == "Enter Text":
                article_text = st.text_area(
                    "Paste article text here:",
                    height=250,
                    placeholder="Paste the complete news article text here...\n\nMinimum 50 characters required for accurate classification.",
                    help="Enter at least 50 characters of news article text for classification"
                )
            else:
                url = st.text_input("Enter article URL:", placeholder="https://example.com/news-article",
                                   help="Enter the full URL of the news article")
                if url:
                    with st.spinner("🌐 Extracting article content..."):
                        article_text = self.extract_article_content(url)
                        if article_text:
                            st.text_area("Extracted Content", article_text, height=250,
                                       help="Preview of extracted article content")

            analyze_clicked = st.button("🔍 Analyze Article", type="primary",
                                       use_container_width=True, disabled=not self.models_loaded)

            if not self.models_loaded:
                st.warning("⚠️ Please train or load models first using the sidebar controls.")

        with col2:
            st.subheader("Analysis Dashboard")

            if analyze_clicked and article_text:
                if self.validate_input(article_text):
                    results, sentiment, polarity, subjectivity, color = self.run_analysis(article_text)
                    if results is not None:  # Only display if analysis was successful
                        self.display_results(results, sentiment, polarity, subjectivity, color, article_text)

    def run(self):
        """Main application runner"""
        self.setup_page()
        self.setup_sidebar()
        
        # Initialize session state
        if 'models_loaded' not in st.session_state:
            st.session_state.models_loaded = False
        
        # Check if models are already loaded in the classifier
        if self.classifier.is_trained:
            self.models_loaded = True
            st.session_state.models_loaded = True
        else:
            self.models_loaded = st.session_state.models_loaded
        
        # Show warning if models aren't loaded
        if not self.models_loaded:
            st.warning("""
            **🔧 Models Not Loaded**
            
            To use the classification features:
            1. **Train Models** - Click "Train Models" in the sidebar to train new models with your dataset
            2. **Load Models** - Click "Load Models" if you have pre-trained models
            
            Make sure your `news_dataset.csv` file exists in the data directory!
            """)
        
        self.main_interface()


# Run the application
if __name__ == "__main__":
    app = NewsClassifierApp()
    app.run()