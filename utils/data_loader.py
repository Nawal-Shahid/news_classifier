import pandas as pd
import numpy as np
from pathlib import Path
import requests
import zipfile
import os
from config import DATA_PATH, CATEGORIES


class DataLoader:
    def __init__(self):
        self.data_path = DATA_PATH

    def load_bbc_data(self):
        """Load BBC news dataset from Kaggle"""
        try:
            if self.data_path.exists():
                # Try different possible formats and column names
                df = self._load_and_validate_dataset()
                print(f"✅ Loaded dataset with {len(df)} articles")
                print(f"📊 Category distribution:\n{df['category'].value_counts()}")
                return df
            else:
                print("❌ Dataset file not found. Please ensure bbc_news.csv is in the data/ folder")
                print("💡 Using sample data for demonstration...")
                return self._create_sample_data()
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return self._create_sample_data()

    def _load_and_validate_dataset(self):
        """Load dataset with proper validation and column mapping"""
        # Try reading with different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1']

        for encoding in encodings:
            try:
                df = pd.read_csv(self.data_path, encoding=encoding)
                print(f"✅ Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Could not read file with any encoding")

        # Display dataset info
        print(f"📁 Dataset shape: {df.shape}")
        print(f"📋 Columns: {df.columns.tolist()}")
        print(f"🔍 First few rows:\n{df.head(2)}")

        # Map column names to standard format
        df = self._standardize_columns(df)

        # Validate required columns
        if 'text' not in df.columns or 'category' not in df.columns:
            raise ValueError("Dataset must contain 'text' and 'category' columns")

        # Clean data
        df = self._clean_data(df)

        return df

    def _standardize_columns(self, df):
        """Map common column names to standard format"""
        column_mapping = {
            # Common column names in BBC datasets
            'Article': 'text',
            'Text': 'text',
            'Content': 'text',
            'Description': 'text',
            'description': 'text',
            'News': 'text',
            'title': 'text',
            'Title': 'text',
            'Category': 'category',
            'Type': 'category',
            'Label': 'category',
            'Class': 'category'
        }

        # Rename columns
        df = df.rename(columns=column_mapping)

        # If we still don't have required columns, try to infer
        if 'text' not in df.columns:
            # Try to find text-like columns
            text_candidates = ['description', 'content', 'article', 'news', 'title']
            for col in df.columns:
                if col.lower() in text_candidates:
                    df = df.rename(columns={col: 'text'})
                    break
        
        if 'category' not in df.columns:
            # If no category column, this might not be a labeled dataset
            print("⚠️ No category column found. Available columns:", df.columns.tolist())
            print("💡 Please ensure your dataset has a 'category' or 'Category' column.")
            raise ValueError("Dataset must contain 'category' column for training")

        # Keep only standard columns
        available_cols = [col for col in ['text', 'category'] if col in df.columns]
        df = df[available_cols]

        return df

    def _clean_data(self, df):
        """Clean and preprocess the dataset"""
        # Remove rows with missing values
        initial_count = len(df)
        df = df.dropna(subset=['text', 'category'])

        # Clean text - basic preprocessing
        df['text'] = df['text'].astype(str).str.strip()

        # Remove empty texts
        df = df[df['text'].str.len() > 50]

        # Standardize category names
        if 'category' in df.columns:
            df['category'] = df['category'].str.lower().str.strip()

            # Map common variations
            category_mapping = {
                'sports': 'sport',
                'technology': 'tech',
                'entertainment': 'entertainment',
                'business': 'business',
                'politics': 'politics'
            }
            df['category'] = df['category'].map(category_mapping).fillna(df['category'])

        cleaned_count = len(df)
        if initial_count != cleaned_count:
            print(f"🧹 Cleaned data: removed {initial_count - cleaned_count} rows")

        return df

    def download_bbc_dataset(self):
        """Download BBC dataset if not available locally"""
        if self.data_path.exists():
            print("✅ Dataset already exists")
            return True

        print("📥 Downloading BBC dataset...")
        try:
            # You would need to set up Kaggle API for this
            # For now, we'll use sample data
            print("💡 Please download the dataset manually from Kaggle")
            print("🔗 https://www.kaggle.com/datasets/gpreda/bbc-news")
            return False
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False

    def _create_sample_data(self):
        """Create comprehensive sample data for demonstration"""
        print("🔄 Creating sample dataset for demonstration...")

        # Create more samples per category for proper train/test split
        business_articles = [
            "Stock markets reached record highs today as major tech companies reported strong quarterly earnings exceeding analyst expectations. The NASDAQ composite rose by 2.5 percent led by gains in technology stocks. Financial analysts predict continued growth in the technology sector despite global economic uncertainties.",
            "Global economic growth is projected to slow next year according to the International Monetary Fund latest report. Inflation concerns and geopolitical tensions are cited as major factors affecting the outlook. Central banks worldwide are considering monetary policy adjustments to address these challenges.",
            "Corporate mergers and acquisitions reached unprecedented levels this quarter with several billion dollar deals announced across various sectors including healthcare and energy industries. Investment bankers report increased activity in private equity and venture capital funding rounds.",
            "Major retail chain announced expansion plans opening fifty new stores nationwide creating thousands of jobs. Company executives cited strong consumer demand and favorable market conditions. Shareholders approved the growth strategy during annual meeting.",
            "Cryptocurrency markets experienced significant volatility with Bitcoin prices fluctuating dramatically. Investors remain divided on digital currency future prospects. Regulatory authorities worldwide continue monitoring developments closely.",
            "Manufacturing sector showed signs of recovery with production output increasing for third consecutive month. Industry leaders attribute improvement to supply chain stabilization and increased consumer spending. Economic indicators suggest sustained growth momentum."
        ]

        entertainment_articles = [
            "The new superhero movie shattered box office records in its opening weekend grossing over 200 million dollars worldwide. Critics praised the film visual effects and compelling storyline. Fans celebrated the successful launch of the latest installment in the popular franchise.",
            "Award winning actor announced their retirement from the film industry after five decades of memorable performances. The star received numerous accolades throughout their career including multiple Oscar nominations and international film awards.",
            "Music streaming platforms reported significant increases in user engagement as new album releases from popular artists dominated the charts this month. Industry experts note a shift in music consumption patterns with digital platforms becoming the primary distribution channel.",
            "Television series finale attracted record viewership numbers becoming most watched episode in network history. Social media platforms buzzed with fan reactions and theories. Producers announced plans for potential spin-off series.",
            "Film festival showcased independent movies from emerging directors receiving critical acclaim. International jury awarded top prizes for innovative storytelling and cinematography. Industry professionals praised festival for promoting diverse voices.",
            "Concert tour announcement generated massive ticket sales with venues selling out within minutes. Fans expressed excitement across social media platforms. Tour organizers added additional dates to meet overwhelming demand."
        ]

        politics_articles = [
            "Parliament engaged in heated debates over new environmental legislation aimed at reducing carbon emissions. The proposed policy has drawn mixed reactions from various political parties. Environmental activists welcomed the initiative while industry representatives expressed concerns about economic impacts.",
            "Government officials announced comprehensive healthcare reforms including increased funding for public hospitals and expanded insurance coverage for low income families. The reforms aim to address systemic challenges in the national healthcare system and improve access to medical services.",
            "International diplomatic talks concluded with agreements on trade and security cooperation between participating nations. The summit addressed global economic challenges and established frameworks for future collaboration on climate change and technology development.",
            "Election campaign intensified as candidates presented policy platforms addressing education infrastructure and economic development. Polls indicated tight race with voters focusing on key domestic issues. Debate schedules announced for upcoming weeks.",
            "Legislative committee approved budget proposal allocating funds for infrastructure projects and social programs. Opposition parties raised concerns about spending priorities and fiscal responsibility. Final vote scheduled for next session.",
            "Foreign minister concluded diplomatic visit strengthening bilateral relations and signing cooperation agreements. Discussions covered trade investment and cultural exchange programs. Both nations committed to enhanced partnership."
        ]

        sport_articles = [
            "National football team secured a dramatic victory in the championship finals with a last minute goal. Thousands of fans celebrated the historic win across the country. The team captain dedicated the victory to their loyal supporters and coaching staff.",
            "Olympic athletes began intensive training programs in preparation for the upcoming international games. New training facilities have been established to support the team with advanced equipment and sports science resources. Coaches are optimistic about medal prospects.",
            "Professional basketball league announced rule changes aimed at improving game safety and enhancing viewer experience for the upcoming season. The modifications include updated foul regulations and revised game timing procedures to increase excitement.",
            "Tennis champion won grand slam tournament defeating top ranked opponent in thrilling five set match. Victory marked career milestone and solidified position among sport greatest players. Fans praised exceptional performance and sportsmanship.",
            "Marathon event attracted thousands of participants from around the world setting new attendance record. Runners competed in various categories with winners receiving prizes and recognition. Organizers praised community support and volunteer efforts.",
            "Baseball team clinched playoff berth after winning crucial game against division rivals. Players celebrated achievement while preparing for postseason competition. Manager credited team chemistry and hard work for success."
        ]

        tech_articles = [
            "Tech giant unveiled groundbreaking artificial intelligence system capable of complex problem solving. The new technology promises to revolutionize various industries including healthcare, finance, and transportation. Researchers demonstrated the system capabilities in live demonstrations.",
            "Cybersecurity firm reported increased threats to digital infrastructure prompting calls for enhanced protection measures across corporate networks worldwide. Security experts recommend immediate software updates and multi-factor authentication implementation.",
            "Research team developed innovative quantum computing chip that significantly improves processing speed while reducing energy consumption requirements. The breakthrough could accelerate scientific discoveries and transform computational capabilities across multiple fields.",
            "Smartphone manufacturer launched latest device featuring advanced camera technology and extended battery life. Early reviews highlighted improved performance and user experience. Pre-orders exceeded expectations indicating strong market demand.",
            "Software company released major platform update introducing new features and security enhancements. Developers praised improved tools and documentation. Users reported positive experiences with streamlined interface and functionality.",
            "Space exploration program achieved successful satellite deployment expanding global communications network. Engineers confirmed all systems operating normally. Mission represents significant advancement in aerospace technology development."
        ]

        sample_data = {
            'text': business_articles + entertainment_articles + politics_articles + sport_articles + tech_articles,
            'category': ['business'] * 6 + ['entertainment'] * 6 + ['politics'] * 6 + ['sport'] * 6 + ['tech'] * 6
        }

        df = pd.DataFrame(sample_data)
        print(f"✅ Created sample dataset with {len(df)} articles")
        return df

    def get_training_data(self):
        """Get processed training data with validation"""
        df = self.load_bbc_data()

        # Validate we have data
        if df is None or len(df) == 0:
            raise ValueError("No training data available")

        texts = df['text'].tolist()
        labels = df['category'].tolist()

        print(f"🎯 Training data: {len(texts)} articles, {len(set(labels))} categories")
        print(f"📈 Category distribution: {pd.Series(labels).value_counts().to_dict()}")

        return texts, labels

    def get_dataset_stats(self):
        """Get detailed dataset statistics"""
        df = self.load_bbc_data()

        stats = {
            'total_articles': len(df),
            'categories': df['category'].nunique(),
            'category_distribution': df['category'].value_counts().to_dict(),
            'avg_text_length': df['text'].str.len().mean(),
            'min_text_length': df['text'].str.len().min(),
            'max_text_length': df['text'].str.len().max()
        }

        return stats
