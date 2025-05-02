#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tweet Sentiment Analysis Streamlit App
--------------------------------------
This script loads a pre-trained DistilBERT model for sentiment analysis of tweets
and deploys it as an interactive web application using Streamlit.
French tweets are translated to English before analysis since the model was trained on English data.
"""

import os
import numpy as np
import tensorflow as tf
import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification, pipeline

# Configure page (must be the first Streamlit command)
st.set_page_config(
    page_title="Tweet Sentiment Analyzer",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern, intuitive look
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .stButton>button {
        background-color: #1e90ff;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0066cc;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid #d3d3d3;
        font-size: 16px;
    }
    .card {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .sentiment-positive { color: #4CC9F0; font-weight: bold; }
    .sentiment-neutral { color: #5E60CE; font-weight: bold; }
    .sentiment-negative { color: #FF4B4B; font-weight: bold; }
    .title { font-size: 2.5em; color: #2c3e50; text-align: center; margin-bottom: 10px; }
    .subtitle { font-size: 1.2em; color: #7f8c8d; text-align: center; margin-bottom: 30px; }
    .stSpinner { text-align: center; }
    </style>
""", unsafe_allow_html=True)

# Global variables
MODEL_PATH = os.environ.get('MODEL_PATH', '/Users/assietoudrame/Downloads/IAA/Analyseur_sentiment/tweet_sentiment_analysis/model_distilBERT')  # Absolute for local testing
# For deployment, use: MODEL_PATH = './tweet_sentiment_analysis/model_distilBERT'
MAX_LENGTH = 128
BATCH_SIZE = 16
LABELS = ['negative', 'neutral', 'positive']
COLORS = {"negative": "#FF4B4B", "neutral": "#5E60CE", "positive": "#4CC9F0"}

# Sidebar with app info
st.sidebar.image("https://img.icons8.com/color/48/000000/twitter--v1.png", width=50)
st.sidebar.title("Tweet Sentiment Analyzer")
st.sidebar.markdown("""
Analyze the sentiment of French tweets with ease!  
- 📝 Enter a single tweet or upload a CSV file.  
- 🌐 Tweets are translated to English for analysis.  
- 📊 View results with interactive charts.  
""")
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ by [Your Name]")

# Create a container for the model loading message
loading_container = st.empty()
loading_container.info("🔄 Loading model and translation pipeline, please wait...")

@st.cache_resource
def load_model():
    """Load the model and tokenizer (cached to prevent reloading)"""
    try:
        st.write(f"Loading model from: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model path does not exist: {MODEL_PATH}")
        
        if os.path.isdir(MODEL_PATH):
            st.write(f"Directory contents: {os.listdir(MODEL_PATH)}")
        
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={'TFDistilBertForSequenceClassification': TFDistilBertForSequenceClassification},
            compile=False
        )
        st.write("✅ Model loaded successfully")
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None

@st.cache_resource
def load_translator():
    """Load the French-to-English translation pipeline"""
    try:
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
        st.write("✅ Translation pipeline loaded successfully")
        return translator
    except Exception as e:
        st.error(f"❌ Failed to load translation pipeline: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

class TweetSentimentAnalyzer:
    def __init__(self, model, tokenizer, translator):
        self.model = model
        self.tokenizer = tokenizer
        self.translator = translator
    
    def translate(self, texts):
        """Translate French texts to English"""
        if isinstance(texts, str):
            texts = [texts]
        
        if self.translator is None:
            raise ValueError("Translation pipeline not initialized")
        
        with st.spinner("🌐 Translating tweets to English..."):
            translated = self.translator(texts, max_length=MAX_LENGTH)
            return [t['translation_text'] for t in translated]
    
    def preprocess(self, texts):
        """Tokenize and prepare texts for the model"""
        if isinstance(texts, str):
            texts = [texts]
            
        encoded = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='tf'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def predict(self, texts):
        """Predict sentiment for a list of French texts after translating to English"""
        translated_texts = self.translate(texts)
        inputs = self.preprocess(translated_texts)
        
        with st.spinner('🔍 Analyzing tweets...'):
            predictions = self.model.predict(inputs, batch_size=BATCH_SIZE)
            
            if isinstance(predictions, tuple) and len(predictions) > 0:
                logits = predictions[0]
            elif hasattr(predictions, 'logits'):
                logits = predictions.logits
            else:
                logits = predictions
            
            results = []
            for i, pred in enumerate(logits):
                sentiment_idx = np.argmax(pred)
                sentiment_label = LABELS[sentiment_idx]
                sentiment_score = float(tf.nn.softmax(pred)[sentiment_idx])
                
                result = {
                    'original_text': texts[i] if isinstance(texts, list) else texts,
                    'translated_text': translated_texts[i],
                    'sentiment': sentiment_label,
                    'confidence': sentiment_score,
                    'scores': {label: float(score) for label, score in zip(LABELS, tf.nn.softmax(pred).numpy())}
                }
                results.append(result)
            
            return results if len(results) > 1 else results[0]

def main():
    # Load model, tokenizer, and translator
    model, tokenizer = load_model()
    translator = load_translator()
    
    # Clear the loading message
    loading_container.empty()
    
    if model is None or tokenizer is None or translator is None:
        st.error("❌ Failed to initialize the sentiment analyzer or translation pipeline. Please check the logs.")
        return
    
    # Initialize analyzer
    analyzer = TweetSentimentAnalyzer(model, tokenizer, translator)
    
    # App header
    st.markdown('<div class="title">Tweet Sentiment Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Analyze the sentiment of French tweets with a fine-tuned DistilBERT model</div>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📝 Single Tweet", "📊 Batch Analysis", "ℹ️ About"])
    
    # Single Tweet Analysis
    with tab1:
        with st.container():
            st.markdown("### Analyze a Single French Tweet")
            st.markdown("Enter a French tweet below, and we'll translate it to English and analyze its sentiment.")
            
            # Input form with reset button
            with st.form("single_tweet_form"):
                tweet_text = st.text_area(
                    "French Tweet",
                    height=100,
                    placeholder="e.g., Je viens de voir un film incroyable ce soir ! #cinéma",
                    help="Enter a French tweet (max 280 characters)."
                )
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    submit_button = st.form_submit_button("Analyze", type="primary")
                with col2:
                    reset_button = st.form_submit_button("Reset")
                
                if reset_button:
                    st.session_state.tweet_text = ""
                    st.rerun()
                
                if submit_button and tweet_text.strip():
                    try:
                        with st.spinner("🔄 Processing tweet..."):
                            result = analyzer.predict(tweet_text)
                        
                        # Display results in a card
                        with st.container():
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.markdown("#### Analysis Results")
                            
                            st.markdown(f"**Original (French):** {result['original_text']}")
                            st.markdown(f"**Translated (English):** {result['translated_text']}")
                            
                            sentiment_class = f"sentiment-{result['sentiment']}"
                            st.markdown(f"**Sentiment:** <span class='{sentiment_class}'>{result['sentiment'].upper()}</span>", unsafe_allow_html=True)
                            st.markdown(f"**Confidence:** {result['confidence']:.2f}")
                            
                            # Sentiment score chart
                            fig = px.bar(
                                x=list(result['scores'].values()),
                                y=list(result['scores'].keys()),
                                orientation='h',
                                labels={'x': 'Score', 'y': 'Sentiment'},
                                color_discrete_sequence=[COLORS[label] for label in LABELS],
                                text_auto='.2f'
                            )
                            fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
                            st.plotly_chart(fig, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"❌ Error analyzing tweet: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                elif submit_button:
                    st.warning("⚠️ Please enter a tweet to analyze.")
    
    # Batch Analysis
    with tab2:
        with st.container():
            st.markdown("### Analyze Multiple French Tweets")
            st.markdown("Upload a CSV file with French tweets or enter multiple tweets (one per line).")
            
            # File upload and text input
            with st.form("batch_analysis_form"):
                uploaded_file = st.file_uploader(
                    "Upload a CSV file",
                    type=["csv"],
                    help="CSV should have a column with French tweets (e.g., 'text' or 'tweet')."
                )
                
                batch_tweets = st.text_area(
                    "Or enter French tweets (one per line)",
                    height=150,
                    placeholder="e.g.,\nService client horrible aujourd'hui. 😣\nRendez-vous chez le dentiste cet après-midi.",
                    help="Enter one French tweet per line."
                )
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    analyze_button = st.form_submit_button("Analyze Batch", type="primary")
                with col2:
                    reset_batch_button = st.form_submit_button("Reset")
                
                if reset_batch_button:
                    st.session_state.batch_tweets = ""
                    st.session_state.uploaded_file = None
                    st.rerun()
                
                if analyze_button:
                    tweets_to_analyze = []
                    
                    # Process uploaded file
                    if uploaded_file is not None:
                        try:
                            df = pd.read_csv(uploaded_file)
                            possible_cols = ['text', 'tweet', 'content', 'message']
                            text_col = None
                            
                            for col in possible_cols:
                                if col in df.columns:
                                    text_col = col
                                    break
                            
                            if text_col is None and len(df.columns) > 0:
                                text_col = df.columns[0]
                            
                            if text_col is not None:
                                tweets_to_analyze = df[text_col].fillna("").tolist()
                                st.info(f"📄 Found {len(tweets_to_analyze)} tweets in the CSV file using column '{text_col}'")
                                # Preview first few rows
                                st.markdown("**CSV Preview (First 5 Rows):**")
                                st.dataframe(df.head(), use_container_width=True)
                            else:
                                st.error("❌ Could not find a suitable text column in the CSV file.")
                        except Exception as e:
                            st.error(f"❌ Error processing CSV file: {str(e)}")
                    
                    # Add tweets from text area
                    if batch_tweets.strip():
                        text_tweets = [t for t in batch_tweets.split('\n') if t.strip()]
                        if text_tweets:
                            tweets_to_analyze.extend(text_tweets)
                            st.info(f"📝 Added {len(text_tweets)} tweets from the text area")
                    
                    # Analyze tweets
                    if tweets_to_analyze:
                        try:
                            max_tweets = 100
                            if len(tweets_to_analyze) > max_tweets:
                                st.warning(f"⚠️ Limiting analysis to the first {max_tweets} tweets to prevent timeout.")
                                tweets_to_analyze = tweets_to_analyze[:max_tweets]
                            
                            progress_bar = st.progress(0)
                            batch_size = 10
                            all_results = []
                            
                            for i in range(0, len(tweets_to_analyze), batch_size):
                                batch = tweets_to_analyze[i:i+batch_size]
                                results = analyzer.predict(batch)
                                all_results.extend(results)
                                progress = min(1.0, (i + batch_size) / len(tweets_to_analyze))
                                progress_bar.progress(progress)
                            
                            progress_bar.empty()
                            
                            # Create results dataframe
                            results_df = pd.DataFrame([
                                {
                                    "Original Tweet": r['original_text'][:100] + "..." if len(r['original_text']) > 100 else r['original_text'],
                                    "Translated Tweet": r['translated_text'][:100] + "..." if len(r['translated_text']) > 100 else r['translated_text'],
                                    "Sentiment": r['sentiment'],
                                    "Confidence": r['confidence'],
                                    "Negative Score": r['scores']['negative'],
                                    "Neutral Score": r['scores']['neutral'],
                                    "Positive Score": r['scores']['positive']
                                }
                                for r in all_results
                            ])
                            
                            # Display results in a card
                            with st.container():
                                st.markdown('<div class="card">', unsafe_allow_html=True)
                                st.markdown("#### Batch Analysis Results")
                                st.dataframe(results_df, use_container_width=True)
                                
                                # Summary visualizations
                                sentiment_counts = results_df['Sentiment'].value_counts().reset_index()
                                sentiment_counts.columns = ['Sentiment', 'Count']
                                
                                col1, col2 = st.columns([1, 1])
                                
                                with col1:
                                    fig1 = px.pie(
                                        sentiment_counts,
                                        values='Count',
                                        names='Sentiment',
                                        color='Sentiment',
                                        color_discrete_map=COLORS,
                                        title="Sentiment Distribution"
                                    )
                                    fig1.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                                    st.plotly_chart(fig1, use_container_width=True)
                                
                                with col2:
                                    fig2 = px.bar(
                                        sentiment_counts,
                                        x='Sentiment',
                                        y='Count',
                                        color='Sentiment',
                                        color_discrete_map=COLORS,
                                        title="Sentiment Counts"
                                    )
                                    fig2.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                                    st.plotly_chart(fig2, use_container_width=True)
                                
                                # Download button
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results as CSV",
                                    data=csv,
                                    file_name="tweet_sentiment_analysis.csv",
                                    mime="text/csv",
                                    type="primary"
                                )
                                st.markdown('</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"❌ Error in batch analysis: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                    else:
                        st.warning("⚠️ Please upload a CSV file or enter tweets to analyze.")
    
    # About tab
    with tab3:
        with st.container():
            st.markdown("### About This App")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("""
            #### Model Information
            This application uses a fine-tuned **DistilBERT** model to analyze the sentiment of tweets. Since the model was trained on English tweets, French tweets are translated to English using the **Helsinki-NLP/opus-mt-fr-en** model before analysis.
            
            **Sentiment Categories:**
            - 😠 **Negative**
            - 😐 **Neutral**
            - 😊 **Positive**
            
            #### How to Use
            - **Single Tweet**: Enter a French tweet and click "Analyze" to see the sentiment.
            - **Batch Analysis**: Upload a CSV file or enter multiple French tweets (one per line) for bulk analysis.
            
            #### Why This App?
            Built to provide an intuitive and visually appealing way to analyze tweet sentiments, this app combines powerful NLP models with a user-friendly interface.
            """)
            st.markdown("---")
            st.markdown("**Tech Stack:** Streamlit, Transformers, TensorFlow, Plotly")
            st.markdown("**Created by:** [Your Name] | Powered by ❤️")
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()