# topic_analysis.py - Topic analysis and modeling functions
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import umap.umap_ as umap
from bertopic import BERTopic
from config import TOPIC_ANALYSIS_CONFIG
from utils import create_custom_stopwords, generate_topic_names

def run_topic_analysis(selected_snippets, nr_topics):
    """Run topic analysis on selected snippets."""
    texts = [snippet.text for snippet in selected_snippets]
    
    if len(texts) < nr_topics * 20:
        st.warning(f"Not enough texts ({len(texts)}) for {nr_topics} topics. Try reducing the number of topics or selecting more snippets.")
        return None, None, None
    
    # Create vectorizer with custom stopwords
    custom_stopwords = create_custom_stopwords()
    vectorizer_model = CountVectorizer(
        stop_words=custom_stopwords,
        max_features=TOPIC_ANALYSIS_CONFIG['vectorizer_max_features'],
        min_df=TOPIC_ANALYSIS_CONFIG['vectorizer_min_df'],
        max_df=TOPIC_ANALYSIS_CONFIG['vectorizer_max_df']
    )
    
    # UMAP configuration for stability
    umap_model = umap.UMAP(
        n_neighbors=min(TOPIC_ANALYSIS_CONFIG['umap_n_neighbors'], len(texts)//5), 
        n_components=TOPIC_ANALYSIS_CONFIG['umap_n_components'],
        metric='cosine', 
        n_jobs=1,
        random_state=42,
        min_dist=TOPIC_ANALYSIS_CONFIG['umap_min_dist'],
        spread=TOPIC_ANALYSIS_CONFIG['umap_spread']
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        top_n_words=TOPIC_ANALYSIS_CONFIG['top_n_words'],
        nr_topics=nr_topics,
        calculate_probabilities=False,
        vectorizer_model=vectorizer_model,
        verbose=False
    )
    
    topics, probs = topic_model.fit_transform(texts)
    topic_info = topic_model.get_topic_info()
    
    return topic_model, topics, topic_info

def display_topic_results(topic_model, topic_info, topic_names):
    """Display topic analysis results."""
    # Display topics with LLM-generated names
    st.subheader("📋 Discovered Topics")
    topic_display_df = topic_info.copy()
    topic_display_df['LLM_Name'] = topic_display_df['Topic'].map(
        lambda x: topic_names.get(x, f"Topic {x}")
    )
    st.dataframe(topic_display_df[['Topic', 'LLM_Name', 'Count', 'Name']])

    # Show Word Clouds for each topic
    st.subheader("☁️ Word Clouds per Topic")
    for topic_num in topic_info['Topic'].tolist():
        if topic_num == -1:
            continue  # Skip outliers

        words = topic_model.get_topic(topic_num)
        if not words:
            st.warning(f"Topic {topic_num} has no words to display.")
            continue

        word_list = [w[0] for w in words if w[0].isalpha()]
        if not word_list:
            st.warning(f"Topic {topic_num} has no valid words to display.")
            continue

        # Generate word cloud
        wc = WordCloud(width=400, height=200, background_color='white').generate(' '.join(word_list))

        llm_name = topic_names.get(topic_num, f"Topic {topic_num}")
        st.markdown(f"**{llm_name}** (Topic {topic_num})")
        st.caption(f"Top words: {', '.join(word_list[:6])}")
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

def create_topic_results_dataframe(selected_snippets, topics, topic_names):
    """Create results dataframe for export."""
    topic_results = []
    for i, snippet in enumerate(selected_snippets):
        topic_num = topics[i]
        topic_name = topic_names.get(topic_num, f"Topic {topic_num}")
        
        topic_results.append({
            'text': snippet.text,
            'company': snippet.company,
            'ticker': snippet.ticker,
            'year': snippet.year,
            'quarter': snippet.quarter,
            'date': snippet.date,
            'speaker': snippet.speaker,
            'profession': snippet.profession,
            'climate_sentiment': snippet.climate_sentiment,
            'topic_number': topic_num,
            'topic_name': topic_name
        })
    
    return pd.DataFrame(topic_results)