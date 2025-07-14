# utils.py - Utility functions
import streamlit as st
import pandas as pd
from typing import List, Dict, Any
from anthropic import Anthropic

def get_selected_snippets():
    """Get the currently selected snippets from session state."""
    return st.session_state.get('selected_snippets', [])

def determine_market(snippet):
    """Determine if snippet is from EU or US market based on available info."""
    if hasattr(snippet, 'market'):
        return snippet.market
    
    # Fallback: try to infer from ticker length or other characteristics
    if len(snippet.ticker) <= 4 and snippet.ticker.isalpha():
        return 'US'
    else:
        return 'EU'

def get_snippet_topics(snippet_text, topic_keywords):
    """Return list of topics that match this snippet."""
    text_lower = snippet_text.lower()
    matching_topics = []
    
    for topic_num, keywords in topic_keywords.items():
        if any(keyword.lower() in text_lower for keyword in keywords):
            matching_topics.append(topic_num)
    
    return matching_topics

def snippet_matches_topic(snippet_text, keywords):
    """Check if snippet matches topic based on keywords."""
    text_lower = snippet_text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)

def create_custom_stopwords():
    """Create custom stopwords for topic modeling."""
    from sklearn.feature_extraction import text
    
    return list(text.ENGLISH_STOP_WORDS.union({
        "million", "quarter", "company", "business", "group", "share", 
        "billion", "sales", "revenues", "revenue", "year", "time",
        "percent", "growth", "market", "increase", "decrease", "good", "well"
    }))

def display_results(results):
    """Display search results in a nice format."""
    if not results:
        st.warning("No results found.")
        return
    
    st.subheader(f"Preview of {len(results)} results")
    
    for i, result in enumerate(results):
        with st.expander(f"Result {i+1}: {result['company']} ({result['score']:.3f})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Text:** {result['text']}")
                st.write(f"**Speaker:** {result['speaker']} ({result['profession']})")
            
            with col2:
                st.metric("Relevance Score", f"{result['score']:.3f}")
                st.write(f"**Date:** {result['date']}")
                st.write(f"**Quarter:** {result['quarter']} {result['year']}")
                if result['climate_sentiment']:
                    sentiment_color = {
                        'opportunity': '🟢',
                        'neutral': '🟡', 
                        'risk': '🔴'
                    }
                    st.write(f"**Sentiment:** {sentiment_color.get(result['climate_sentiment'], '')} {result['climate_sentiment']}")

def generate_topic_names(topic_model, topic_info: pd.DataFrame, api_key: str = None) -> Dict[int, str]:
    """Generate meaningful topic names using Anthropic Claude based on top words for each topic."""
    from config import CLAUDE_CONFIG
    
    topic_names = {}

    try:
        client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

        for topic_num in topic_info['Topic'].tolist():
            if topic_num == -1:
                continue  # Skip outliers

            # Get top words for this topic
            words = topic_model.get_topic(topic_num)
            if not words:
                topic_names[topic_num] = f"Topic {topic_num}"
                continue

            top_words = [word for word, _ in words[:10]]
            words_str = ", ".join(top_words)

            # Create Claude prompt
            prompt = f"""Based on these keywords from a topic model analysis of green investment and climate-related earnings call transcripts, suggest a concise and descriptive topic name (2-4 words):

Keywords: {words_str}

Provide only the topic name, no explanation. If the words are incoherent, return only the topic number followed by 'Incoherent'."""

            try:
                response = client.messages.create(
                    model=CLAUDE_CONFIG['model'],
                    max_tokens=CLAUDE_CONFIG['max_tokens'],
                    temperature=CLAUDE_CONFIG['temperature'],
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                topic_name = response.content[0].text.strip()
                topic_names[topic_num] = topic_name

            except Exception as e:
                st.warning(f"Error generating name for topic {topic_num}: {str(e)}")
                topic_names[topic_num] = f"Topic {topic_num}"

    except Exception as e:
        st.error(f"Error setting up Anthropic client: {str(e)}")
        return {topic_num: f"Topic {topic_num}" for topic_num in topic_info['Topic'].tolist() if topic_num != -1}

    return topic_names