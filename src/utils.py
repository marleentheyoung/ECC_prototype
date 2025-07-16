# utils.py - Utility functions
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
import pandas as pd
from typing import List, Dict, Any
from collections import Counter
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
    from src.config import CLAUDE_CONFIG
    
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

def format_topic_summary(topic_results: Dict[str, List]) -> str:
    """Format topic search results into a summary string."""
    summary_lines = []
    total_snippets = 0
    
    for topic_name, snippets in topic_results.items():
        count = len(snippets)
        total_snippets += count
        
        if count > 0:
            # Get some basic stats
            companies = len(set(s.ticker for s in snippets))
            avg_score = sum(getattr(s, 'score', 0.5) for s in snippets) / count
            
            summary_lines.append(
                f"• **{topic_name}**: {count} snippets, {companies} companies, "
                f"avg relevance: {avg_score:.3f}"
            )
        else:
            summary_lines.append(f"• **{topic_name}**: No snippets found")
    
    summary = f"**Total: {total_snippets} snippets across {len(topic_results)} topics**\n\n"
    summary += "\n".join(summary_lines)
    
    return summary

def validate_search_parameters(query: str, threshold: float) -> tuple[bool, str]:
    """Validate search parameters and return (is_valid, error_message)."""
    if not query or not query.strip():
        return False, "Please enter a search query."
    
    if not (0.0 <= threshold <= 1.0):
        return False, "Relevance threshold must be between 0.0 and 1.0."
    
    if len(query.strip()) < 3:
        return False, "Search query must be at least 3 characters long."
    
    return True, ""

def calculate_topic_overlap(topic_results_1: Dict[str, List], topic_results_2: Dict[str, List]) -> Dict[str, Dict[str, float]]:
    """Calculate overlap between two sets of topic results."""
    overlap_matrix = {}
    
    for topic1, snippets1 in topic_results_1.items():
        overlap_matrix[topic1] = {}
        texts1 = set(s.text for s in snippets1)
        
        for topic2, snippets2 in topic_results_2.items():
            texts2 = set(s.text for s in snippets2)
            
            if len(texts1) == 0 or len(texts2) == 0:
                overlap_matrix[topic1][topic2] = 0.0
            else:
                intersection = len(texts1.intersection(texts2))
                union = len(texts1.union(texts2))
                jaccard_similarity = intersection / union if union > 0 else 0.0
                overlap_matrix[topic1][topic2] = jaccard_similarity
    
    return overlap_matrix

def export_topic_comparison(topic_results: Dict[str, List], filename: str = "topic_comparison.csv") -> pd.DataFrame:
    """Export topic comparison data to CSV format."""
    comparison_data = []
    
    for topic_name, snippets in topic_results.items():
        for snippet in snippets:
            comparison_data.append({
                'topic_name': topic_name,
                'text': snippet.text,
                'company': snippet.company,
                'ticker': snippet.ticker,
                'year': snippet.year,
                'quarter': snippet.quarter,
                'date': snippet.date,
                'speaker': snippet.speaker,
                'profession': snippet.profession,
                'climate_sentiment': snippet.climate_sentiment,
                'relevance_score': getattr(snippet, 'score', 0.5),
                'llm_validation': getattr(snippet, 'llm_validation', 'N/A'),
                'llm_confidence': getattr(snippet, 'llm_confidence', 0.5)
            })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        return df
    else:
        return pd.DataFrame()

def get_topic_timeline(snippets: List, time_granularity: str = "Yearly") -> pd.DataFrame:
    """Create a timeline of topic mentions."""
    timeline_data = []
    
    for snippet in snippets:
        if snippet.year and str(snippet.year).isdigit():
            if time_granularity == "Yearly":
                period = str(snippet.year)
            else:  # Quarterly
                period = f"{snippet.year}-Q{snippet.quarter}"
            
            timeline_data.append({
                'period': period,
                'company': snippet.company,
                'ticker': snippet.ticker,
                'text': snippet.text,
                'sentiment': snippet.climate_sentiment,
                'score': getattr(snippet, 'score', 0.5)
            })
    
    return pd.DataFrame(timeline_data)

def filter_snippets_by_criteria(snippets: List, 
                               companies: List[str] = None,
                               sentiment: str = None,
                               year_range: tuple = None,
                               min_score: float = None) -> List:
    """Filter snippets based on various criteria."""
    filtered = []
    
    for snippet in snippets:
        # Company filter
        if companies and snippet.ticker not in companies:
            continue
        
        # Sentiment filter
        if sentiment and snippet.climate_sentiment != sentiment:
            continue
        
        # Year range filter
        if year_range and snippet.year:
            try:
                year = int(snippet.year)
                if not (year_range[0] <= year <= year_range[1]):
                    continue
            except (ValueError, TypeError):
                continue
        
        # Score filter
        if min_score and hasattr(snippet, 'score') and snippet.score < min_score:
            continue
        
        filtered.append(snippet)
    
    return filtered

def create_topic_word_cloud_data(topic_results: Dict[str, List]) -> Dict[str, Dict[str, int]]:
    """Create word frequency data for word clouds for each topic."""
    from collections import Counter
    import re
    
    topic_word_data = {}
    
    for topic_name, snippets in topic_results.items():
        # Combine all text for this topic
        all_text = " ".join([snippet.text for snippet in snippets])
        
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
        
        # Remove common stopwords
        stopwords = create_custom_stopwords()
        words = [word for word in words if word not in stopwords]
        
        # Count frequency
        word_freq = Counter(words)
        topic_word_data[topic_name] = dict(word_freq.most_common(50))
    
    return topic_word_data

def calculate_topic_statistics(topic_results: Dict[str, List]) -> Dict[str, Dict[str, Any]]:
    """Calculate comprehensive statistics for each topic."""
    stats = {}
    
    for topic_name, snippets in topic_results.items():
        if not snippets:
            stats[topic_name] = {
                'total_snippets': 0,
                'unique_companies': 0,
                'avg_relevance': 0,
                'sentiment_distribution': {'opportunity': 0, 'neutral': 0, 'risk': 0},
                'year_range': None,
                'top_companies': []
            }
            continue
        
        # Basic counts
        total_snippets = len(snippets)
        unique_companies = len(set(s.ticker for s in snippets))
        
        # Relevance score
        avg_relevance = sum(getattr(s, 'score', 0.5) for s in snippets) / total_snippets
        
        # Sentiment distribution
        sentiment_counts = {'opportunity': 0, 'neutral': 0, 'risk': 0}
        for snippet in snippets:
            sentiment = getattr(snippet, 'climate_sentiment', 'neutral')
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
        
        # Year range
        years = [int(s.year) for s in snippets if s.year and str(s.year).isdigit()]
        year_range = (min(years), max(years)) if years else None
        
        # Top companies by mention count
        company_counts = Counter(s.ticker for s in snippets)
        top_companies = company_counts.most_common(5)
        
        stats[topic_name] = {
            'total_snippets': total_snippets,
            'unique_companies': unique_companies,
            'avg_relevance': round(avg_relevance, 3),
            'sentiment_distribution': sentiment_counts,
            'year_range': year_range,
            'top_companies': top_companies
        }
    
    return stats

def merge_topic_results(results1: Dict[str, List], results2: Dict[str, List]) -> Dict[str, List]:
    """Merge two topic results dictionaries, combining snippets for matching topics."""
    merged = results1.copy()
    
    for topic_name, snippets in results2.items():
        if topic_name in merged:
            # Combine snippets, removing duplicates based on text
            existing_texts = set(s.text for s in merged[topic_name])
            new_snippets = [s for s in snippets if s.text not in existing_texts]
            merged[topic_name].extend(new_snippets)
        else:
            merged[topic_name] = snippets
    
    return merged

def save_topic_results_to_session(topic_results: Dict[str, List], session_key: str):
    """Save topic results to Streamlit session state."""
    st.session_state[session_key] = topic_results
    st.session_state[f"{session_key}_timestamp"] = pd.Timestamp.now()

def load_topic_results_from_session(session_key: str) -> Dict[str, List]:
    """Load topic results from Streamlit session state."""
    return st.session_state.get(session_key, {})

def create_topic_comparison_matrix(topic_results: Dict[str, List]) -> pd.DataFrame:
    """Create a matrix showing topic similarities based on shared snippets."""
    topics = list(topic_results.keys())
    matrix_data = []
    
    for topic1 in topics:
        row = []
        snippets1 = set(s.text for s in topic_results[topic1])
        
        for topic2 in topics:
            snippets2 = set(s.text for s in topic_results[topic2])
            
            if len(snippets1) == 0 or len(snippets2) == 0:
                similarity = 0.0
            else:
                intersection = len(snippets1.intersection(snippets2))
                union = len(snippets1.union(snippets2))
                similarity = intersection / union if union > 0 else 0.0
            
            row.append(similarity)
        
        matrix_data.append(row)
    
    return pd.DataFrame(matrix_data, index=topics, columns=topics)