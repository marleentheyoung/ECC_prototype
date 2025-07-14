# evolution_analysis.py - Topic evolution analysis functions
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import get_snippet_topics, determine_market, snippet_matches_topic
from data_loaders import load_eu_rag, load_us_rag

def analyze_topic_evolution_simple(selected_snippets, topic_model, selected_topic, year_range, time_granularity, topic_names):
    """Analyze how a specific topic evolves over time for selected snippets."""
    
    # Get the topic words to identify relevant snippets
    topic_words = topic_model.get_topic(selected_topic)
    if not topic_words:
        return []
    
    # Extract keywords from the topic
    keywords = [word for word, _ in topic_words[:10]]  # Top 10 words
    
    # Filter snippets that match the topic and are in the year range
    relevant_snippets = []
    for snippet in selected_snippets:
        if snippet.year and str(snippet.year).isdigit():
            year = int(snippet.year)
            if year_range[0] <= year <= year_range[1]:
                if snippet_matches_topic(snippet.text, keywords):
                    relevant_snippets.append(snippet)
    
    # Group by time periods
    time_groups = {}
    for snippet in relevant_snippets:
        if time_granularity == "Yearly":
            period = str(snippet.year)
        else:  # Quarterly
            period = f"{snippet.year}-Q{snippet.quarter}"
        
        if period not in time_groups:
            time_groups[period] = {
                'count': 0, 
                'companies': set(), 
                'sentiment': {'opportunity': 0, 'neutral': 0, 'risk': 0}
            }
        
        time_groups[period]['count'] += 1
        time_groups[period]['companies'].add(snippet.ticker)
        if snippet.climate_sentiment:
            time_groups[period]['sentiment'][snippet.climate_sentiment] += 1
    
    # Convert to list format for visualization
    evolution_data = []
    for period in sorted(time_groups.keys()):
        data = time_groups[period]
        evolution_data.append({
            'period': period,
            'count': data['count'],
            'companies': len(data['companies']),
            'sentiment_opportunity': data['sentiment']['opportunity'],
            'sentiment_neutral': data['sentiment']['neutral'],
            'sentiment_risk': data['sentiment']['risk']
        })
    
    return evolution_data

def analyze_all_topics_evolution(rag, topic_model, topic_names, show_eu, show_us, year_range, time_granularity):
    """Analyze how all topics evolve over time across markets."""
    
    # Get all valid topics (excluding outliers)
    topic_info = topic_model.get_topic_info()
    valid_topics = topic_info[topic_info['Topic'] != -1]['Topic'].tolist()
    
    if not valid_topics:
        return [], []
    
    # Get topic words for each topic
    topic_keywords = {}
    for topic_num in valid_topics:
        topic_words = topic_model.get_topic(topic_num)
        if topic_words:
            topic_keywords[topic_num] = [word for word, _ in topic_words[:10]]
    
    # Handle different market scenarios
    current_market = st.session_state.current_market
    
    if current_market == "Full":
        # Use all snippets from the full index, determine market for each
        market_snippets = []
        for snippet in rag.snippets:
            if snippet.year and str(snippet.year).isdigit():
                year = int(snippet.year)
                if year_range[0] <= year <= year_range[1]:
                    topics = get_snippet_topics(snippet.text, topic_keywords)
                    if topics:  # Only include if it matches at least one topic
                        market = determine_market(snippet)
                        if (market == 'EU' and show_eu) or (market == 'US' and show_us):
                            market_snippets.append((snippet, market, topics))
    
    elif current_market == "Combined":
        # Load both markets separately (fallback method)
        try:
            eu_rag = load_eu_rag()
            eu_snippets = eu_rag.snippets
        except:
            eu_snippets = []
        
        try:
            us_rag = load_us_rag()
            us_snippets = us_rag.snippets
        except:
            us_snippets = []
        
        # Create market-labeled snippets
        market_snippets = []
        if show_eu:
            for snippet in eu_snippets:
                if snippet.year and str(snippet.year).isdigit():
                    year = int(snippet.year)
                    if year_range[0] <= year <= year_range[1]:
                        topics = get_snippet_topics(snippet.text, topic_keywords)
                        if topics:  # Only include if it matches at least one topic
                            market_snippets.append((snippet, 'EU', topics))
        
        if show_us:
            for snippet in us_snippets:
                if snippet.year and str(snippet.year).isdigit():
                    year = int(snippet.year)
                    if year_range[0] <= year <= year_range[1]:
                        topics = get_snippet_topics(snippet.text, topic_keywords)
                        if topics:  # Only include if it matches at least one topic
                            market_snippets.append((snippet, 'US', topics))
    
    else:
        # Single market data
        if (current_market == "EU" and not show_eu) or (current_market == "US" and not show_us):
            return [], []
        
        market_snippets = []
        for snippet in rag.snippets:
            if snippet.year and str(snippet.year).isdigit():
                year = int(snippet.year)
                if year_range[0] <= year <= year_range[1]:
                    topics = get_snippet_topics(snippet.text, topic_keywords)
                    if topics:  # Only include if it matches at least one topic
                        market_snippets.append((snippet, current_market, topics))
    
    # Group by time periods and topics
    time_groups = {}
    for snippet, market, topics in market_snippets:
        if time_granularity == "Yearly":
            period = str(snippet.year)
        else:  # Quarterly
            period = f"{snippet.year}-Q{snippet.quarter}"
        
        if period not in time_groups:
            time_groups[period] = {}
        
        # Count each topic for this snippet
        for topic_num in topics:
            topic_key = f"{market}_{topic_num}"
            if topic_key not in time_groups[period]:
                time_groups[period][topic_key] = 0
            time_groups[period][topic_key] += 1
    
    # Convert to list format for visualization
    evolution_data = []
    all_periods = sorted(time_groups.keys())
    
    for period in all_periods:
        period_data = {'period': period}
        
        # Add counts for each topic and market combination
        for topic_num in valid_topics:
            topic_name = topic_names.get(topic_num, f"Topic {topic_num}")
            
            if show_eu:
                eu_key = f"EU_{topic_num}"
                period_data[f"EU_{topic_name}"] = time_groups[period].get(eu_key, 0)
            
            if show_us:
                us_key = f"US_{topic_num}"
                period_data[f"US_{topic_name}"] = time_groups[period].get(us_key, 0)
        
        evolution_data.append(period_data)
    
    return evolution_data, valid_topics

def analyze_topic_evolution(rag, topic_model, selected_topic, show_eu, show_us, year_range, time_granularity):
    """Analyze how a specific topic evolves over time across markets (legacy function for compatibility)."""
    # Get the topic words to identify relevant snippets
    topic_words = topic_model.get_topic(selected_topic)
    if not topic_words:
        return []
    
    # Extract keywords from the topic
    keywords = [word for word, _ in topic_words[:10]]
    
    # Helper function to check if snippet matches topic
    def snippet_matches_topic_local(snippet_text, keywords):
        text_lower = snippet_text.lower()
        return any(keyword.lower() in text_lower for keyword in keywords)
    
    # Handle different market scenarios
    current_market = st.session_state.current_market
    
    if current_market == "Full":
        # Use all snippets from the full index, determine market for each
        market_snippets = []
        for snippet in rag.snippets:
            if snippet.year and str(snippet.year).isdigit():
                year = int(snippet.year)
                if year_range[0] <= year <= year_range[1]:
                    if snippet_matches_topic_local(snippet.text, keywords):
                        market = determine_market(snippet)
                        if (market == 'EU' and show_eu) or (market == 'US' and show_us):
                            market_snippets.append((snippet, market))
    
    elif current_market == "Combined":
        # Load both markets separately (fallback method)
        try:
            eu_rag = load_eu_rag()
            eu_snippets = eu_rag.snippets
        except:
            eu_snippets = []
        
        try:
            us_rag = load_us_rag()
            us_snippets = us_rag.snippets
        except:
            us_snippets = []
        
        # Create market-labeled snippets
        market_snippets = []
        if show_eu:
            for snippet in eu_snippets:
                if snippet.year and str(snippet.year).isdigit():
                    year = int(snippet.year)
                    if year_range[0] <= year <= year_range[1]:
                        if snippet_matches_topic_local(snippet.text, keywords):
                            market_snippets.append((snippet, 'EU'))
        
        if show_us:
            for snippet in us_snippets:
                if snippet.year and str(snippet.year).isdigit():
                    year = int(snippet.year)
                    if year_range[0] <= year <= year_range[1]:
                        if snippet_matches_topic_local(snippet.text, keywords):
                            market_snippets.append((snippet, 'US'))
    
    else:
        # Single market data
        if (current_market == "EU" and not show_eu) or (current_market == "US" and not show_us):
            return []
        
        market_snippets = []
        for snippet in rag.snippets:
            if snippet.year and str(snippet.year).isdigit():
                year = int(snippet.year)
                if year_range[0] <= year <= year_range[1]:
                    if snippet_matches_topic_local(snippet.text, keywords):
                        market_snippets.append((snippet, current_market))
    
    # Group by time periods
    time_groups = {}
    for snippet, market in market_snippets:
        if time_granularity == "Yearly":
            period = str(snippet.year)
        else:  # Quarterly
            period = f"{snippet.year}-Q{snippet.quarter}"
        
        if period not in time_groups:
            time_groups[period] = {
                'EU_count': 0, 'US_count': 0,
                'EU_companies': set(), 'US_companies': set(),
                'EU_sentiment': {'opportunity': 0, 'neutral': 0, 'risk': 0},
                'US_sentiment': {'opportunity': 0, 'neutral': 0, 'risk': 0}
            }
        
        time_groups[period][f'{market}_count'] += 1
        time_groups[period][f'{market}_companies'].add(snippet.ticker)
        if snippet.climate_sentiment:
            time_groups[period][f'{market}_sentiment'][snippet.climate_sentiment] += 1
    
    # Convert to list format for visualization
    evolution_data = []
    for period in sorted(time_groups.keys()):
        data = time_groups[period]
        evolution_data.append({
            'period': period,
            'EU_count': data['EU_count'] if show_eu else 0,
            'US_count': data['US_count'] if show_us else 0,
            'EU_companies': len(data['EU_companies']) if show_eu else 0,
            'US_companies': len(data['US_companies']) if show_us else 0,
            'EU_sentiment_opportunity': data['EU_sentiment']['opportunity'] if show_eu else 0,
            'EU_sentiment_neutral': data['EU_sentiment']['neutral'] if show_eu else 0,
            'EU_sentiment_risk': data['EU_sentiment']['risk'] if show_eu else 0,
            'US_sentiment_opportunity': data['US_sentiment']['opportunity'] if show_us else 0,
            'US_sentiment_neutral': data['US_sentiment']['neutral'] if show_us else 0,
            'US_sentiment_risk': data['US_sentiment']['risk'] if show_us else 0
        })
    
    return evolution_data