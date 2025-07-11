# interface.py - Simplified threading for macOS compatibility
import os
import sys

# Force single-threaded execution for macOS stability
if sys.platform == 'darwin':  # macOS
    os.environ['LOKY_MAX_CPU_COUNT'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'
    os.environ['NUMBA_NUM_THREADS'] = '1'

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from green_investment_rag import GreenInvestmentRAG
from helpers import load_data, filter_results
from sklearn.feature_extraction import text
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from typing import List, Dict

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import libraries with error handling
try:
    from umap import UMAP
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    LIBRARIES_AVAILABLE = True
except ImportError as e:
    st.error(f"Missing required libraries: {e}")
    st.info("Please install: pip install umap-learn bertopic")
    LIBRARIES_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Green Investment Analyzer",
    page_icon="🌱",
    layout="wide"
)

def generate_topic_names(topic_model, topic_info: pd.DataFrame, api_key: str = None) -> Dict[int, str]:
    """Generate meaningful topic names using Anthropic Claude based on top words for each topic."""
    topic_names = {}

    # Set up Anthropic client
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
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=100,
                    temperature=0.2,
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

# Cached loaders for different markets
@st.cache_resource
def load_eu_rag():
    rag = GreenInvestmentRAG()
    rag.load_market_data('EU')
    return rag

@st.cache_resource
def load_us_rag():
    rag = GreenInvestmentRAG()
    rag.load_market_data('US')
    return rag

@st.cache_resource
def load_combined_rag():
    rag = GreenInvestmentRAG()
    rag.load_combined_data()
    return rag

def get_selected_snippets():
    """Get the currently selected snippets from session state."""
    return st.session_state.get('selected_snippets', [])

def main():
    st.title("🌱 Green Investment Analyzer")
    st.subheader("Extract climate investment insights from earnings calls")
    
    # Check if libraries are available
    if not LIBRARIES_AVAILABLE:
        st.error("Required libraries not available. Please install them first.")
        st.code("pip install umap-learn bertopic")
        return
    
    # Sidebar for market selection
    with st.sidebar:
        st.header("🌍 Market Selection")
        
        market_option = st.selectbox(
            "Select Market(s)",
            ["EU (STOXX 600)", "US (S&P 500)", "Combined (EU + US)"],
            index=0
        )
        
        if st.button("Load Market Data"):
            with st.spinner(f"Loading {market_option} data..."):
                try:
                    if market_option == "EU (STOXX 600)":
                        rag = load_eu_rag()
                        st.session_state.current_market = "EU"
                    elif market_option == "US (S&P 500)":
                        rag = load_us_rag()
                        st.session_state.current_market = "US"
                    else:  # Combined
                        rag = load_combined_rag()
                        st.session_state.current_market = "Combined"
                    
                    st.session_state.rag_system = rag
                    st.session_state.data_loaded = True
                    st.success(f"Loaded {len(rag.snippets)} snippets from {market_option}!")
                    
                except Exception as e:
                    st.error(f"Error loading {market_option} data: {e}")
                    st.session_state.data_loaded = False

        # API Key input for LLM topic naming
        st.header("🤖 LLM Settings")
        api_key = st.text_input("Anthropic API Key (optional)", type="password", 
                               help="Enter your Anthropic API key for automatic topic naming")
        if api_key:
            st.session_state.anthropic_api_key = api_key

    # Initialize session state keys if not present
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'current_market' not in st.session_state:
        st.session_state.current_market = None
    if 'anthropic_api_key' not in st.session_state:
        st.session_state.anthropic_api_key = None
    if 'selected_snippets' not in st.session_state:
        st.session_state.selected_snippets = []

    if not st.session_state.data_loaded:
        st.warning("Please select and load market data using the sidebar.")
        return
    
    rag = st.session_state.rag_system
    
    # Display current market info
    st.info(f"📊 Currently analyzing: {st.session_state.current_market} market with {len(rag.snippets)} snippets")
    st.info("🔧 Threading: Single-threaded (macOS optimized)")
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Snippet Selection", "📈 Topic Analysis", "📊 Evolution Analysis"])
    
    with tab1:
        st.header("Snippet Selection")
        st.write("Select snippets for topic analysis - either search for specific topics or use all snippets")
        
        # Selection mode
        selection_mode = st.radio(
            "Selection Mode:",
            ["Use All Snippets", "Search Specific Topic"],
            help="Choose whether to analyze all snippets or search for specific topics first"
        )
        
        if selection_mode == "Use All Snippets":
            st.subheader("📚 Use All Snippets")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"Total available snippets: **{len(rag.snippets)}**")
                
                # Sample size limit
                max_snippets = st.number_input(
                    "Maximum snippets to use", 
                    min_value=100, 
                    max_value=len(rag.snippets), 
                    value=min(3000, len(rag.snippets)),  # Reduced default for better performance
                    help="Limit for performance. Random sampling will be used if needed."
                )
            
            with col2:
                st.subheader("Optional Filters")
                
                # Company filter
                all_companies = list(set([s.ticker for s in rag.snippets]))
                selected_companies = st.multiselect("Filter by Company", all_companies)
                
                # Sentiment filter
                sentiment_filter = st.selectbox("Filter by Sentiment", 
                                              ["All", "opportunity", "neutral", "risk"])
                
                # Year range
                years = [int(s.year) for s in rag.snippets if s.year and str(s.year).isdigit()]
                if years:
                    year_range = st.slider("Year Range", 
                                         min_value=min(years), 
                                         max_value=max(years),
                                         value=(min(years), max(years)))
                else:
                    year_range = None
            
            if st.button("📋 Select All Snippets", type="primary"):
                with st.spinner("Selecting and filtering snippets..."):
                    # Apply filters
                    filtered_snippets = []
                    for snippet in rag.snippets:
                        # Company filter
                        if selected_companies and snippet.ticker not in selected_companies:
                            continue
                            
                        # Sentiment filter
                        if sentiment_filter != "All" and snippet.climate_sentiment != sentiment_filter:
                            continue
                            
                        # Year filter
                        if year_range and snippet.year:
                            try:
                                year = int(snippet.year)
                                if not (year_range[0] <= year <= year_range[1]):
                                    continue
                            except (ValueError, TypeError):
                                continue
                        
                        filtered_snippets.append(snippet)
                    
                    # Apply sample size limit
                    if len(filtered_snippets) > max_snippets:
                        import random
                        random.seed(42)  # For reproducibility
                        filtered_snippets = random.sample(filtered_snippets, max_snippets)
                    
                    # Store in session state
                    st.session_state.selected_snippets = filtered_snippets
                    st.session_state.selection_method = "All snippets"
                    
                    st.success(f"✅ Selected {len(filtered_snippets)} snippets for topic analysis!")
                    st.info("Go to the 'Topic Analysis' tab to analyze these snippets.")
        
        else:  # Search Specific Topic
            st.subheader("🔍 Search Specific Topic")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                search_type = st.radio("Search Type", ["Category", "Custom Query", "Semantic Search"])
                
                if search_type == "Category":
                    category = st.selectbox("Select Investment Category", 
                                          list(rag.investment_categories.keys()))
                elif search_type == "Semantic Search":
                    query = st.text_input("Enter your semantic query")
                    relevance_threshold = st.slider("Relevance Threshold", 
                                                   min_value=0.0, max_value=1.0, 
                                                   value=0.30, step=0.05,
                                                   help="Higher values return more relevant but fewer results")
                else:
                    query = st.text_input("Enter your search query")
            
            with col2:
                st.subheader("Filters")
                
                # Company filter
                all_companies = list(set([s.ticker for s in rag.snippets]))
                selected_companies = st.multiselect("Filter by Company", all_companies)
                
                # Sentiment filter
                sentiment_filter = st.selectbox("Filter by Sentiment", 
                                              ["All", "opportunity", "neutral", "risk"])
                
                # Year range
                years = [int(s.year) for s in rag.snippets if s.year and str(s.year).isdigit()]
                if years:
                    year_range = st.slider("Year Range", 
                                         min_value=min(years), 
                                         max_value=max(years),
                                         value=(min(years), max(years)))
                else:
                    year_range = None
            
            if st.button("🔍 Search & Select", type="primary"):
                with st.spinner("Searching for relevant snippets..."):
                    # Perform search based on type
                    if search_type == "Category":
                        results = rag.search_by_category(category, top_k=None,
                                                       selected_companies=selected_companies if selected_companies else None,
                                                       sentiment_filter=sentiment_filter,
                                                       year_range=year_range)
                    elif search_type == "Semantic Search" and query:
                        results = rag.query_embedding_index(
                            query, top_k=None, relevance_threshold=relevance_threshold,
                            selected_companies=selected_companies if selected_companies else None,
                            sentiment_filter=sentiment_filter, year_range=year_range)
                    elif search_type == "Custom Query" and query:
                        results = rag.search_by_query(query, top_k=None,
                                                    selected_companies=selected_companies if selected_companies else None,
                                                    sentiment_filter=sentiment_filter,
                                                    year_range=year_range)
                    else:
                        results = []
                        st.warning("Please enter a search query.")
                    
                    if results:
                        # Convert results back to snippets
                        selected_snippets = []
                        for result in results:
                            # Find the corresponding snippet
                            for snippet in rag.snippets:
                                if (snippet.text == result['text'] and 
                                    snippet.company == result['company'] and
                                    snippet.ticker == result['ticker']):
                                    selected_snippets.append(snippet)
                                    break
                        
                        # Store in session state
                        st.session_state.selected_snippets = selected_snippets
                        search_term = category if search_type == "Category" else query
                        st.session_state.selection_method = f"Search: {search_term}"
                        
                        st.success(f"✅ Found and selected {len(selected_snippets)} snippets!")
                        st.info("Go to the 'Topic Analysis' tab to analyze these snippets.")
                        
                        # Show preview of results
                        display_results(results[:5])
                        if len(results) > 5:
                            st.info(f"Showing first 5 results. Total: {len(results)} snippets selected.")
                    else:
                        st.warning("No results found. Try adjusting your search terms or filters.")
        
        # Show current selection status
        if st.session_state.selected_snippets:
            st.markdown("---")
            st.subheader("📋 Current Selection")
            st.write(f"**Selected:** {len(st.session_state.selected_snippets)} snippets")
            st.write(f"**Method:** {st.session_state.get('selection_method', 'Unknown')}")
            
            if st.button("🗑️ Clear Selection"):
                st.session_state.selected_snippets = []
                st.session_state.selection_method = ""
                st.success("Selection cleared!")

    with tab2:
        st.header("Topic Analysis")
        
        # Check if snippets are selected
        selected_snippets = get_selected_snippets()
        
        if not selected_snippets:
            st.warning("⚠️ No snippets selected for analysis.")
            st.info("Go to the 'Snippet Selection' tab first to select snippets.")
            return
        
        st.info(f"📊 Ready to analyze {len(selected_snippets)} selected snippets")
        st.write(f"**Selection method:** {st.session_state.get('selection_method', 'Unknown')}")
        
        # Topic analysis parameters
        col1, col2 = st.columns([2, 1])
        
        with col1:
            nr_topics = st.slider("Number of Topics to Find", 
                                 min_value=2, 
                                 max_value=min(15, len(selected_snippets)//20),  # More conservative
                                 value=min(6, len(selected_snippets)//20))  # Lower default
        
        with col2:
            st.subheader("Analysis Info")
            st.metric("Selected Snippets", len(selected_snippets))
            st.metric("Max Topics", min(15, len(selected_snippets)//20))
        
        if st.button("🚀 Run Topic Analysis", type="primary"):
            with st.spinner("Running topic analysis..."):
                try:
                    # Extract texts
                    texts = [snippet.text for snippet in selected_snippets]
                    
                    if len(texts) < nr_topics * 20:  # Need at least 20 docs per topic
                        st.warning(f"Not enough texts ({len(texts)}) for {nr_topics} topics. Try reducing the number of topics or selecting more snippets.")
                        return
                    
                    # Create vectorizer with more aggressive stopword removal
                    custom_stopwords = list(text.ENGLISH_STOP_WORDS.union({
                        "million", "quarter", "company", "business", "group", "share", 
                        "billion", "sales", "revenues", "revenue", "year", "time",
                        "percent", "growth", "market", "increase", "decrease", "good", "well"
                    }))
                    vectorizer_model = CountVectorizer(
                        stop_words=custom_stopwords,
                        max_features=1000,  # Limit features for performance
                        min_df=3,  # Word must appear in at least 3 documents
                        max_df=0.8  # Word must appear in less than 80% of documents
                    )
                    
                    # Simple UMAP configuration for stability
                    umap_model = UMAP(
                        n_neighbors=min(10, len(texts)//5), 
                        n_components=3,  # Reduced dimensions
                        metric='cosine', 
                        n_jobs=1,  # Single-threaded for stability
                        random_state=42,
                        min_dist=0.1,
                        spread=1.0
                    )

                    topic_model = BERTopic(
                        umap_model=umap_model,
                        top_n_words=8,  # Fewer words for cleaner topics
                        nr_topics=nr_topics,
                        calculate_probabilities=False,
                        vectorizer_model=vectorizer_model,
                        verbose=False  # Reduce output
                    )
                    
                    topics, probs = topic_model.fit_transform(texts)
                    
                    # Generate topic names
                    topic_info = topic_model.get_topic_info()
                    
                    st.subheader("🤖 Generating Topic Names...")
                    with st.spinner("Generating meaningful topic names using LLM..."):
                        topic_names = generate_topic_names(
                            topic_model, 
                            topic_info, 
                            st.session_state.get('anthropic_api_key')
                        )
                    
                    # Display topics with LLM-generated names
                    st.subheader("📋 Discovered Topics")
                    topic_display_df = topic_info.copy()
                    topic_display_df['LLM_Name'] = topic_display_df['Topic'].map(
                        lambda x: topic_names.get(x, f"Topic {x}")
                    )
                    st.dataframe(topic_display_df[['Topic', 'LLM_Name', 'Count', 'Name']])

                    # Show Word Clouds for each topic with LLM names
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

                        wc = WordCloud(width=400, height=200, background_color='white').generate(' '.join(word_list))

                        # Use LLM-generated name
                        llm_name = topic_names.get(topic_num, f"Topic {topic_num}")
                        st.markdown(f"**{llm_name}** (Topic {topic_num})")
                        st.caption(f"Top words: {', '.join(word_list[:6])}")
                        
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)

                    # Show topic distribution
                    st.subheader("📊 Topic Distribution")
                    topic_counts = topic_info[topic_info['Topic'] != -1]['Count'].tolist()
                    topic_labels = [topic_names.get(t, f"Topic {t}") for t in topic_info[topic_info['Topic'] != -1]['Topic'].tolist()]
                    
                    fig = px.bar(
                        x=topic_labels, 
                        y=topic_counts,
                        title="Number of Documents per Topic",
                        labels={'x': 'Topic', 'y': 'Document Count'}
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Store results in session state
                    st.session_state.topic_model = topic_model
                    st.session_state.topic_names = topic_names
                    st.session_state.topic_info = topic_info
                    
                    # Export functionality
                    st.markdown("#### Export Results")
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
                    
                    results_df = pd.DataFrame(topic_results)
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Topic Analysis Results as CSV",
                        data=csv,
                        file_name="topic_analysis_results.csv",
                        mime="text/csv"
                    )
                    
                    st.success("✅ Topic analysis completed! Go to 'Evolution Analysis' tab to analyze trends over time.")
                    
                except Exception as e:
                    st.error(f"Error running topic analysis: {str(e)}")
                    st.info("Try reducing the number of topics or selecting fewer snippets.")

    with tab3:
        st.header("📈 Topic Evolution Analysis")
        
        # Check if topic model exists from tab 2
        if 'topic_model' not in st.session_state or 'topic_names' not in st.session_state:
            st.warning("⚠️ Please run Topic Analysis in Tab 2 first to generate topics.")
            st.info("Go to the 'Topic Analysis' tab and click 'Run Topic Analysis'.")
            return
        
        # Market selection for evolution analysis
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🌍 Market Selection")
            show_eu = st.checkbox("Show EU (STOXX 600)", value=True)
            show_us = st.checkbox("Show US (S&P 500)", value=True)
            
            if not show_eu and not show_us:
                st.warning("Please select at least one market to display.")
                return
        
        with col2:
            st.subheader("🎯 Topic Selection")
            # Get available topics (excluding outliers)
            topic_info = st.session_state.topic_info
            available_topics = topic_info[topic_info['Topic'] != -1]['Topic'].tolist()
            topic_names = st.session_state.topic_names
            
            # Create topic options with LLM names
            topic_options = {f"{topic_names.get(t, f'Topic {t}')} (Topic {t})": t for t in available_topics}
            
            selected_topic_display = st.selectbox(
                "Select Topic to Analyze",
                options=list(topic_options.keys()),
                help="Choose a topic to see its evolution over time"
            )
            
            if selected_topic_display:
                selected_topic = topic_options[selected_topic_display]
        
        # Time period selection
        st.subheader("📅 Time Period")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            time_granularity = st.radio(
                "Time Granularity",
                ["Quarterly", "Yearly"],
                help="Choose how to group the data over time"
            )
        
        with col2:
            # Get available years from selected snippets
            selected_snippets = get_selected_snippets()
            all_years = [int(s.year) for s in selected_snippets if s.year and str(s.year).isdigit()]
            if all_years:
                min_year, max_year = min(all_years), max(all_years)
                selected_years = st.slider(
                    "Year Range",
                    min_value=min_year,
                    max_value=max_year,
                    value=(min_year, max_year),
                    help="Select the time range for analysis"
                )
            else:
                st.error("No valid years found in selected snippets.")
                return
        
        # Analysis button
        if st.button("🔍 Analyze Topic Evolution", type="primary"):
            with st.spinner("Analyzing topic evolution over time..."):
                try:
                    # Get the topic model and analyze evolution
                    topic_model = st.session_state.topic_model
                    
                    # Analyze evolution for the selected topic
                    evolution_data = analyze_topic_evolution_simple(
                        selected_snippets, topic_model, selected_topic, 
                        selected_years, time_granularity, topic_names
                    )
                    
                    if not evolution_data:
                        st.warning("No data found for the selected topic and time period.")
                    else:
                        # Display evolution charts
                        display_evolution_charts_simple(evolution_data, selected_topic_display, time_granularity)
                        
                        # Display detailed insights
                        display_evolution_insights_simple(evolution_data, selected_topic_display, time_granularity)
                        
                except Exception as e:
                    st.error(f"Error analyzing topic evolution: {str(e)}")
                    st.info("Please ensure you have run the topic analysis first and selected valid parameters.")


def analyze_topic_evolution_simple(selected_snippets, topic_model, selected_topic, year_range, time_granularity, topic_names):
    """Analyze how a specific topic evolves over time for selected snippets."""
    
    # Get the topic words to identify relevant snippets
    topic_words = topic_model.get_topic(selected_topic)
    if not topic_words:
        return []
    
    # Extract keywords from the topic
    keywords = [word for word, _ in topic_words[:10]]  # Top 10 words
    
    # Helper function to check if snippet matches topic
    def snippet_matches_topic(snippet_text, keywords):
        text_lower = snippet_text.lower()
        return any(keyword.lower() in text_lower for keyword in keywords)
    
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


def display_evolution_charts_simple(evolution_data, topic_name, time_granularity):
    """Display evolution charts for the selected topic."""
    
    if not evolution_data:
        st.warning("No data available for visualization.")
        return
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(evolution_data)
    
    st.subheader(f"📊 Evolution of {topic_name}")
    
    # 1. Topic mention frequency over time
    st.markdown("#### Topic Mention Frequency")
    fig = px.line(
        df, 
        x='period', 
        y='count', 
        title=f"Topic Mentions Over Time ({time_granularity})",
        labels={'count': 'Number of Mentions', 'period': 'Time Period'},
        markers=True
    )
    fig.update_layout(
        xaxis_title="Time Period",
        yaxis_title="Number of Mentions",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 2. Company diversity over time
    st.markdown("#### Company Diversity")
    fig2 = px.line(
        df, 
        x='period', 
        y='companies', 
        title=f"Number of Companies Discussing Topic ({time_granularity})",
        labels={'companies': 'Number of Companies', 'period': 'Time Period'},
        markers=True
    )
    fig2.update_layout(
        xaxis_title="Time Period",
        yaxis_title="Number of Companies",
        hovermode='x unified'
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # 3. Sentiment evolution
    st.markdown("#### Sentiment Evolution")
    
    # Prepare sentiment data
    sentiment_data = []
    for _, row in df.iterrows():
        for sentiment in ['opportunity', 'neutral', 'risk']:
            sentiment_data.append({
                'period': row['period'],
                'sentiment': sentiment.title(),
                'count': row[f'sentiment_{sentiment}']
            })
    
    sentiment_df = pd.DataFrame(sentiment_data)
    
    fig3 = px.bar(
        sentiment_df, 
        x='period', 
        y='count', 
        color='sentiment',
        title="Sentiment Evolution Over Time",
        color_discrete_map={'Opportunity': 'green', 'Neutral': 'yellow', 'Risk': 'red'}
    )
    fig3.update_layout(xaxis_title="Time Period", yaxis_title="Count")
    st.plotly_chart(fig3, use_container_width=True)


def display_evolution_insights_simple(evolution_data, topic_name, time_granularity):
    """Display insights and key metrics about topic evolution."""
    
    if not evolution_data:
        return
    
    df = pd.DataFrame(evolution_data)
    
    st.subheader(f"🔍 Key Insights for {topic_name}")
    
    # Calculate key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_mentions = df['count'].sum()
        st.metric("Total Mentions", total_mentions)
    
    with col2:
        total_companies = df['companies'].sum()
        st.metric("Total Companies", total_companies)
    
    with col3:
        avg_mentions_per_period = df['count'].mean()
        st.metric("Avg Mentions/Period", f"{avg_mentions_per_period:.1f}")
    
    with col4:
        # Calculate trend (simple linear trend)
        if len(df) > 1:
            periods = list(range(len(df)))
            mentions = df['count'].values
            if len(mentions) > 1:
                trend = np.polyfit(periods, mentions, 1)[0]
                trend_emoji = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
                st.metric("Trend", f"{trend_emoji} {trend:.1f}")
            else:
                st.metric("Trend", "N/A")
        else:
            st.metric("Trend", "N/A")
    
    # Period-wise breakdown
    st.markdown("#### Period-wise Breakdown")
    period_summary = df.copy()
    period_summary.columns = ['Period', 'Mentions', 'Companies', 'Opportunity', 'Neutral', 'Risk']
    st.dataframe(period_summary, use_container_width=True)
    
    # Download option
    st.markdown("#### Export Data")
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Evolution Data as CSV",
        data=csv,
        file_name=f"topic_evolution_{topic_name.replace(' ', '_')}.csv",
        mime="text/csv"
    )

def display_results(results):
    """Display search results in a nice format"""
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

if __name__ == "__main__":
    main()