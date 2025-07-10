# interface.py - Fixed version with threading issues resolved and LLM topic naming
import os
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

# Set environment variables BEFORE importing numba-dependent libraries
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'
os.environ['NUMBA_NUM_THREADS'] = '1'

# Now import numba and set config
from numba import config
config.THREADING_LAYER = 'workqueue'

# Import UMAP and BERTopic after setting threading configuration
from umap import UMAP
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

# Page config
st.set_page_config(
    page_title="Green Investment Analyzer",
    page_icon="🌱",
    layout="wide"
)

def generate_topic_names(topic_model, topic_info: pd.DataFrame, api_key: str = None) -> Dict[int, str]:
    """
    Generate meaningful topic names using Anthropic Claude based on top words for each topic.
    
    Args:
        topic_model: BERTopic model
        topic_info: DataFrame with topic information
        api_key: Anthropic API key (optional, uses environment variable if not provided)
    
    Returns:
        Dictionary mapping topic numbers to generated names
    """
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
            prompt = f"""
{HUMAN_PROMPT} Based on these keywords from a topic model analysis of green investment and climate-related earnings call transcripts, suggest a concise and descriptive topic name (2-4 words):

Keywords: {words_str}

Provide only the topic name, no explanation. If the words are incoherent, return only the topic number followed by 'Incoherent'.
{AI_PROMPT}
            """.strip()

            try:
                response = client.messages.create(
                model="claude-3-5-sonnet-20240620",  # Use the latest correct Claude 3.5 Sonnet release name
                max_tokens=600,
                temperature=0.2,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
                topic_name = response.content[0].text
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

def main():
    st.title("🌱 Green Investment Analyzer")
    st.subheader("Extract climate investment insights from earnings calls")
    
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
        api_key = st.text_input("OpenAI API Key (optional)", type="password", 
                               help="Enter your OpenAI API key for automatic topic naming, or set OPENAI_API_KEY environment variable")
        if api_key:
            st.session_state.openai_api_key = api_key

    # Initialize session state keys if not present
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'current_market' not in st.session_state:
        st.session_state.current_market = None
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = None

    if not st.session_state.data_loaded:
        st.warning("Please select and load market data using the sidebar.")
        return
    
    rag = st.session_state.rag_system
    
    # Display current market info
    st.info(f"📊 Currently analyzing: {st.session_state.current_market} market with {len(rag.snippets)} snippets")
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "📈 Subtopic identification", "📊 Company Comparison"])
    
    with tab1:
        st.header("Topic Search")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_type = st.radio("Search Type", ["Category", "Custom Query", "Semantic Search"])
            
            if search_type == "Category":
                category = st.selectbox("Select Investment Category", 
                                      list(rag.investment_categories.keys()))
            elif search_type == "Semantic Search":
                query = st.text_input("Enter your semantic query")
                # Add relevance threshold slider for semantic search
                relevance_threshold = st.slider("Relevance Threshold", 
                                               min_value=0.0, max_value=1.0, 
                                               value=0.45, step=0.05,
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
        
        # Search button and results
        if search_type == "Category":
            if st.button("Search by Category"):
                # Use all snippets, no limit on search
                results = rag.search_by_category(category, top_k=None)
                filtered_results = filter_results(
                    results, 
                    selected_companies=selected_companies if selected_companies else None,
                    sentiment_filter=sentiment_filter,
                    year_range=year_range
                )
                # Display first 10 results but store all filtered results
                display_results(filtered_results[:10])
                st.session_state.filtered_results = filtered_results
                st.info(f"Found {len(filtered_results)} total results (showing first 10)")

        elif search_type == "Semantic Search":
            if st.button("Semantic Search") and query:
                # Use all snippets with relevance threshold
                results = rag.query_embedding_index(
                    query, 
                    top_k=None,  # Get all results above threshold
                    relevance_threshold=relevance_threshold
                )
                filtered_results = filter_results(
                    results,
                    selected_companies=selected_companies if selected_companies else None,
                    sentiment_filter=sentiment_filter,
                    year_range=year_range
                )
                # Display first 10 results but store all filtered results
                display_results(filtered_results[:10])
                st.session_state.filtered_results = filtered_results
                st.info(f"Found {len(filtered_results)} total results (showing first 10)")

        else:
            if st.button("Search") and query:
                # Use all snippets, no limit on search
                results = rag.search_by_query(query, top_k=None)
                filtered_results = filter_results(
                    results, 
                    selected_companies=selected_companies if selected_companies else None,
                    sentiment_filter=sentiment_filter,
                    year_range=year_range
                )
                # Display first 10 results but store all filtered results
                display_results(filtered_results[:10])
                st.session_state.filtered_results = filtered_results
                st.info(f"Found {len(filtered_results)} total results (showing first 10)")

    with tab2:
        st.header("Subtopic Identification")

        # Select the number of topics
        nr_topics = st.slider("Number of Topics to Find", min_value=2, max_value=15, value=5)

        texts = [res['text'] for res in st.session_state.get('filtered_results', [])]
        if not texts:
            st.warning("No filtered results found. Using the first 5000 snippets.")
            texts = [s.text for s in rag.snippets][:5000]

        st.write(f"Analyzing {len(texts)} snippets...")

        # Create vectorizer with stopword removal
        custom_stopwords = list(text.ENGLISH_STOP_WORDS.union({"million", "quarter", "company", "business", "group", "share", "billion", "sales", "revenues", "revenue"}))
        vectorizer_model = CountVectorizer(stop_words=custom_stopwords)

        # Train BERTopic with fixed threading
        if st.button("Run Topic Analysis"):
            with st.spinner("Running BERTopic..."):
                try:
                    # Force single-threaded execution for UMAP
                    umap_model = UMAP(
                        n_neighbors=15, 
                        n_components=5, 
                        metric='cosine', 
                        n_jobs=1,  # Single-threaded
                        random_state=42  # For reproducibility
                    )

                    topic_model = BERTopic(
                        umap_model=umap_model,
                        top_n_words=10,
                        nr_topics=nr_topics,
                        calculate_probabilities=False,
                        vectorizer_model=vectorizer_model,
                        verbose=True
                    )
                    
                    topics, probs = topic_model.fit_transform(texts)

                    # Show topics and their sizes
                    topic_info = topic_model.get_topic_info()
                    
                    # Generate LLM-based topic names
                    st.subheader("🤖 Generating Topic Names...")
                    with st.spinner("Generating meaningful topic names using LLM..."):
                        topic_names = generate_topic_names(
                            topic_model, 
                            topic_info, 
                            st.session_state.get('openai_api_key')
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

                        # Guard against empty topics
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
                        st.caption(f"Top words: {', '.join(word_list[:8])}")
                        
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
                    
                except Exception as e:
                    st.error(f"Error running BERTopic: {str(e)}")
                    st.info("Try installing Intel TBB with: pip install tbb")
                    st.info("Or check your OpenAI API key if using LLM topic naming.")
        
    with tab3:
        st.header("📈 Subtopic Evolution Analysis")
        
        # Check if topic model exists from tab 2
        if 'topic_model' not in st.session_state or 'topic_names' not in st.session_state:
            st.warning("⚠️ Please run Topic Analysis in the 'Subtopic identification' tab first to generate topics.")
            st.info("Go to tab 2 and click 'Run Topic Analysis' to identify subtopics before analyzing their evolution.")
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
            # Get available years from all snippets
            all_years = [int(s.year) for s in rag.snippets if s.year and str(s.year).isdigit()]
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
                st.error("No valid years found in the data.")
                return
        
        # Analysis button
        if st.button("🔍 Analyze Topic Evolution", type="primary"):
            col1, col2 = st.columns(2)
        
            with col1:
                analyze_single = st.button("🔍 Analyze Single Topic Evolution", type="primary")
            
            with col2:
                analyze_all = st.button("📊 Analyze All Topics Evolution", type="secondary")
            
            # Single topic analysis
            if analyze_single and selected_topic_display:
                with st.spinner("Analyzing single topic evolution over time..."):
                    try:
                        # Get the topic model and analyze evolution
                        topic_model = st.session_state.topic_model
                        
                        # Load market-specific data if needed
                        evolution_data = analyze_topic_evolution(
                            rag, topic_model, selected_topic, 
                            show_eu, show_us, selected_years, time_granularity
                        )
                        
                        if not evolution_data:
                            st.warning("No data found for the selected topic and time period.")
                        else:
                            # Display evolution charts
                            display_evolution_charts(evolution_data, selected_topic_display, time_granularity)
                            
                            # Display detailed insights
                            display_evolution_insights(evolution_data, selected_topic_display, time_granularity)
                            
                    except Exception as e:
                        st.error(f"Error analyzing topic evolution: {str(e)}")
                        st.info("Please ensure you have run the topic analysis first and selected valid parameters.")
            
            # All topics analysis
            if analyze_all:
                with st.spinner("Analyzing all topics evolution over time..."):
                    try:
                        # Get the topic model and analyze all topics evolution
                        topic_model = st.session_state.topic_model
                        topic_names = st.session_state.topic_names
                        
                        # Analyze all topics evolution
                        all_topics_data, valid_topics = analyze_all_topics_evolution(
                            rag, topic_model, topic_names,
                            show_eu, show_us, selected_years, time_granularity
                        )
                        
                        if not all_topics_data:
                            st.warning("No data found for the selected time period and markets.")
                        else:
                            # Display stacked bar chart for all topics
                            display_all_topics_stacked_chart(
                                all_topics_data, valid_topics, topic_names,
                                show_eu, show_us, time_granularity
                            )
                            
                    except Exception as e:
                        st.error(f"Error analyzing all topics evolution: {str(e)}")
                        st.info("Please ensure you have run the topic analysis first and selected valid parameters.")
                        
# Add these functions to your interface.py file, right after the display_results function
# and before the if __name__ == "__main__": line
def analyze_all_topics_evolution(rag, topic_model, topic_names, show_eu, show_us, year_range, time_granularity):
    """Analyze how all topics evolve over time across markets."""
    
    # Get all valid topics (excluding outliers)
    topic_info = topic_model.get_topic_info()
    valid_topics = topic_info[topic_info['Topic'] != -1]['Topic'].tolist()
    
    if not valid_topics:
        return []
    
    # Get topic words for each topic
    topic_keywords = {}
    for topic_num in valid_topics:
        topic_words = topic_model.get_topic(topic_num)
        if topic_words:
            topic_keywords[topic_num] = [word for word, _ in topic_words[:10]]
    
    # Helper function to check if snippet matches any topic
    def get_snippet_topics(snippet_text, topic_keywords):
        """Return list of topics that match this snippet."""
        text_lower = snippet_text.lower()
        matching_topics = []
        
        for topic_num, keywords in topic_keywords.items():
            if any(keyword.lower() in text_lower for keyword in keywords):
                matching_topics.append(topic_num)
        
        return matching_topics
    
    # For combined data, we need to load both markets separately
    if st.session_state.current_market == "Combined":
        # Load EU data
        try:
            eu_rag = load_eu_rag()
            eu_snippets = eu_rag.snippets
        except:
            eu_snippets = []
        
        # Load US data
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
        current_market = st.session_state.current_market
        if (current_market == "EU" and not show_eu) or (current_market == "US" and not show_us):
            return []
        
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


def display_all_topics_stacked_chart(evolution_data, valid_topics, topic_names, show_eu, show_us, time_granularity):
    """Display stacked bar chart for all topics evolution."""
    
    if not evolution_data:
        st.warning("No data available for all topics visualization.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(evolution_data)
    
    st.subheader("📊 All Topics Evolution - Stacked Bar Chart")
    
    # Create separate charts for EU and US if both are selected
    if show_eu and show_us:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### EU Market")
            eu_columns = [col for col in df.columns if col.startswith('EU_') and col != 'period']
            if eu_columns:
                # Prepare data for EU stacked bar chart
                eu_data = df[['period'] + eu_columns].set_index('period')
                # Remove 'EU_' prefix from column names for cleaner display
                eu_data.columns = [col.replace('EU_', '') for col in eu_data.columns]
                
                # Create stacked bar chart
                fig_eu = px.bar(
                    eu_data.reset_index(),
                    x='period',
                    y=eu_data.columns.tolist(),
                    title=f"EU Topics Evolution ({time_granularity})",
                    labels={'value': 'Number of Mentions', 'period': 'Time Period'},
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_eu.update_layout(
                    xaxis_title="Time Period",
                    yaxis_title="Number of Mentions",
                    legend_title="Topics",
                    hovermode='x unified'
                )
                st.plotly_chart(fig_eu, use_container_width=True)
        
        with col2:
            st.markdown("#### US Market")
            us_columns = [col for col in df.columns if col.startswith('US_') and col != 'period']
            if us_columns:
                # Prepare data for US stacked bar chart
                us_data = df[['period'] + us_columns].set_index('period')
                # Remove 'US_' prefix from column names for cleaner display
                us_data.columns = [col.replace('US_', '') for col in us_data.columns]
                
                # Create stacked bar chart
                fig_us = px.bar(
                    us_data.reset_index(),
                    x='period',
                    y=us_data.columns.tolist(),
                    title=f"US Topics Evolution ({time_granularity})",
                    labels={'value': 'Number of Mentions', 'period': 'Time Period'},
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_us.update_layout(
                    xaxis_title="Time Period",
                    yaxis_title="Number of Mentions",
                    legend_title="Topics",
                    hovermode='x unified'
                )
                st.plotly_chart(fig_us, use_container_width=True)
    
    else:
        # Single market view
        market_prefix = "EU_" if show_eu else "US_"
        market_name = "EU" if show_eu else "US"
        
        market_columns = [col for col in df.columns if col.startswith(market_prefix) and col != 'period']
        if market_columns:
            # Prepare data for stacked bar chart
            market_data = df[['period'] + market_columns].set_index('period')
            # Remove market prefix from column names for cleaner display
            market_data.columns = [col.replace(market_prefix, '') for col in market_data.columns]
            
            # Create stacked bar chart
            fig = px.bar(
                market_data.reset_index(),
                x='period',
                y=market_data.columns.tolist(),
                title=f"{market_name} Topics Evolution ({time_granularity})",
                labels={'value': 'Number of Mentions', 'period': 'Time Period'},
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(
                xaxis_title="Time Period",
                yaxis_title="Number of Mentions",
                legend_title="Topics",
                hovermode='x unified',
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Add summary statistics
    st.subheader("📋 Topics Summary")
    
    # Calculate total mentions per topic across all periods
    topic_totals = {}
    for topic_num in valid_topics:
        topic_name = topic_names.get(topic_num, f"Topic {topic_num}")
        total = 0
        
        if show_eu:
            eu_col = f"EU_{topic_name}"
            if eu_col in df.columns:
                total += df[eu_col].sum()
        
        if show_us:
            us_col = f"US_{topic_name}"
            if us_col in df.columns:
                total += df[us_col].sum()
        
        topic_totals[topic_name] = total
    
    # Display as a simple table
    summary_df = pd.DataFrame(list(topic_totals.items()), columns=['Topic', 'Total Mentions'])
    summary_df = summary_df.sort_values('Total Mentions', ascending=False)
    st.dataframe(summary_df, use_container_width=True)
    
    # Download option for all topics data
    st.markdown("#### Export All Topics Data")
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download All Topics Evolution Data as CSV",
        data=csv,
        file_name=f"all_topics_evolution_{time_granularity.lower()}.csv",
        mime="text/csv"
    )

def analyze_topic_evolution(rag, topic_model, selected_topic, show_eu, show_us, year_range, time_granularity):
    """Analyze how a specific topic evolves over time across markets."""
    
    # Get the topic words to identify relevant snippets
    topic_words = topic_model.get_topic(selected_topic)
    if not topic_words:
        return []
    
    # Extract keywords from the topic
    keywords = [word for word, _ in topic_words[:10]]  # Top 10 words
    
    evolution_data = []
    
    # Helper function to check if snippet matches topic
    def snippet_matches_topic(snippet_text, keywords):
        text_lower = snippet_text.lower()
        return any(keyword.lower() in text_lower for keyword in keywords)
    
    # For combined data, we need to load both markets separately
    if st.session_state.current_market == "Combined":
        # Load EU data
        try:
            eu_rag = load_eu_rag()
            eu_snippets = eu_rag.snippets
        except:
            eu_snippets = []
        
        # Load US data
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
                        if snippet_matches_topic(snippet.text, keywords):
                            market_snippets.append((snippet, 'EU'))
        
        if show_us:
            for snippet in us_snippets:
                if snippet.year and str(snippet.year).isdigit():
                    year = int(snippet.year)
                    if year_range[0] <= year <= year_range[1]:
                        if snippet_matches_topic(snippet.text, keywords):
                            market_snippets.append((snippet, 'US'))
    
    else:
        # Single market data
        current_market = st.session_state.current_market
        if (current_market == "EU" and not show_eu) or (current_market == "US" and not show_us):
            return []
        
        market_snippets = []
        for snippet in rag.snippets:
            if snippet.year and str(snippet.year).isdigit():
                year = int(snippet.year)
                if year_range[0] <= year <= year_range[1]:
                    if snippet_matches_topic(snippet.text, keywords):
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
                'EU': {'count': 0, 'companies': set(), 'sentiment': {'opportunity': 0, 'neutral': 0, 'risk': 0}},
                'US': {'count': 0, 'companies': set(), 'sentiment': {'opportunity': 0, 'neutral': 0, 'risk': 0}}
            }
        
        time_groups[period][market]['count'] += 1
        time_groups[period][market]['companies'].add(snippet.ticker)
        if snippet.climate_sentiment:
            time_groups[period][market]['sentiment'][snippet.climate_sentiment] += 1
    
    # Convert to list format for visualization
    for period in sorted(time_groups.keys()):
        for market in ['EU', 'US']:
            if (market == 'EU' and show_eu) or (market == 'US' and show_us):
                data = time_groups[period][market]
                evolution_data.append({
                    'period': period,
                    'market': market,
                    'count': data['count'],
                    'companies': len(data['companies']),
                    'sentiment_opportunity': data['sentiment']['opportunity'],
                    'sentiment_neutral': data['sentiment']['neutral'],
                    'sentiment_risk': data['sentiment']['risk']
                })
    
    return evolution_data

def display_evolution_charts(evolution_data, topic_name, time_granularity):
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
        color='market',
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
        color='market',
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
    col1, col2 = st.columns(2)
    
    # Prepare sentiment data
    sentiment_data = []
    for _, row in df.iterrows():
        for sentiment in ['opportunity', 'neutral', 'risk']:
            sentiment_data.append({
                'period': row['period'],
                'market': row['market'],
                'sentiment': sentiment.title(),
                'count': row[f'sentiment_{sentiment}']
            })
    
    sentiment_df = pd.DataFrame(sentiment_data)
    
    with col1:
        if len(df[df['market'] == 'EU']) > 0:
            eu_sentiment = sentiment_df[sentiment_df['market'] == 'EU']
            fig3 = px.bar(
                eu_sentiment, 
                x='period', 
                y='count', 
                color='sentiment',
                title="EU Sentiment Evolution",
                color_discrete_map={'Opportunity': 'green', 'Neutral': 'yellow', 'Risk': 'red'}
            )
            fig3.update_layout(xaxis_title="Time Period", yaxis_title="Count")
            st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        if len(df[df['market'] == 'US']) > 0:
            us_sentiment = sentiment_df[sentiment_df['market'] == 'US']
            fig4 = px.bar(
                us_sentiment, 
                x='period', 
                y='count', 
                color='sentiment',
                title="US Sentiment Evolution",
                color_discrete_map={'Opportunity': 'green', 'Neutral': 'yellow', 'Risk': 'red'}
            )
            fig4.update_layout(xaxis_title="Time Period", yaxis_title="Count")
            st.plotly_chart(fig4, use_container_width=True)

def display_evolution_insights(evolution_data, topic_name, time_granularity):
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
            periods = list(range(len(df['period'].unique())))
            mentions_by_period = df.groupby('period')['count'].sum().values
            if len(mentions_by_period) > 1:
                trend = np.polyfit(periods, mentions_by_period, 1)[0]
                trend_emoji = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
                st.metric("Trend", f"{trend_emoji} {trend:.1f}")
            else:
                st.metric("Trend", "N/A")
        else:
            st.metric("Trend", "N/A")
    
    # Market comparison
    if len(df['market'].unique()) > 1:
        st.markdown("#### Market Comparison")
        market_summary = df.groupby('market').agg({
            'count': ['sum', 'mean'],
            'companies': 'sum'
        }).round(2)
        
        market_summary.columns = ['Total Mentions', 'Avg Mentions/Period', 'Total Companies']
        st.dataframe(market_summary, use_container_width=True)
    
    # Period-wise breakdown
    st.markdown("#### Period-wise Breakdown")
    period_summary = df.groupby('period').agg({
        'count': 'sum',
        'companies': 'sum',
        'sentiment_opportunity': 'sum',
        'sentiment_neutral': 'sum',
        'sentiment_risk': 'sum'
    }).round(2)
    
    period_summary.columns = ['Total Mentions', 'Companies', 'Opportunity', 'Neutral', 'Risk']
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
    
    st.subheader(f"Top {len(results)} results")
    
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