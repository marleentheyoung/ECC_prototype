# ui_components.py - UI component functions for different tabs
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
import pandas as pd
import random
from datetime import datetime, date, timedelta  # <-- Make sure datetime is imported

from src.config import APP_CONFIG
from src.utils import display_results, get_selected_snippets
from src.data_loaders import load_market_data
# from topic_analysis import run_topic_analysis, display_topic_results, create_topic_results_dataframe
from src.topic_search import (
    perform_topic_search, validate_topic_relevance_with_llm, 
    analyze_topic_distribution, visualize_topic_comparison, export_topic_results
)
# At the top of ui_components.py
from src.simplified_snippet_selection import analyze_snippets_evolution
from src.adaptive_threshold_validation import AdaptiveThresholdValidator, display_threshold_search_results
from src.evolution_analysis import analyze_topic_evolution_simple
from src.visualization import (
    display_evolution_charts_simple, 
    display_evolution_insights_simple
)

def render_manual_topic_id_tab_with_adaptive_validation(rag):
    """Enhanced Manual Topic Identification tab with adaptive threshold validation."""
    st.header("📝 Manual Topic Identification")
    st.write("Define custom topics using semantic search with adaptive threshold validation")
    
    # Topic management interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("➕ Add New Topic")
        
        # Validation method selection
        validation_method = st.radio(
            "Validation Method:",
            ["Standard Validation", "Adaptive Threshold Search"],
            help="Choose between fixed threshold or adaptive threshold optimization"
        )
        
        # New topic form
        with st.form("add_topic_form"):
            new_topic_name = st.text_input(
                "Topic Name", 
                placeholder="e.g., Paris Agreement, EU ETS, Carbon Credits"
            )
            new_topic_query = st.text_area(
                "Search Query", 
                placeholder="Enter keywords and phrases to search for this topic",
                help="Use descriptive terms that would appear in earnings calls when discussing this topic"
            )
            
            if validation_method == "Standard Validation":
                # Standard validation options
                relevance_threshold = st.slider(
                    "Relevance Threshold", 
                    min_value=0.1, 
                    max_value=0.8, 
                    value=0.30, 
                    step=0.05
                )
                
                use_llm_validation = st.checkbox("Use LLM Validation", value=True)
                
            else:
                # Adaptive validation options
                st.subheader("🎯 Adaptive Search Settings")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    initial_threshold = st.slider(
                        "Starting Threshold", 
                        min_value=0.1, 
                        max_value=0.6, 
                        value=0.30, 
                        step=0.05,
                        help="Initial threshold to start the adaptive search"
                    )
                
                with col_b:
                    quality_threshold = st.slider(
                        "Quality Threshold", 
                        min_value=0.1, 
                        max_value=0.5, 
                        value=0.25, 
                        step=0.05,
                        help="Maximum % of irrelevant snippets allowed (25% = 5 out of 20)"
                    )
                
                st.info("💡 The system will automatically find the optimal threshold where ≤25% of boundary snippets are irrelevant")
            
            submitted = st.form_submit_button("🔍 Search & Add Topic", type="primary")
            
        # Handle form submission outside the form
        if submitted and new_topic_name and new_topic_query:
        # Handle form submission outside the form
            if validation_method == "Standard Validation":
                add_manual_topic_standard(rag, new_topic_name, new_topic_query, 
                                        relevance_threshold, use_llm_validation)
            else:
                add_manual_topic_adaptive(rag, new_topic_name, new_topic_query, 
                                        initial_threshold, quality_threshold)
        
    with col2:
        st.subheader("📋 Current Topics")
        manual_topics = st.session_state.get('manual_topics', {})
        
        if manual_topics:
            for topic_name, topic_data in manual_topics.items():
                with st.expander(f"{topic_name} ({len(topic_data['snippets'])} snippets)"):
                    st.write(f"**Query:** {topic_data['query']}")
                    
                    if 'adaptive_results' in topic_data:
                        # Show adaptive validation results
                        st.write(f"**Method:** Adaptive (Final threshold: {topic_data['adaptive_results']['final_threshold']:.2f})")
                        st.write(f"**API calls used:** {topic_data['adaptive_results']['total_api_calls']}")
                    else:
                        # Show standard validation results
                        st.write(f"**Threshold:** {topic_data.get('threshold', 'N/A')}")
                        st.write(f"**Method:** Standard")
                    
                    st.write(f"**Validated:** {'Yes' if topic_data.get('validated', False) else 'No'}")
                    
                    if st.button(f"🗑️ Remove {topic_name}", key=f"remove_{topic_name}"):
                        remove_manual_topic(topic_name)
                        st.rerun()
        else:
            st.info("No custom topics defined yet.")
    
    # Display current manual topics results
    if st.session_state.get('manual_topics'):
        st.markdown("---")
        display_manual_topics_overview()

def add_manual_topic_standard(rag, topic_name, query, threshold, use_validation):
    """Add a manual topic using standard validation method."""
    with st.spinner(f"Searching for topic: {topic_name}..."):
        # Perform semantic search
        from src.topic_search import perform_topic_search, validate_topic_relevance_with_llm
        
        search_results = perform_topic_search(rag, {topic_name: query}, threshold)
        snippets = search_results.get(topic_name, [])
        
        # LLM validation if requested
        if use_validation and st.session_state.get('anthropic_api_key') and snippets:
            st.info("🤖 Validating with LLM...")
            snippets = validate_topic_relevance_with_llm(
                snippets, topic_name, query, st.session_state.anthropic_api_key
            )
        
        # Store in session state
        if 'manual_topics' not in st.session_state:
            st.session_state.manual_topics = {}
        
        st.session_state.manual_topics[topic_name] = {
            'query': query,
            'threshold': threshold,
            'validated': use_validation,
            'snippets': snippets,
            'method': 'standard'
        }
        
        st.success(f"✅ Added topic '{topic_name}' with {len(snippets)} snippets!")

def add_manual_topic_adaptive(rag, topic_name, query, initial_threshold, quality_threshold):
    """Add a manual topic using adaptive threshold validation."""
    if not st.session_state.get('anthropic_api_key'):
        st.error("❌ Anthropic API key required for adaptive validation!")
        return
    
    st.info("🚀 Starting adaptive threshold search...")
    
    # Initialize validator
    validator = AdaptiveThresholdValidator(st.session_state.anthropic_api_key)
    validator.irrelevant_threshold = quality_threshold
    
    # Run adaptive search
    try:
        validation_result = validator.adaptive_threshold_search(
            rag, topic_name, query, initial_threshold
        )
        
        if validation_result:
            # Store in session state
            if 'manual_topics' not in st.session_state:
                st.session_state.manual_topics = {}
            
            st.session_state.manual_topics[topic_name] = {
                'query': query,
                'threshold': validation_result['final_threshold'],
                'validated': True,
                'snippets': validation_result['validated_snippets'],
                'method': 'adaptive',
                'adaptive_results': validation_result
            }
            
            # Display results
            display_threshold_search_results(validation_result)
            
            st.success(f"✅ Added topic '{topic_name}' with optimized threshold {validation_result['final_threshold']:.2f} and {len(validation_result['validated_snippets'])} high-quality snippets!")
        
    except Exception as e:
        st.error(f"❌ Error during adaptive search: {str(e)}")

# Add this to your main interface.py file to replace the existing manual topic tab
def render_enhanced_manual_topic_tab(rag):
    """Main function to call from interface.py"""
    render_manual_topic_id_tab_with_adaptive_validation(rag)

def render_sidebar():
    """Render the sidebar with market selection and API key input."""
    with st.sidebar:
        st.header("🌍 Market Selection")
        
        market_option = st.selectbox(
            "Select Market(s)",
            ["EU (STOXX 600)", "US (S&P 500)", "Combined (EU + US)", "Full Index (EU + US)"],
            index=0
        )
        
        if st.button("Load Market Data"):
            with st.spinner(f"Loading {market_option} data..."):
                rag, market_key, success = load_market_data(market_option)
                
                if success and rag:
                    st.session_state.rag_system = rag
                    st.session_state.current_market = market_key
                    st.session_state.data_loaded = True
                    st.success(f"Loaded {len(rag.snippets)} snippets from {market_option}!")
                    
                    # Show loading method info
                    if market_option == "Full Index (EU + US)":
                        st.info("✅ Using optimized full index files for better performance")
                    elif market_option == "Combined (EU + US)":
                        st.info("⚠️ Using fallback method - consider using 'Full Index' option if available")
                else:
                    st.session_state.data_loaded = False

        # API Key input for LLM topic naming
        st.header("🤖 LLM Settings")
        api_key = st.text_input("Anthropic API Key (optional)", type="password", 
                               help="Enter your Anthropic API key for automatic topic naming and validation")
        if api_key:
            st.session_state.anthropic_api_key = api_key

def render_snippet_selection_tab(rag):
    """Render the snippet selection tab."""
    st.header("Snippet Selection")
    st.write("Select snippets for topic analysis - either search for specific topics or use all snippets")
    
    # Selection mode
    selection_mode = st.radio(
        "Selection Mode:",
        ["Use All Snippets", "Search Specific Topic"],
        help="Choose whether to analyze all snippets or search for specific topics first"
    )
    
    if selection_mode == "Use All Snippets":
        render_all_snippets_selection(rag)
    else:
        render_search_selection(rag)
    
    # Show current selection status
    render_selection_status()

def render_topic_search_tab(rag):
    """Render the topic search tab for predefined investment categories."""
    st.header("🎯 Topic Search")
    st.write("Search for specific investment topics and compare their presence in earnings calls")
    
    # Category selection
    st.subheader("📋 Investment Categories")
    available_categories = list(rag.investment_categories.keys())
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_categories = st.multiselect(
            "Select categories to search",
            available_categories,
            default=available_categories[:3] if len(available_categories) >= 3 else available_categories,
            help="Choose which investment categories to analyze"
        )
    
    with col2:
        st.subheader("Search Settings")
        relevance_threshold = st.slider(
            "Relevance Threshold", 
            min_value=0.1, 
            max_value=0.8, 
            value=0.30, 
            step=0.05,
            help="Higher values return more relevant but fewer results"
        )
        
        use_llm_validation = st.checkbox(
            "Use LLM Validation", 
            value=True,
            help="Use Claude to validate search results"
        )
    
    # Display selected categories and their keywords
    if selected_categories:
        st.subheader("📝 Selected Categories & Keywords")
        for category in selected_categories:
            keywords = rag.investment_categories[category]
            st.write(f"**{category}:** {', '.join(keywords)}")
    
    # Search execution
    if st.button("🔍 Search Topics", type="primary", disabled=not selected_categories):
        if not selected_categories:
            st.warning("Please select at least one category to search.")
            return
        
        with st.spinner("Searching for topics..."):
            # Create search queries from categories
            search_queries = {}
            for category in selected_categories:
                keywords = rag.investment_categories[category]
                # Create a query from the keywords
                query = " ".join(keywords)
                search_queries[category] = query
            
            # Perform searches
            topic_results = perform_topic_search(rag, search_queries, relevance_threshold)
            
            # LLM validation if requested
            if use_llm_validation and st.session_state.get('anthropic_api_key'):
                st.info("🤖 Validating results with LLM...")
                validated_results = {}
                for topic_name, snippets in topic_results.items():
                    if snippets:  # Only validate if there are results
                        query = search_queries[topic_name]
                        validated_snippets = validate_topic_relevance_with_llm(
                            snippets, topic_name, query, st.session_state.anthropic_api_key
                        )
                        validated_results[topic_name] = validated_snippets
                    else:
                        validated_results[topic_name] = snippets
                
                topic_results = validated_results
                st.success("✅ LLM validation completed!")
            
            # Store results in session state
            st.session_state.topic_search_results = topic_results
            
            # Display results
            display_topic_search_results(topic_results)

def remove_manual_topic(topic_name):
    """Remove a manual topic."""
    if 'manual_topics' in st.session_state and topic_name in st.session_state.manual_topics:
        del st.session_state.manual_topics[topic_name]
        st.success(f"Removed topic: {topic_name}")

def display_manual_topics_overview():
    """Display overview of all manual topics."""
    st.subheader("📊 Manual Topics Overview")
    
    manual_topics = st.session_state.get('manual_topics', {})
    if not manual_topics:
        return
    
    # Create results dict for analysis
    topic_results = {name: data['snippets'] for name, data in manual_topics.items()}
    
    # Display summary table
    analysis_df = analyze_topic_distribution(topic_results)
    st.dataframe(analysis_df, use_container_width=True)
    
    # Visualizations
    visualize_topic_comparison(analysis_df)
    
    # Export functionality
    st.markdown("#### Export Manual Topics")
    if st.button("📥 Export Manual Topics Data"):
        export_topic_results(topic_results, "manual_topics_results")

def display_topic_search_results(topic_results):
    """Display results from topic search."""
    st.subheader("🔍 Topic Search Results")
    
    if not topic_results:
        st.warning("No search results to display.")
        return
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    total_snippets = sum(len(snippets) for snippets in topic_results.values())
    total_topics = len([t for t, s in topic_results.items() if s])  # Topics with results
    
    with col1:
        st.metric("Total Topics Searched", len(topic_results))
    with col2:
        st.metric("Topics with Results", total_topics)
    with col3:
        st.metric("Total Snippets Found", total_snippets)
    
    # Results table
    analysis_df = analyze_topic_distribution(topic_results)
    st.dataframe(analysis_df, use_container_width=True)
    
    # Visualizations
    visualize_topic_comparison(analysis_df)
    
    # Store for evolution analysis
    st.session_state.current_topic_results = topic_results
    
    # Export functionality
    st.markdown("#### Export Results")
    export_topic_results(topic_results, "investment_categories_search")
    
    st.success("✅ Topic search completed! You can now use these topics in Evolution Analysis.")

def render_all_snippets_selection(rag):
    """Render the 'Use All Snippets' section."""
    st.subheader("📚 Use All Snippets")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write(f"Total available snippets: **{len(rag.snippets)}**")
        
        # Sample size limit
        max_snippets = st.number_input(
            "Maximum snippets to use", 
            min_value=APP_CONFIG['min_snippets'], 
            max_value=len(rag.snippets), 
            value=min(APP_CONFIG['default_max_snippets'], len(rag.snippets)),
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
            filtered_snippets = apply_snippet_filters(
                rag.snippets, selected_companies, sentiment_filter, year_range
            )
            
            # Apply sample size limit
            if len(filtered_snippets) > max_snippets:
                random.seed(42)  # For reproducibility
                filtered_snippets = random.sample(filtered_snippets, max_snippets)
            
            # Store in session state
            st.session_state.selected_snippets = filtered_snippets
            st.session_state.selection_method = "All snippets"
            
            st.success(f"✅ Selected {len(filtered_snippets)} snippets for topic analysis!")
            st.info("Go to the 'Topic Analysis' tab to analyze these snippets.")

def render_search_selection(rag):
    """Render the search-specific selection section."""
    st.subheader("🔍 Search Specific Topic")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_type = st.radio("Search Type", ["Category", "Custom Query", "Semantic Search"])
        
        if search_type == "Category":
            category = st.selectbox("Select Investment Category", 
                                  list(rag.investment_categories.keys()))
            query = None
        elif search_type == "Semantic Search":
            query = st.text_input("Enter your semantic query")
            relevance_threshold = st.slider("Relevance Threshold", 
                                           min_value=0.0, max_value=1.0, 
                                           value=0.30, step=0.05,
                                           help="Higher values return more relevant but fewer results")
            category = None
        else:
            query = st.text_input("Enter your search query")
            category = None
    
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
        perform_search_and_select(rag, search_type, category, query, 
                                selected_companies, sentiment_filter, year_range,
                                relevance_threshold if search_type == "Semantic Search" else None)

def apply_snippet_filters(snippets, selected_companies, sentiment_filter, year_range):
    """Apply filters to snippets."""
    filtered_snippets = []
    for snippet in snippets:
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
    
    return filtered_snippets

def perform_search_and_select(rag, search_type, category, query, selected_companies, 
                            sentiment_filter, year_range, relevance_threshold=None):
    """Perform search and select snippets based on search type."""
    with st.spinner("Searching for relevant snippets..."):
        # Perform search based on type
        if search_type == "Category" and category:
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
            selected_snippets = convert_results_to_snippets(rag, results)
            
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

def convert_results_to_snippets(rag, results):
    """Convert search results back to snippet objects."""
    selected_snippets = []
    for result in results:
        # Find the corresponding snippet
        for snippet in rag.snippets:
            if (snippet.text == result['text'] and 
                snippet.company == result['company'] and
                snippet.ticker == result['ticker']):
                selected_snippets.append(snippet)
                break
    return selected_snippets

def render_selection_status():
    """Render current selection status."""
    if st.session_state.selected_snippets:
        st.markdown("---")
        st.subheader("📋 Current Selection")
        st.write(f"**Selected:** {len(st.session_state.selected_snippets)} snippets")
        st.write(f"**Method:** {st.session_state.get('selection_method', 'Unknown')}")
        
        if st.button("🗑️ Clear Selection"):
            st.session_state.selected_snippets = []
            st.session_state.selection_method = ""
            st.success("Selection cleared!")

def render_evolution_analysis_tab(rag):
    """Render the evolution analysis tab with support for multiple topic sources."""
    st.header("📈 Evolution Analysis")
    st.write("Analyze how topics evolve over time across markets")
    
    # Check available topic sources
    topic_sources = []
    
    # Check for BERTopic model from topic analysis
    if 'topic_model' in st.session_state and 'topic_names' in st.session_state:
        topic_sources.append("BERTopic Analysis")
    
    # Check for topic search results
    if 'topic_search_results' in st.session_state and st.session_state.topic_search_results:
        topic_sources.append("Investment Categories Search")
    
    # Check for manual topics
    if 'manual_topics' in st.session_state and st.session_state.manual_topics:
        topic_sources.append("Manual Topics")
    
    if not topic_sources:
        st.warning("⚠️ No topics available for analysis.")
        st.info("Please run one of the following first:")
        st.write("- Topic Search (Investment Categories)")
        st.write("- Manual Topic Identification")
        st.write("- Traditional Topic Analysis (BERTopic)")
        return
    
    # Topic source selection
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Topic Source")
        selected_source = st.selectbox(
            "Select topic source for analysis",
            topic_sources,
            help="Choose which set of topics to analyze"
        )
    
    with col2:
        st.subheader("🌍 Market Selection")
        show_eu = st.checkbox("Show EU (STOXX 600)", value=True)
        show_us = st.checkbox("Show US (S&P 500)", value=True)
        
        if not show_eu and not show_us:
            st.warning("Please select at least one market to display.")
            return
    
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
    
    # Topic selection based on source
    if selected_source == "BERTopic Analysis":
        render_bertopic_evolution_analysis(rag, show_eu, show_us, selected_years, time_granularity)
    elif selected_source == "Investment Categories Search":
        render_topic_search_evolution_analysis(rag, show_eu, show_us, selected_years, time_granularity)
    elif selected_source == "Manual Topics":
        render_manual_topics_evolution_analysis(rag, show_eu, show_us, selected_years, time_granularity)

def render_bertopic_evolution_analysis(rag, show_eu, show_us, selected_years, time_granularity):
    """Render evolution analysis for BERTopic results."""
    st.subheader("🎯 BERTopic Topic Selection")
    
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
    
    selected_topic = topic_options[selected_topic_display] if selected_topic_display else None
    
    if st.button("🔍 Analyze BERTopic Evolution", type="primary"):
        execute_bertopic_evolution_analysis(rag, selected_topic, selected_topic_display, 
                                          show_eu, show_us, selected_years, time_granularity)

def render_topic_search_evolution_analysis(rag, show_eu, show_us, selected_years, time_granularity):
    """Render evolution analysis for topic search results."""
    st.subheader("🎯 Investment Category Selection")
    
    topic_search_results = st.session_state.get('topic_search_results', {})
    available_topics = list(topic_search_results.keys())
    
    selected_topic = st.selectbox(
        "Select Investment Category to Analyze",
        options=available_topics,
        help="Choose an investment category to see its evolution over time"
    )
    
    if st.button("🔍 Analyze Category Evolution", type="primary"):
        execute_topic_search_evolution_analysis(rag, selected_topic, show_eu, show_us, 
                                               selected_years, time_granularity)

def render_manual_topics_evolution_analysis(rag, show_eu, show_us, selected_years, time_granularity):
    """Render evolution analysis for manual topics."""
    st.subheader("🎯 Manual Topic Selection")
    
    manual_topics = st.session_state.get('manual_topics', {})
    available_topics = list(manual_topics.keys())
    
    analysis_type = st.radio(
        "Analysis Type",
        ["Single Topic", "All Manual Topics"],
        help="Analyze one topic or compare all manual topics"
    )
    
    if analysis_type == "Single Topic":
        selected_topic = st.selectbox(
            "Select Manual Topic to Analyze",
            options=available_topics,
            help="Choose a manual topic to see its evolution over time"
        )
        
        if st.button("🔍 Analyze Manual Topic Evolution", type="primary"):
            execute_manual_topic_evolution_analysis(rag, selected_topic, show_eu, show_us, 
                                                   selected_years, time_granularity)
    else:
        if st.button("📊 Analyze All Manual Topics Evolution", type="primary"):
            execute_all_manual_topics_evolution_analysis(rag, show_eu, show_us, 
                                                        selected_years, time_granularity)

def execute_bertopic_evolution_analysis(rag, selected_topic, selected_topic_display, 
                                       show_eu, show_us, selected_years, time_granularity):
    """Execute BERTopic evolution analysis."""
    with st.spinner("Analyzing BERTopic evolution over time..."):
        try:
            topic_model = st.session_state.topic_model
            selected_snippets = get_selected_snippets()
            topic_names = st.session_state.topic_names
            
            evolution_data = analyze_topic_evolution_simple(
                selected_snippets, topic_model, selected_topic, 
                selected_years, time_granularity, topic_names
            )
            
            if not evolution_data:
                st.warning("No data found for the selected topic and time period.")
            else:
                display_evolution_charts_simple(evolution_data, selected_topic_display, time_granularity)
                display_evolution_insights_simple(evolution_data, selected_topic_display, time_granularity)
                
        except Exception as e:
            st.error(f"Error analyzing topic evolution: {str(e)}")

def execute_topic_search_evolution_analysis(rag, selected_topic, show_eu, show_us, 
                                           selected_years, time_granularity):
    """Execute topic search evolution analysis."""
    with st.spinner("Analyzing investment category evolution over time..."):
        try:
            # Get the snippets for this topic
            topic_search_results = st.session_state.get('topic_search_results', {})
            topic_snippets = topic_search_results.get(selected_topic, [])
            
            if not topic_snippets:
                st.warning(f"No snippets found for topic: {selected_topic}")
                return
            
            # Analyze evolution using the snippets
            evolution_data = analyze_snippets_evolution(
                topic_snippets, selected_topic, show_eu, show_us, 
                selected_years, time_granularity
            )
            
            if not evolution_data:
                st.warning("No data found for the selected time period and markets.")
            else:
                display_evolution_charts_simple(evolution_data, selected_topic, time_granularity)
                display_evolution_insights_simple(evolution_data, selected_topic, time_granularity)
                
        except Exception as e:
            st.error(f"Error analyzing category evolution: {str(e)}")

def analyze_all_topics_evolution_from_results(topic_results, show_eu, show_us, year_range, time_granularity):
    """Analyze evolution of all topics from results dict."""
    from src.utils import determine_market
    
    valid_topics = list(topic_results.keys())
    current_market = st.session_state.current_market
    
    # Group by time periods and topics
    time_groups = {}
    
    for topic_name, snippets in topic_results.items():
        for snippet in snippets:
            if snippet.year and str(snippet.year).isdigit():
                year = int(snippet.year)
                if year_range[0] <= year <= year_range[1]:
                    # Determine market
                    if current_market == "Full":
                        market = determine_market(snippet)
                    elif current_market == "Combined":
                        market = determine_market(snippet)
                    else:
                        market = current_market
                    
                    # Check if we should include this market
                    if not ((market == 'EU' and show_eu) or (market == 'US' and show_us)):
                        continue
                    
                    # Create period key
                    if time_granularity == "Yearly":
                        period = str(snippet.year)
                    else:  # Quarterly
                        period = f"{snippet.year}-Q{snippet.quarter}"
                    
                    if period not in time_groups:
                        time_groups[period] = {}
                    
                    # Count this topic for this market and period
                    topic_key = f"{market}_{topic_name}"
                    if topic_key not in time_groups[period]:
                        time_groups[period][topic_key] = 0
                    time_groups[period][topic_key] += 1
    
    # Convert to list format for visualization
    evolution_data = []
    all_periods = sorted(time_groups.keys()) if time_groups else []
    
    for period in all_periods:
        period_data = {'period': period}
        
        # Add counts for each topic and market combination
        for topic_name in valid_topics:
            if show_eu:
                eu_key = f"EU_{topic_name}"
                period_data[f"EU_{topic_name}"] = time_groups[period].get(eu_key, 0)
            
            if show_us:
                us_key = f"US_{topic_name}"
                period_data[f"US_{topic_name}"] = time_groups[period].get(us_key, 0)
        
        evolution_data.append(period_data)
    
    return evolution_data, valid_topics

def execute_manual_topic_evolution_analysis(rag, selected_topic, show_eu, show_us, 
                                           selected_years, time_granularity):
    """Execute manual topic evolution analysis."""
    with st.spinner("Analyzing manual topic evolution over time..."):
        try:
            from src.evolution_analysis import analyze_snippets_evolution
            from src.visualization import display_evolution_charts_simple, display_evolution_insights_simple
            
            # Get the snippets for this manual topic
            manual_topics = st.session_state.get('manual_topics', {})
            topic_data = manual_topics.get(selected_topic, {})
            topic_snippets = topic_data.get('snippets', [])
            
            if not topic_snippets:
                st.warning(f"No snippets found for manual topic: {selected_topic}")
                return
            
            # Analyze evolution using the snippets
            evolution_data = analyze_snippets_evolution(
                topic_snippets, selected_topic, show_eu, show_us, 
                selected_years, time_granularity
            )
            
            if not evolution_data:
                st.warning("No data found for the selected time period and markets.")
            else:
                display_evolution_charts_simple(evolution_data, selected_topic, time_granularity)
                display_evolution_insights_simple(evolution_data, selected_topic, time_granularity)
                
        except Exception as e:
            st.error(f"Error analyzing manual topic evolution: {str(e)}")

def execute_all_manual_topics_evolution_analysis(rag, show_eu, show_us, selected_years, time_granularity):
    """Execute evolution analysis for all manual topics."""
    with st.spinner("Analyzing all manual topics evolution over time..."):
        try:
            from src.evolution_analysis import analyze_all_topics_evolution_from_results
            from src.visualization import display_all_topics_stacked_chart, display_topics_summary
            
            manual_topics = st.session_state.get('manual_topics', {})
            
            # Create topic results dict
            topic_results = {name: data['snippets'] for name, data in manual_topics.items()}
            
            # Analyze all topics evolution
            all_topics_data, valid_topics = analyze_all_topics_evolution_from_results(
                topic_results, show_eu, show_us, selected_years, time_granularity
            )
            
            if not all_topics_data:
                st.warning("No data found for the selected time period and markets.")
            else:
                # Create topic names dict for visualization
                topic_names_for_viz = {i: name for i, name in enumerate(valid_topics)}
                
                display_all_topics_stacked_chart(
                    all_topics_data, list(range(len(valid_topics))), topic_names_for_viz,
                    show_eu, show_us, time_granularity
                )
                
                display_topics_summary(all_topics_data, list(range(len(valid_topics))), 
                                     topic_names_for_viz, show_eu, show_us)
                
        except Exception as e:
            st.error(f"Error analyzing all manual topics evolution: {str(e)}")

def render_manual_topics_evolution_analysis(rag, show_eu, show_us, selected_years, time_granularity):
    """Render evolution analysis for manual topics."""
    st.subheader("🎯 Manual Topic Selection")
    
    manual_topics = st.session_state.get('manual_topics', {})
    available_topics = list(manual_topics.keys())
    
    if not available_topics:
        st.warning("No manual topics defined yet.")
        st.info("Go to the 'Manual Topic ID' tab to create custom topics first.")
        return
    
    analysis_type = st.radio(
        "Analysis Type",
        ["Single Topic", "All Manual Topics"],
        help="Analyze one topic or compare all manual topics"
    )
    
    if analysis_type == "Single Topic":
        selected_topic = st.selectbox(
            "Select Manual Topic to Analyze",
            options=available_topics,
            help="Choose a manual topic to see its evolution over time"
        )
        
        if st.button("🔍 Analyze Manual Topic Evolution", type="primary"):
            execute_manual_topic_evolution_analysis(rag, selected_topic, show_eu, show_us, 
                                                   selected_years, time_granularity)
    else:
        st.write("Analyze how all manual topics evolve over time")
        
        if st.button("📊 Analyze All Manual Topics Evolution", type="primary"):
            execute_all_manual_topics_evolution_analysis(rag, show_eu, show_us, 
                                                        selected_years, time_granularity)

def execute_all_manual_topics_evolution_analysis(rag, show_eu, show_us, selected_years, time_granularity):
    """Execute evolution analysis for all manual topics."""
    with st.spinner("Analyzing all manual topics evolution over time..."):
        try:
            from src.evolution_analysis import analyze_all_topics_evolution_from_results
            from src.visualization import display_all_topics_stacked_chart, display_topics_summary
            
            manual_topics = st.session_state.get('manual_topics', {})
            
            # Create topic results dict
            topic_results = {name: data['snippets'] for name, data in manual_topics.items()}
            
            # Analyze all topics evolution
            all_topics_data, valid_topics = analyze_all_topics_evolution_from_results(
                topic_results, show_eu, show_us, selected_years, time_granularity
            )
            
            if not all_topics_data:
                st.warning("No data found for the selected time period and markets.")
            else:
                # Create topic names dict for visualization
                topic_names_for_viz = {i: name for i, name in enumerate(valid_topics)}
                
                display_all_topics_stacked_chart(
                    all_topics_data, list(range(len(valid_topics))), topic_names_for_viz,
                    show_eu, show_us, time_granularity
                )
                
                display_topics_summary(all_topics_data, list(range(len(valid_topics))), 
                                     topic_names_for_viz, show_eu, show_us)
                
        except Exception as e:
            st.error(f"Error analyzing all manual topics evolution: {str(e)}")

def render_event_definition_section():
    st.subheader("📋 Event Definition")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Predefined Climate Events")
        predefined_events = load_predefined_events()
        selected_events = st.multiselect(
            "Select events to analyze",
            list(predefined_events.keys()),
            default=["Paris Agreement Adoption", "EU Green Deal"]
        )
    
    with col2:
        st.markdown("#### Custom Event")
        custom_event_name = st.text_input("Event Name")
        custom_event_date = st.date_input("Event Date")
        if st.button("Add Custom Event"):
            add_custom_event(custom_event_name, custom_event_date)

# In ui_components.py, add this new function:

def render_event_studies_tab(rag):
    """Render the event studies tab for analyzing policy events."""
    st.header("📅 Event Studies")
    st.write("Analyze how climate policy events affect firm discussions in earnings calls")
    
    # Check if data is loaded
    if not st.session_state.data_loaded:
        st.warning("Please load market data first using the sidebar.")
        return
    
    # Event definition section
    render_event_definition_section()
    
    # Keywords for analysis
    render_keywords_selection_section()
    
    # Event window settings
    event_window, aggregation = render_event_window_settings()
    
    # Regional analysis options
    render_regional_analysis_options()
    
    # Analysis execution
    if st.button("🚀 Run Event Study Analysis", type="primary"):
        execute_event_study_analysis(rag, event_window, aggregation)

def render_event_definition_section():
    """Render the event definition and selection section."""
    st.subheader("📋 Event Definition")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Predefined Climate Events")
        predefined_events = {
            "Paris Agreement Adoption": "2015-12-12",
            "Trump Election": "2016-11-08", 
            "Biden Election": "2020-11-07",
            "EU Green Deal": "2019-12-11",
            "US IRA Passage": "2022-08-16",
            "COP21 Opening": "2015-11-30",
            "US Paris Withdrawal": "2017-06-01",
            "US Paris Re-entry": "2021-01-20"
        }
        
        selected_events = st.multiselect(
            "Select events to analyze",
            list(predefined_events.keys()),
            default=["Paris Agreement Adoption", "EU Green Deal"],
            help="Choose which climate policy events to analyze"
        )
        
        # Store in session state
        st.session_state.selected_events = {
            name: predefined_events[name] for name in selected_events
        }
    
    with col2:
        st.markdown("#### Custom Event")
        custom_event_name = st.text_input("Event Name", placeholder="e.g., State Climate Law")
        custom_event_date = st.date_input("Event Date")
        
        if st.button("➕ Add Custom Event"):
            if custom_event_name and custom_event_date:
                if 'custom_events' not in st.session_state:
                    st.session_state.custom_events = {}
                st.session_state.custom_events[custom_event_name] = str(custom_event_date)
                st.success(f"Added custom event: {custom_event_name}")
                st.rerun()

def render_keywords_selection_section():
    """Render semantic search query selection for event analysis."""
    st.subheader("🔍 Semantic Search Queries")
    
    # Predefined event-specific queries
    event_queries = {
        "Paris Agreement": [
            "paris agreement climate accord international climate deal",
            "COP21 climate summit global climate agreement",
            "international climate policy framework"
        ],
        "EU Green Deal": [
            "european green deal climate policy",
            "EU climate legislation environmental regulation",
            "european climate framework sustainability policy"
        ],
        "General Climate Policy": [
            "climate policy environmental regulation",
            "carbon regulation emissions policy",
            "climate legislation regulatory framework"
        ],
        "Carbon Pricing": [
            "carbon pricing emissions trading",
            "carbon tax climate pricing mechanism",
            "cap and trade carbon markets"
        ]
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Event-Specific Queries")
        selected_events = st.session_state.get('selected_events', {})
        
        # Show relevant queries based on selected events
        suggested_queries = []
        for event_name in selected_events.keys():
            if "Paris Agreement" in event_name:
                suggested_queries.extend(event_queries["Paris Agreement"])
            elif "Green Deal" in event_name:
                suggested_queries.extend(event_queries["EU Green Deal"])
        
        # Add general queries
        suggested_queries.extend(event_queries["General Climate Policy"])
        
        # Remove duplicates
        suggested_queries = list(set(suggested_queries))
        
        selected_queries = st.multiselect(
            "Select semantic search queries",
            suggested_queries,
            default=suggested_queries[:3] if len(suggested_queries) >= 3 else suggested_queries,
            help="Choose semantic search queries that capture the essence of climate policy discussions"
        )
    
    with col2:
        st.markdown("#### Custom Queries")
        custom_queries = st.text_area(
            "Additional semantic queries (one per line)",
            placeholder="climate transition strategy\nregulatory uncertainty environment\ncarbon neutral sustainability",
            help="Add custom semantic search queries"
        )
        
        # Relevance threshold
        relevance_threshold = st.slider(
            "Relevance Threshold",
            min_value=0.1,
            max_value=0.6,
            value=0.25,
            step=0.05,
            help="Minimum similarity score for including snippets"
        )
        
        st.session_state.relevance_threshold = relevance_threshold
    
    # Combine all queries
    all_queries = selected_queries.copy()
    if custom_queries:
        custom_list = [q.strip() for q in custom_queries.split('\n') if q.strip()]
        all_queries.extend(custom_list)
    
    # Store queries in session state
    st.session_state.event_study_queries = all_queries
    
    # Show selected queries
    if all_queries:
        st.markdown("**Selected Semantic Queries:**")
        for i, query in enumerate(all_queries, 1):
            st.write(f"{i}. *{query}*")

def render_event_window_settings():
    """Render event window configuration."""
    st.subheader("⏰ Event Window Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pre_event_days = st.number_input(
            "Days Before Event", 
            min_value=1, max_value=365, value=90,
            help="Number of days before the event to include in analysis"
        )
    
    with col2:
        post_event_days = st.number_input(
            "Days After Event", 
            min_value=1, max_value=365, value=90,
            help="Number of days after the event to include in analysis"
        )
    
    with col3:
        aggregation_level = st.selectbox(
            "Aggregation Level",
            ["Daily", "Weekly", "Monthly", "Quarterly"],
            index=2,  # Default to Monthly
            help="How to group the data for analysis"
        )
    
    return (-pre_event_days, post_event_days), aggregation_level

def render_regional_analysis_options():
    """Render options for regional comparison."""
    st.subheader("🌍 Regional Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        compare_regions = st.checkbox(
            "Compare EU vs US Response", 
            value=True,
            help="Analyze how EU and US firms differ in their response to events"
        )
    
    with col2:
        if compare_regions:
            analysis_type = st.radio(
                "Comparison Type",
                ["Side-by-side", "Difference"],
                help="How to display regional comparisons"
            )
            st.session_state.regional_comparison = analysis_type
    
    st.session_state.compare_regions = compare_regions

def execute_event_study_analysis(rag, event_window, aggregation):
    """Execute the event study analysis using semantic search."""
    
    # Get stored parameters
    selected_events = st.session_state.get('selected_events', {})
    search_queries = st.session_state.get('event_study_queries', [])
    compare_regions = st.session_state.get('compare_regions', False)
    
    if not selected_events:
        st.warning("Please select at least one event to analyze.")
        return
    
    if not search_queries:
        st.warning("Please select semantic search queries for analysis.")
        return
    
    # Store event window in session state for analyzer to access
    st.session_state.event_window = event_window
    
    with st.spinner("Running semantic event study analysis..."):
        try:
            # Import the event study analyzer here to avoid circular imports
            from event_study_analyzer import EventStudyAnalyzer
            
            analyzer = EventStudyAnalyzer(rag)
            
            # Analyze each selected event
            results = {}
            for event_name, event_date in selected_events.items():
                st.info(f"Analyzing: {event_name} using semantic search")
                
                # Add validation for event date
                try:
                    # Handle different possible date formats
                    if isinstance(event_date, str):
                        # Try different formats
                        for date_format in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"]:
                            try:
                                datetime.strptime(event_date, date_format)
                                break
                            except ValueError:
                                continue
                        else:
                            st.error(f"Invalid date format for {event_name}: {event_date}")
                            continue
                    elif isinstance(event_date, date):
                        # Convert date object to string
                        event_date = event_date.strftime("%Y-%m-%d")
                    else:
                        st.error(f"Invalid date type for {event_name}: {type(event_date)}")
                        continue
                    
                    # Use semantic search instead of keywords
                    event_results = analyzer.analyze_event_impact(
                        event_date, event_window, search_queries, compare_regions
                    )
                    results[event_name] = event_results
                    
                except Exception as e:
                    st.error(f"Error analyzing {event_name}: {str(e)}")
                    continue
            
            if results:
                # Display results
                display_event_study_results(results, compare_regions)
                
                # Store results for export
                st.session_state.event_study_results = results
                st.success("✅ Semantic event study analysis completed!")
            else:
                st.warning("No valid events were analyzed.")
            
        except Exception as e:
            st.error(f"Error in event study analysis: {str(e)}")
            st.error("Please check the error details below:")
            st.exception(e)
            st.info("Try selecting different events or search queries.")

def display_event_study_results(results, compare_regions):
    """Display the event study results."""
    st.subheader("📊 Event Study Results")
    
    # Results tabs for each event
    if len(results) == 1:
        print("yes it 1 event")
        # Single event - show directly
        event_name, event_data = list(results.items())[0]
        display_single_event_results(event_name, event_data, compare_regions)
    else:
        print('multiple events')
        # Multiple events - create sub-tabs
        event_tabs = st.tabs(list(results.keys()))
        for i, (event_name, event_data) in enumerate(results.items()):
            with event_tabs[i]:
                display_single_event_results(event_name, event_data, compare_regions)
    
    # Multi-event comparison if more than one event
    if len(results) > 1:
        st.markdown("---")
        st.subheader("🔄 Multi-Event Comparison")
        display_multi_event_comparison(results)

def display_single_event_results(event_name, event_data, compare_regions):
    """Display results for a single event."""
    # This function will contain the visualization code
    # that we defined earlier (timeline plots, statistical tests, etc.)
    pass

def display_multi_event_comparison(results):
    """Display comparison across multiple events."""
    # This function will create comparison visualizations
    pass

