# ui_components.py - UI component functions for different tabs
import streamlit as st
import pandas as pd
import random
from config import APP_CONFIG
from utils import display_results, get_selected_snippets
from data_loaders import load_market_data
from topic_analysis import run_topic_analysis, display_topic_results, create_topic_results_dataframe
from evolution_analysis import analyze_topic_evolution_simple, analyze_all_topics_evolution
from visualization import (
    display_topic_distribution, display_evolution_charts_simple, 
    display_evolution_insights_simple, display_all_topics_stacked_chart, 
    display_topics_summary
)
from utils import generate_topic_names

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
                               help="Enter your Anthropic API key for automatic topic naming")
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

def render_topic_analysis_tab():
    """Render the topic analysis tab."""
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
        max_possible_topics = min(APP_CONFIG['max_topics'], len(selected_snippets)//20)
        nr_topics = st.slider("Number of Topics to Find", 
                             min_value=2, 
                             max_value=max_possible_topics,
                             value=min(APP_CONFIG['default_topics'], max_possible_topics))
    
    with col2:
        st.subheader("Analysis Info")
        st.metric("Selected Snippets", len(selected_snippets))
        st.metric("Max Topics", max_possible_topics)
    
    if st.button("🚀 Run Topic Analysis", type="primary"):
        run_topic_analysis_workflow(selected_snippets, nr_topics)

def run_topic_analysis_workflow(selected_snippets, nr_topics):
    """Run the complete topic analysis workflow."""
    with st.spinner("Running topic analysis..."):
        try:
            # Run topic analysis
            topic_model, topics, topic_info = run_topic_analysis(selected_snippets, nr_topics)
            
            if topic_model is None:
                return
            
            # Generate topic names
            st.subheader("🤖 Generating Topic Names...")
            with st.spinner("Generating meaningful topic names using LLM..."):
                topic_names = generate_topic_names(
                    topic_model, 
                    topic_info, 
                    st.session_state.get('anthropic_api_key')
                )
            
            # Display results
            display_topic_results(topic_model, topic_info, topic_names)
            display_topic_distribution(topic_info, topic_names)
            
            # Store results in session state
            st.session_state.topic_model = topic_model
            st.session_state.topic_names = topic_names
            st.session_state.topic_info = topic_info
            
            # Export functionality
            st.markdown("#### Export Results")
            results_df = create_topic_results_dataframe(selected_snippets, topics, topic_names)
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

def render_evolution_analysis_tab(rag):
    """Render the evolution analysis tab."""
    st.header("📈 Subtopic Evolution Analysis")
    
    # Check if topic model exists from tab 2
    if 'topic_model' not in st.session_state or 'topic_names' not in st.session_state:
        st.warning("⚠️ Please run Topic Analysis in the 'Subtopic identification' tab first to generate topics.")
        st.info("Go to tab 2 and click 'Run Topic Analysis' to identify subtopics before analyzing their evolution.")
        return
    
    # Market and topic selection
    show_eu, show_us, selected_topic_display, selected_topic = render_evolution_controls()
    
    if not show_eu and not show_us:
        st.warning("Please select at least one market to display.")
        return
    
    # Time period selection
    time_granularity, selected_years = render_time_controls(rag)
    
    if selected_years is None:
        return
    
    # Analysis type selection and execution
    render_analysis_controls(rag, selected_topic, selected_topic_display, 
                           show_eu, show_us, selected_years, time_granularity)

def render_evolution_controls():
    """Render market and topic selection controls."""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🌍 Market Selection")
        show_eu = st.checkbox("Show EU (STOXX 600)", value=True)
        show_us = st.checkbox("Show US (S&P 500)", value=True)
    
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
        
        selected_topic = topic_options[selected_topic_display] if selected_topic_display else None
    
    return show_eu, show_us, selected_topic_display, selected_topic

def render_time_controls(rag):
    """Render time period selection controls."""
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
            return None, None
    
    return time_granularity, selected_years

def render_analysis_controls(rag, selected_topic, selected_topic_display, 
                           show_eu, show_us, selected_years, time_granularity):
    """Render analysis type controls and execute analysis."""
    st.subheader("🔬 Analysis Type")
    analysis_type = st.radio(
        "Choose Analysis Type:",
        ["Single Topic Evolution", "All Topics Evolution"],
        help="Select whether to analyze one specific topic or all topics together"
    )
    
    # Single analysis button
    if analysis_type == "Single Topic Evolution":
        if st.button("🔍 Analyze Single Topic Evolution", type="primary"):
            execute_single_topic_analysis(selected_topic, selected_topic_display, 
                                         selected_years, time_granularity)
    
    # All topics analysis button
    else:  # "All Topics Evolution"
        if st.button("📊 Analyze All Topics Evolution", type="primary"):
            execute_all_topics_analysis(rag, show_eu, show_us, selected_years, time_granularity)

def execute_single_topic_analysis(selected_topic, selected_topic_display, selected_years, time_granularity):
    """Execute single topic evolution analysis."""
    with st.spinner("Analyzing single topic evolution over time..."):
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
                # Display evolution charts
                display_evolution_charts_simple(evolution_data, selected_topic_display, time_granularity)
                
                # Display detailed insights
                display_evolution_insights_simple(evolution_data, selected_topic_display, time_granularity)
                
                # Download option
                df = pd.DataFrame(evolution_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Evolution Data as CSV",
                    data=csv,
                    file_name=f"topic_evolution_{selected_topic_display.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error analyzing topic evolution: {str(e)}")
            st.info("Please ensure you have run the topic analysis first and selected valid parameters.")

def execute_all_topics_analysis(rag, show_eu, show_us, selected_years, time_granularity):
    """Execute all topics evolution analysis."""
    with st.spinner("Analyzing all topics evolution over time..."):
        try:
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
                
                # Display summary statistics
                display_topics_summary(all_topics_data, valid_topics, topic_names, show_eu, show_us)
                
                # Download option for all topics data
                st.markdown("#### Export All Topics Data")
                df = pd.DataFrame(all_topics_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download All Topics Evolution Data as CSV",
                    data=csv,
                    file_name=f"all_topics_evolution_{time_granularity.lower()}.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error analyzing all topics evolution: {str(e)}")
            st.info("Please ensure you have run the topic analysis first and selected valid parameters.")