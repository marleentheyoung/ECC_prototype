# ui_components.py - UI component functions for different tabs
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st

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
