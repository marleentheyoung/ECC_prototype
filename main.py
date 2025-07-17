# main.py - Main application file
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
from src.config import setup_environment, check_required_libraries, APP_CONFIG
from src.ui_components import (
    render_sidebar, render_evolution_analysis_tab, render_enhanced_manual_topic_tab, render_event_studies_tab
)
from src.topic_analysis import run_topic_analysis, display_topic_results, create_topic_results_dataframe
from src.utils import generate_topic_names
from src.simplified_snippet_selection import render_simplified_snippet_selection

def initialize_session_state():
    """Initialize session state variables."""
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
    if 'manual_topics' not in st.session_state:
        st.session_state.manual_topics = {}
    if 'topic_search_results' not in st.session_state:
        st.session_state.topic_search_results = {}
    if 'selected_events' not in st.session_state:
        st.session_state.selected_events = {}
    if 'event_study_keywords' not in st.session_state:
        st.session_state.event_study_keywords = []
    if 'event_study_results' not in st.session_state:
        st.session_state.event_study_results = {}

def main():
    """Main application function."""
    # Setup environment
    setup_environment()
    
    # Configure Streamlit page
    st.set_page_config(
        page_title=APP_CONFIG['page_title'],
        page_icon=APP_CONFIG['page_icon'],
        layout=APP_CONFIG['layout']
    )
    
    # Check required libraries
    libraries_available, error_msg = check_required_libraries()
    
    if not libraries_available:
        st.error("Required libraries not available. Please install them first.")
        st.error(f"Error: {error_msg}")
        st.code("pip install umap-learn bertopic")
        return
    
    # Initialize session state
    initialize_session_state()
    
    # Main app title and description
    st.title("🌱 Green Investment Analyzer")
    st.subheader("Extract climate investment insights from earnings calls")
    
    # Render sidebar
    render_sidebar()
    
    # Check if data is loaded
    if not st.session_state.data_loaded:
        st.warning("Please select and load market data using the sidebar.")
        return
    
    rag = st.session_state.rag_system
    
    # Display current market info
    st.info(f"📊 Currently analyzing: {st.session_state.current_market} market with {len(rag.snippets)} snippets")
    st.info("🔧 Threading: Single-threaded (macOS optimized)")
    
    # Main interface tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Snippet Selection", 
        "🎯 Topic Search", 
        "📝 Manual Topic ID", 
        "📅 Event Studies",  # <-- ADD THIS NEW TAB
        "📊 Evolution Analysis"
    ])
    
    with tab1:
        render_simplified_snippet_selection(rag)  # New simplified version
    
    with tab2:
        render_topic_analysis_tab(rag)  # Restored BERTopic functionality
    
    with tab3:
        render_enhanced_manual_topic_tab(rag)  # Existing with adaptive validation
    
    with tab4:
        render_event_studies_tab(rag)  # Existing

    with tab5:
        render_evolution_analysis_tab(rag)

def render_topic_analysis_tab(rag):
    """Render the BERTopic analysis tab."""
    st.header("📈 Topic Analysis")
    
    # Check if snippets are selected
    selected_snippets = st.session_state.get('selected_snippets', [])
    
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
                             max_value=min(15, len(selected_snippets)//20),
                             value=min(6, len(selected_snippets)//20))
    
    with col2:
        st.subheader("Analysis Info")
        st.metric("Selected Snippets", len(selected_snippets))
        st.metric("Max Topics", min(15, len(selected_snippets)//20))
    
    if st.button("🚀 Run Topic Analysis", type="primary"):
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
                
                # Show topic distribution chart
                st.subheader("📊 Topic Distribution")
                topic_counts = topic_info[topic_info['Topic'] != -1]['Count'].tolist()
                topic_labels = [topic_names.get(t, f"Topic {t}") for t in topic_info[topic_info['Topic'] != -1]['Topic'].tolist()]
                
                if topic_counts and topic_labels:
                    fig = px.bar(
                        x=topic_labels, 
                        y=topic_counts,
                        title="Number of Documents per Topic",
                        labels={'x': 'Topic', 'y': 'Document Count'}
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Store results in session state for evolution analysis
                st.session_state.topic_model = topic_model
                st.session_state.topic_names = topic_names
                st.session_state.topic_info = topic_info
                st.session_state.topics = topics
                
                # Export functionality
                st.markdown("#### 📥 Export Results")
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
                st.info("Try reducing the number of topics or selecting more snippets.")


if __name__ == "__main__":
    main()