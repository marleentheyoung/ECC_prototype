# main.py - Streamlined main application file
import streamlit as st
from src.config import setup_environment, check_required_libraries, APP_CONFIG
from src.ui_components import render_sidebar
from tabs.event_studies import render_event_studies_tab
from tabs.manual_topics import render_manual_topics_tab
from tabs.evolution_analysis import render_evolution_analysis_tab  # Fixed import
from src.simplified_snippet_selection import render_simplified_snippet_selection
from src.topic_analysis import run_topic_analysis, display_topic_results, create_topic_results_dataframe
from src.utils import generate_topic_names
import plotly.express as px

def initialize_session_state():
    """Initialize all session state variables in one place."""
    defaults = {
        'data_loaded': False,
        'rag_system': None,
        'current_market': None,
        'anthropic_api_key': None,
        'selected_snippets': [],
        'selection_method': '',
        'manual_topics': {},
        'topic_search_results': {},
        'selected_events': {},
        'event_study_keywords': [],
        'event_study_results': {}
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def check_prerequisites():
    """Check if required libraries and data are available."""
    # Check libraries
    libraries_available, error_msg = check_required_libraries()
    if not libraries_available:
        st.error("❌ Required libraries not available")
        st.error(f"Error: {error_msg}")
        st.code("pip install umap-learn bertopic sentence-transformers")
        return False
    
    # Check data loaded
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please select and load market data using the sidebar")
        return False
    
    return True

def render_topic_analysis_tab(rag):
    """Streamlined BERTopic analysis tab."""
    st.header("📈 Topic Analysis (BERTopic)")
    
    # Check prerequisites
    selected_snippets = st.session_state.get('selected_snippets', [])
    if not selected_snippets:
        st.warning("⚠️ No snippets selected for analysis")
        st.info("👆 Go to 'Snippet Selection' tab first to select snippets")
        return
    
    # Display current selection info
    st.success(f"✅ Ready to analyze {len(selected_snippets)} selected snippets")
    st.caption(f"Selection method: {st.session_state.get('selection_method', 'Unknown')}")
    
    # Topic analysis configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        max_topics_possible = min(15, len(selected_snippets) // 20)
        nr_topics = st.slider(
            "Number of Topics to Find", 
            min_value=2, 
            max_value=max_topics_possible,
            value=min(6, max_topics_possible),
            help=f"Maximum {max_topics_possible} topics for {len(selected_snippets)} snippets"
        )
    
    with col2:
        st.metric("Selected Snippets", len(selected_snippets))
        st.metric("Max Recommended Topics", max_topics_possible)
    
    # Run analysis
    if st.button("🚀 Run Topic Analysis", type="primary"):
        with st.spinner("Running BERTopic analysis..."):
            try:
                # Run topic modeling
                topic_model, topics, topic_info = run_topic_analysis(selected_snippets, nr_topics)
                
                if topic_model is None:
                    st.error("❌ Topic analysis failed")
                    return
                
                # Generate LLM names for topics
                if st.session_state.get('anthropic_api_key'):
                    with st.spinner("🤖 Generating topic names with LLM..."):
                        topic_names = generate_topic_names(
                            topic_model, topic_info, st.session_state.anthropic_api_key
                        )
                else:
                    topic_names = {i: f"Topic {i}" for i in topic_info['Topic'].tolist() if i != -1}
                    st.info("💡 Add Anthropic API key in sidebar for automatic topic naming")
                
                # Display results
                display_topic_results(topic_model, topic_info, topic_names)
                
                # Topic distribution chart
                st.subheader("📊 Topic Distribution")
                valid_topics = topic_info[topic_info['Topic'] != -1]
                if not valid_topics.empty:
                    fig = px.bar(
                        x=[topic_names.get(t, f"Topic {t}") for t in valid_topics['Topic']], 
                        y=valid_topics['Count'],
                        title="Documents per Topic",
                        labels={'x': 'Topic', 'y': 'Document Count'}
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Store results for other tabs
                st.session_state.topic_model = topic_model
                st.session_state.topic_names = topic_names
                st.session_state.topic_info = topic_info
                st.session_state.topics = topics
                
                # Export functionality
                st.markdown("#### 📥 Export Results")
                results_df = create_topic_results_dataframe(selected_snippets, topics, topic_names)
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="topic_analysis_results.csv",
                    mime="text/csv"
                )
                
                st.success("✅ Analysis completed! Use 'Evolution Analysis' tab to see trends over time")
                
            except Exception as e:
                st.error(f"❌ Error in topic analysis: {str(e)}")
                st.info("💡 Try reducing number of topics or selecting more snippets")

def main():
    """Main application function - streamlined and clear."""
    
    # Setup
    setup_environment()
    st.set_page_config(
        page_title=APP_CONFIG['page_title'],
        page_icon=APP_CONFIG['page_icon'],
        layout=APP_CONFIG['layout']
    )
    
    # Initialize
    initialize_session_state()
    
    # Header
    st.title("🌱 Green Investment Analyzer")
    st.markdown("*Extract climate investment insights from earnings calls using semantic search and topic modeling*")
    
    # Sidebar
    render_sidebar()
    
    # Check prerequisites
    if not check_prerequisites():
        return
    
    # Main content
    rag = st.session_state.rag_system
    st.success(f"📊 Loaded: {st.session_state.current_market} market ({len(rag.snippets)} snippets)")
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Snippet Selection", 
        "📈 Topic Analysis", 
        "📝 Manual Topics", 
        "📅 Event Studies",
        "📊 Evolution Analysis"
    ])
    
    with tab1:
        render_simplified_snippet_selection(rag)
    
    with tab2:
        render_topic_analysis_tab(rag)
    
    with tab3:
        render_manual_topics_tab(rag)
    
    with tab4:
        render_event_studies_tab(rag)

    with tab5:
        render_evolution_analysis_tab(rag)

if __name__ == "__main__":
    main()