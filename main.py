# main.py - Main application file
import streamlit as st
from config import setup_environment, check_required_libraries, APP_CONFIG
from ui_components import (
    render_sidebar, render_snippet_selection_tab, 
    render_topic_analysis_tab, render_evolution_analysis_tab
)

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
    tab1, tab2, tab3 = st.tabs(["🔍 Snippet Selection", "📈 Topic Analysis", "📊 Evolution Analysis"])
    
    with tab1:
        render_snippet_selection_tab(rag)
    
    with tab2:
        render_topic_analysis_tab()
    
    with tab3:
        render_evolution_analysis_tab(rag)

if __name__ == "__main__":
    main()