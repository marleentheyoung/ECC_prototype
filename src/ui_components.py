# ui_components.py - Core UI components only (cleaned and simplified)
import streamlit as st
import random
from src.data_loaders import load_market_data

# Import the new tab modules
from tabs.event_studies import render_event_studies_tab
from tabs.manual_topics import render_manual_topics_tab

def render_sidebar():
    """Streamlined sidebar with market selection and API key."""
    with st.sidebar:
        st.header("🌍 Market Data")
        
        # Market selection
        market_option = st.selectbox(
            "Select Market",
            [
                "EU (STOXX 600)", 
                "US (S&P 500)", 
                "Full Index (EU + US)",
                "Combined (EU + US)"  # Fallback option
            ],
            index=0,
            help="Choose which market data to analyze"
        )
        
        # Load button
        if st.button("📊 Load Data", type="primary"):
            load_market_data_with_feedback(market_option)
        
        # Show current status
        display_data_status()
        
        # API Configuration
        st.markdown("---")
        st.header("🤖 AI Settings")
        
        api_key = st.text_input(
            "Anthropic API Key", 
            type="password",
            help="Required for AI topic naming and validation",
            placeholder="sk-ant-..."
        )
        
        if api_key:
            st.session_state.anthropic_api_key = api_key
            st.success("✅ API key configured")
        else:
            st.info("💡 Add API key for AI features")

def load_market_data_with_feedback(market_option):
    """Load market data with user feedback."""
    with st.spinner(f"Loading {market_option} data..."):
        try:
            rag, market_key, success = load_market_data(market_option)
            
            if success and rag:
                # Store in session state
                st.session_state.rag_system = rag
                st.session_state.current_market = market_key
                st.session_state.data_loaded = True
                
                # Success feedback
                st.success(f"✅ Loaded {len(rag.snippets):,} snippets from {market_option}")
                
                # Show data info
                if market_option == "Full Index (EU + US)":
                    st.info("🚀 Using optimized full index for best performance")
                elif market_option == "Combined (EU + US)":
                    st.warning("⚠️ Using fallback method - consider 'Full Index' if available")
                
                st.rerun()
                
            else:
                st.error(f"❌ Failed to load {market_option} data")
                st.session_state.data_loaded = False
                
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.session_state.data_loaded = False

def display_data_status():
    """Display current data loading status."""
    if st.session_state.get('data_loaded', False):
        rag = st.session_state.get('rag_system')
        market = st.session_state.get('current_market', 'Unknown')
        
        if rag:
            st.success(f"✅ **{market}** market loaded")
            st.caption(f"{len(rag.snippets):,} snippets available")
        else:
            st.error("❌ Data loaded but RAG system missing")
    else:
        st.warning("⚠️ No data loaded")

def render_enhanced_manual_topic_tab(rag):
    """Wrapper for the new manual topics tab."""
    render_manual_topics_tab(rag)

def render_selection_status():
    """Display current snippet selection status."""
    selected_snippets = st.session_state.get('selected_snippets', [])
    selection_method = st.session_state.get('selection_method', '')
    
    if selected_snippets:
        st.markdown("---")
        st.subheader("📋 Current Selection")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.success(f"✅ **{len(selected_snippets)} snippets** selected")
            if selection_method:
                st.caption(f"Method: {selection_method}")
        
        with col2:
            # Show basic stats
            if selected_snippets:
                companies = len(set(s.ticker for s in selected_snippets))
                years = [int(s.year) for s in selected_snippets if s.year and str(s.year).isdigit()]
                year_range = f"{min(years)}-{max(years)}" if years else "N/A"
                
                st.metric("Companies", companies)
                st.caption(f"Years: {year_range}")
        
        with col3:
            if st.button("🗑️ Clear Selection", help="Remove current selection"):
                clear_selection()
                st.rerun()
    else:
        st.info("📝 No snippets selected yet")

def clear_selection():
    """Clear current snippet selection."""
    st.session_state.selected_snippets = []
    st.session_state.selection_method = ""

def apply_snippet_filters(snippets, selected_companies=None, sentiment_filter="All", year_range=None):
    """Apply filters to a list of snippets."""
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

def convert_results_to_snippets(rag, results):
    """Convert search results back to snippet objects."""
    selected_snippets = []
    
    for result in results:
        # Find the corresponding snippet
        for snippet in rag.snippets:
            if (snippet.text == result['text'] and 
                snippet.company == result['company'] and
                snippet.ticker == result['ticker']):
                # Add score if available
                if 'score' in result:
                    snippet.score = result['score']
                selected_snippets.append(snippet)
                break
    
    return selected_snippets

def get_filter_options(rag):
    """Get available filter options from the RAG system."""
    # Get unique companies
    companies = sorted(list(set(s.ticker for s in rag.snippets)))
    
    # Get year range
    years = [int(s.year) for s in rag.snippets if s.year and str(s.year).isdigit()]
    year_range = (min(years), max(years)) if years else (2010, 2024)
    
    # Sentiment options
    sentiments = ["All", "opportunity", "neutral", "risk"]
    
    return {
        'companies': companies,
        'year_range': year_range,
        'sentiments': sentiments
    }

def render_filter_sidebar(rag, key_prefix="filter"):
    """Render a standardized filter sidebar."""
    filter_options = get_filter_options(rag)
    
    st.sidebar.markdown("### 🔍 Filters")
    
    # Company filter
    selected_companies = st.sidebar.multiselect(
        "Companies",
        filter_options['companies'],
        key=f"{key_prefix}_companies"
    )
    
    # Sentiment filter
    sentiment_filter = st.sidebar.selectbox(
        "Sentiment",
        filter_options['sentiments'],
        key=f"{key_prefix}_sentiment"
    )
    
    # Year range filter
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=filter_options['year_range'][0],
        max_value=filter_options['year_range'][1],
        value=filter_options['year_range'],
        key=f"{key_prefix}_years"
    )
    
    return {
        'companies': selected_companies if selected_companies else None,
        'sentiment': sentiment_filter,
        'year_range': year_range
    }

def sample_snippets_safely(snippets, max_size, seed=42):
    """Safely sample snippets with a maximum size limit."""
    if len(snippets) <= max_size:
        return snippets
    
    # Use consistent random sampling
    random.seed(seed)
    return random.sample(snippets, max_size)

def validate_snippet_selection(snippets, min_required=10):
    """Validate that snippet selection meets minimum requirements."""
    if not snippets:
        return False, "No snippets selected"
    
    if len(snippets) < min_required:
        return False, f"Need at least {min_required} snippets, got {len(snippets)}"
    
    # Check for basic data quality
    valid_snippets = [s for s in snippets if s.text and len(s.text.strip()) > 10]
    
    if len(valid_snippets) < len(snippets) * 0.8:  # 80% should have valid text
        return False, "Too many snippets have invalid or very short text"
    
    return True, "Selection is valid"

def display_snippet_preview(snippets, max_preview=3):
    """Display a preview of selected snippets."""
    if not snippets:
        st.info("No snippets to preview")
        return
    
    st.markdown("#### 📝 Snippet Preview")
    
    # Show basic stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Snippets", len(snippets))
    
    with col2:
        companies = len(set(s.ticker for s in snippets))
        st.metric("Companies", companies)
    
    with col3:
        avg_length = sum(len(s.text) for s in snippets) / len(snippets)
        st.metric("Avg Length", f"{avg_length:.0f} chars")
    
    # Show sample snippets
    st.markdown("**Sample Snippets:**")
    
    preview_snippets = snippets[:max_preview]
    
    for i, snippet in enumerate(preview_snippets, 1):
        with st.expander(f"Snippet {i}: {snippet.company} ({snippet.year})"):
            st.write(f"**Text:** {snippet.text}")
            st.write(f"**Speaker:** {snippet.speaker} ({snippet.profession})")
            st.write(f"**Date:** {snippet.date} | **Sentiment:** {snippet.climate_sentiment}")
            
            # Show relevance score if available
            if hasattr(snippet, 'score'):
                st.write(f"**Relevance Score:** {snippet.score:.3f}")
    
    if len(snippets) > max_preview:
        st.caption(f"Showing {max_preview} of {len(snippets)} snippets")

# Legacy wrapper functions for backward compatibility
def render_evolution_analysis_tab(rag):
    """Evolution analysis tab - delegated to existing implementation."""
    # Import here to avoid circular dependencies
    from src.ui_components_legacy import render_evolution_analysis_tab as legacy_evolution
    legacy_evolution(rag)

# Simple error handling decorator
def handle_ui_errors(func):
    """Decorator to handle common UI errors gracefully."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"❌ Error in {func.__name__}: {str(e)}")
            st.info("💡 Try refreshing the page or checking your inputs")
            if st.checkbox("Show technical details", key=f"error_details_{func.__name__}"):
                st.exception(e)
    return wrapper