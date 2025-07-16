# simplified_snippet_selection.py - Simplified Tab 1 with keyword and semantic search
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
from src.adaptive_threshold_validation import AdaptiveThresholdValidator, display_threshold_search_results
from src.utils import display_results

def render_simplified_snippet_selection_tab(rag):
    """Simplified snippet selection tab with keyword and semantic search options."""
    st.header("🔍 Snippet Selection")
    st.write("Select snippets for analysis using keyword or semantic search")
    
    # Search method selection
    search_method = st.radio(
        "Search Method:",
        ["Keyword Search", "Semantic Search"],
        help="Choose between exact keyword matching or semantic similarity search"
    )
    
    if search_method == "Keyword Search":
        render_keyword_search_section(rag)
    else:
        render_semantic_search_section(rag)
    
    # Show current selection status
    render_selection_status()

def render_keyword_search_section(rag):
    """Render keyword search section with exact matching."""
    st.subheader("🔤 Keyword Search")
    st.write("Search for exact keyword or phrase matches in earnings call text")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Keywords input
        keywords_input = st.text_area(
            "Keywords/Phrases (comma-separated)",
            placeholder="e.g., Paris Agreement, climate change, carbon neutral, net zero",
            help="Enter keywords or phrases separated by commas. Search will find exact matches (case-insensitive)"
        )
        
        # Search logic options
        search_logic = st.radio(
            "Search Logic:",
            ["Any keyword (OR)", "All keywords (AND)"],
            help="OR: Find snippets containing any of the keywords. AND: Find snippets containing all keywords"
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
    
    if st.button("🔍 Search Keywords", type="primary"):
        if keywords_input.strip():
            perform_keyword_search(rag, keywords_input, search_logic, 
                                 selected_companies, sentiment_filter, year_range)
        else:
            st.warning("Please enter keywords to search for.")

def render_semantic_search_section(rag):
    """Render semantic search section with adaptive threshold validation."""
    st.subheader("🧠 Semantic Search")
    st.write("Search for snippets using semantic similarity with adaptive threshold optimization")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Topic and query input
        topic_name = st.text_input(
            "Topic Name", 
            placeholder="e.g., Paris Agreement, EU Taxonomy, Carbon Pricing"
        )
        
        search_query = st.text_area(
            "Search Query", 
            placeholder="Enter descriptive terms and phrases for semantic search",
            help="Use natural language describing what you're looking for"
        )
        
        # Validation method selection
        validation_method = st.radio(
            "Validation Method:",
            ["Standard Threshold", "Adaptive Threshold Search"],
            help="Choose between fixed threshold or adaptive threshold optimization"
        )
        
        if validation_method == "Standard Threshold":
            # Standard validation options
            col_a, col_b = st.columns(2)
            with col_a:
                relevance_threshold = st.slider(
                    "Relevance Threshold", 
                    min_value=0.1, 
                    max_value=0.8, 
                    value=0.30, 
                    step=0.05
                )
            with col_b:
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
    
    if st.button("🧠 Search Semantically", type="primary"):
        if topic_name.strip() and search_query.strip():
            if validation_method == "Standard Threshold":
                perform_semantic_search_standard(rag, topic_name, search_query, 
                                                relevance_threshold, use_llm_validation,
                                                selected_companies, sentiment_filter, year_range)
            else:
                perform_semantic_search_adaptive(rag, topic_name, search_query, 
                                                initial_threshold, quality_threshold,
                                                selected_companies, sentiment_filter, year_range)
        else:
            st.warning("Please enter both topic name and search query.")

def perform_keyword_search(rag, keywords_input, search_logic, selected_companies, sentiment_filter, year_range):
    """Perform keyword search with exact matching."""
    with st.spinner("Searching for keyword matches..."):
        # Parse keywords
        keywords = [kw.strip().lower() for kw in keywords_input.split(",") if kw.strip()]
        
        if not keywords:
            st.warning("No valid keywords found.")
            return
        
        # Search through snippets
        matching_snippets = []
        
        for snippet in rag.snippets:
            text_lower = snippet.text.lower()
            
            # Apply year filter first
            if year_range and snippet.year:
                try:
                    year = int(snippet.year)
                    if not (year_range[0] <= year <= year_range[1]):
                        continue
                except (ValueError, TypeError):
                    continue
            
            # Apply company filter
            if selected_companies and snippet.ticker not in selected_companies:
                continue
            
            # Apply sentiment filter
            if sentiment_filter != "All" and snippet.climate_sentiment != sentiment_filter:
                continue
            
            # Check keyword matches
            keyword_matches = [kw for kw in keywords if kw in text_lower]
            
            if search_logic == "Any keyword (OR)":
                if keyword_matches:  # At least one keyword found
                    matching_snippets.append({
                        'snippet': snippet,
                        'matched_keywords': keyword_matches,
                        'match_count': len(keyword_matches)
                    })
            else:  # "All keywords (AND)"
                if len(keyword_matches) == len(keywords):  # All keywords found
                    matching_snippets.append({
                        'snippet': snippet,
                        'matched_keywords': keyword_matches,
                        'match_count': len(keyword_matches)
                    })
        
        # Sort by number of matches (descending)
        matching_snippets.sort(key=lambda x: x['match_count'], reverse=True)
        
        if matching_snippets:
            # Store in session state
            snippets_only = [ms['snippet'] for ms in matching_snippets]
            st.session_state.selected_snippets = snippets_only
            st.session_state.selection_method = f"Keyword search: {keywords_input}"
            
            st.success(f"✅ Found {len(matching_snippets)} snippets matching your keywords!")
            
            # Display summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Matches", len(matching_snippets))
            with col2:
                unique_companies = len(set(ms['snippet'].ticker for ms in matching_snippets))
                st.metric("Companies", unique_companies)
            with col3:
                avg_matches = sum(ms['match_count'] for ms in matching_snippets) / len(matching_snippets)
                st.metric("Avg Keywords/Snippet", f"{avg_matches:.1f}")
            
            # Show preview of top results
            st.subheader("📝 Top Results Preview")
            for i, match in enumerate(matching_snippets[:5]):
                snippet = match['snippet']
                matched_kw = ", ".join(match['matched_keywords'])
                
                with st.expander(f"Result {i+1}: {snippet.company} ({match['match_count']} keywords)"):
                    st.write(f"**Matched Keywords:** {matched_kw}")
                    st.write(f"**Text:** {snippet.text}")
                    st.write(f"**Speaker:** {snippet.speaker} ({snippet.profession})")
                    st.write(f"**Date:** {snippet.date} | **Sentiment:** {snippet.climate_sentiment}")
            
            if len(matching_snippets) > 5:
                st.info(f"Showing top 5 results. Total: {len(matching_snippets)} snippets selected.")
            
            st.info("💡 Go to other tabs to analyze these snippets or track their evolution over time.")
            
        else:
            st.warning("No snippets found matching your keyword criteria. Try:")
            st.write("- Using different or broader keywords")
            st.write("- Switching to 'Any keyword (OR)' logic")
            st.write("- Adjusting your filters")

def perform_semantic_search_standard(rag, topic_name, search_query, relevance_threshold, 
                                   use_llm_validation, selected_companies, sentiment_filter, year_range):
    """Perform standard semantic search."""
    with st.spinner("Performing semantic search..."):
        # Perform semantic search
        results = rag.query_embedding_index(
            search_query, 
            top_k=None,
            relevance_threshold=relevance_threshold,
            selected_companies=selected_companies if selected_companies else None,
            sentiment_filter=sentiment_filter,
            year_range=year_range
        )
        
        if results:
            # Convert to snippets
            snippets = convert_results_to_snippets(rag, results)
            
            # LLM validation if requested
            if use_llm_validation and st.session_state.get('anthropic_api_key') and snippets:
                st.info("🤖 Validating with LLM...")
                from src.topic_search import validate_topic_relevance_with_llm
                snippets = validate_topic_relevance_with_llm(
                    snippets, topic_name, search_query, st.session_state.anthropic_api_key
                )
            
            # Store in session state
            st.session_state.selected_snippets = snippets
            st.session_state.selection_method = f"Semantic search: {topic_name}"
            
            st.success(f"✅ Found {len(snippets)} semantically similar snippets!")
            
            # Show preview
            display_results(results[:5])
            if len(results) > 5:
                st.info(f"Showing first 5 results. Total: {len(results)} snippets selected.")
            
        else:
            st.warning("No results found. Try lowering the relevance threshold or adjusting your query.")

def perform_semantic_search_adaptive(rag, topic_name, search_query, initial_threshold, 
                                   quality_threshold, selected_companies, sentiment_filter, year_range):
    """Perform adaptive semantic search."""
    if not st.session_state.get('anthropic_api_key'):
        st.error("❌ Anthropic API key required for adaptive validation!")
        return
    
    st.info("🚀 Starting adaptive threshold search...")
    
    # Initialize validator
    validator = AdaptiveThresholdValidator(st.session_state.anthropic_api_key)
    validator.irrelevant_threshold = quality_threshold
    
    # Instead of creating a filtered RAG, we'll apply filters in post-processing
    try:
        validation_result = validator.adaptive_threshold_search(
            rag, topic_name, search_query, initial_threshold
        )
        
        if validation_result:
            # Apply filters to the final results
            filtered_snippets = apply_snippet_filters(
                validation_result['validated_snippets'], 
                selected_companies, 
                sentiment_filter, 
                year_range
            )
            
            # Store in session state
            st.session_state.selected_snippets = filtered_snippets
            st.session_state.selection_method = f"Adaptive semantic search: {topic_name} (threshold: {validation_result['final_threshold']:.2f})"
            
            # Update the validation result for display
            validation_result['validated_snippets'] = filtered_snippets
            
            # Display results
            display_threshold_search_results(validation_result)
            
            st.success(f"✅ Found {len(filtered_snippets)} high-quality snippets with optimized threshold {validation_result['final_threshold']:.2f}!")
            
            if len(filtered_snippets) < len(validation_result['validated_snippets']):
                original_count = len(validation_result['validated_snippets'])
                st.info(f"📊 Applied filters reduced results from {original_count} to {len(filtered_snippets)} snippets")
            
    except Exception as e:
        st.error(f"❌ Error during adaptive search: {str(e)}")

def apply_snippet_filters(snippets, selected_companies, sentiment_filter, year_range):
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
                snippet.score = result['score']
                selected_snippets.append(snippet)
                break
    return selected_snippets

def render_selection_status():
    """Render current selection status."""
    if st.session_state.get('selected_snippets'):
        st.markdown("---")
        st.subheader("📋 Current Selection")
        st.write(f"**Selected:** {len(st.session_state.selected_snippets)} snippets")
        st.write(f"**Method:** {st.session_state.get('selection_method', 'Unknown')}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Selection"):
                st.session_state.selected_snippets = []
                st.session_state.selection_method = ""
                st.success("Selection cleared!")
        
        with col2:
            st.write("**Next Steps:**")
            st.write("- Go to **Topic Analysis** for topic modeling")
            st.write("- Go to **Evolution Analysis** to track trends")

# Integration function for main interface
def render_simplified_snippet_selection(rag):
    """Main function to call from interface.py"""
    render_simplified_snippet_selection_tab(rag)

# You'll also need this helper function - add to simplified_snippet_selection.py or utils.py
def analyze_snippets_evolution(snippets, topic_name, show_eu, show_us, year_range, time_granularity):
    """Analyze evolution of a specific set of snippets over time."""
    from src.utils import determine_market
    
    # Filter snippets by market and year range
    relevant_snippets = []
    current_market = st.session_state.current_market
    
    for snippet in snippets:
        if snippet.year and str(snippet.year).isdigit():
            year = int(snippet.year)
            if year_range[0] <= year <= year_range[1]:
                # Determine market
                if current_market == "Full":
                    market = determine_market(snippet)
                    if (market == 'EU' and show_eu) or (market == 'US' and show_us):
                        relevant_snippets.append(snippet)
                elif current_market == "Combined":
                    # For combined data, we need to infer the market
                    market = determine_market(snippet)
                    if (market == 'EU' and show_eu) or (market == 'US' and show_us):
                        relevant_snippets.append(snippet)
                else:
                    # Single market
                    if (current_market == "EU" and show_eu) or (current_market == "US" and show_us):
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