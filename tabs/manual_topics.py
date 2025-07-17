# tabs/manual_topics.py - Simplified manual topic identification
import streamlit as st
import pandas as pd
from typing import Dict, List

def render_manual_topics_tab(rag):
    """Simplified manual topic identification tab."""
    st.header("📝 Manual Topic Identification")
    st.markdown("*Define custom topics using semantic search*")
    
    # Two-column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_topic_creation_form(rag)
    
    with col2:
        render_current_topics_list()
    
    # Display results if topics exist
    if st.session_state.get('manual_topics'):
        st.markdown("---")
        display_topics_overview()

def render_topic_creation_form(rag):
    """Form to create new manual topics."""
    st.subheader("➕ Create New Topic")
    
    with st.form("create_topic_form"):
        # Topic details
        topic_name = st.text_input(
            "Topic Name",
            placeholder="e.g., Paris Agreement, Carbon Pricing, EU ETS",
            help="Give your topic a descriptive name"
        )
        
        search_query = st.text_area(
            "Search Description",
            placeholder="Enter words and phrases that describe this topic...",
            help="Describe what you're looking for in natural language",
            height=100
        )
        
        # Search settings
        col_a, col_b = st.columns(2)
        
        with col_a:
            relevance_threshold = st.slider(
                "Search Precision",
                min_value=0.15,
                max_value=0.60,
                value=0.30,
                step=0.05,
                help="Higher = more precise results"
            )
        
        with col_b:
            use_llm_validation = st.checkbox(
                "Validate with AI",
                value=True,
                help="Use AI to check if results are actually about your topic"
            )
        
        # Submit button
        submitted = st.form_submit_button("🔍 Create Topic", type="primary")
        
        if submitted:
            if topic_name and search_query:
                create_manual_topic(rag, topic_name, search_query, relevance_threshold, use_llm_validation)
            else:
                st.error("Please provide both topic name and search description")

def create_manual_topic(rag, topic_name, search_query, threshold, use_validation):
    """Create a new manual topic."""
    # Check for duplicate names
    if topic_name in st.session_state.get('manual_topics', {}):
        st.error(f"Topic '{topic_name}' already exists. Please choose a different name.")
        return
    
    with st.spinner(f"Searching for '{topic_name}'..."):
        try:
            # Perform semantic search
            search_results = rag.query_embedding_index(
                search_query,
                top_k=None,
                relevance_threshold=threshold
            )
            
            # Convert to snippets
            snippets = []
            for result in search_results:
                for snippet in rag.snippets:
                    if (snippet.text == result['text'] and 
                        snippet.company == result['company'] and
                        snippet.ticker == result['ticker']):
                        snippet.score = result['score']
                        snippets.append(snippet)
                        break
            
            # Optional LLM validation
            if use_validation and st.session_state.get('anthropic_api_key') and snippets:
                st.info("🤖 Validating results with AI...")
                from src.topic_search import validate_topic_relevance_with_llm
                snippets = validate_topic_relevance_with_llm(
                    snippets, topic_name, search_query, st.session_state.anthropic_api_key
                )
            
            # Store topic
            if 'manual_topics' not in st.session_state:
                st.session_state.manual_topics = {}
            
            st.session_state.manual_topics[topic_name] = {
                'query': search_query,
                'threshold': threshold,
                'validated': use_validation,
                'snippets': snippets,
                'created_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
            }
            
            st.success(f"✅ Created '{topic_name}' with {len(snippets)} relevant snippets!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error creating topic: {str(e)}")

def render_current_topics_list():
    """Display list of current manual topics."""
    st.subheader("📋 Your Topics")
    
    manual_topics = st.session_state.get('manual_topics', {})
    
    if not manual_topics:
        st.info("No topics created yet")
        st.markdown("👈 Use the form to create your first topic")
        return
    
    # Display each topic
    for topic_name, topic_data in manual_topics.items():
        with st.expander(f"**{topic_name}**", expanded=False):
            st.write(f"**Query:** {topic_data['query']}")
            st.write(f"**Snippets:** {len(topic_data['snippets'])}")
            st.write(f"**Threshold:** {topic_data['threshold']:.2f}")
            st.write(f"**AI Validated:** {'Yes' if topic_data.get('validated') else 'No'}")
            st.caption(f"Created: {topic_data.get('created_at', 'Unknown')}")
            
            # Actions
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button(f"📊 View Results", key=f"view_{topic_name}"):
                    st.session_state.selected_topic_for_view = topic_name
            
            with col_b:
                if st.button(f"🗑️ Delete", key=f"delete_{topic_name}"):
                    delete_topic(topic_name)
                    st.rerun()

def delete_topic(topic_name):
    """Delete a manual topic."""
    if 'manual_topics' in st.session_state and topic_name in st.session_state.manual_topics:
        del st.session_state.manual_topics[topic_name]
        st.success(f"Deleted topic: {topic_name}")

def display_topics_overview():
    """Display overview of all manual topics."""
    st.subheader("📊 Topics Overview")
    
    manual_topics = st.session_state.get('manual_topics', {})
    
    # Create summary data
    summary_data = []
    for topic_name, topic_data in manual_topics.items():
        snippets = topic_data['snippets']
        
        # Calculate metrics
        total_snippets = len(snippets)
        unique_companies = len(set(s.ticker for s in snippets)) if snippets else 0
        avg_relevance = sum(getattr(s, 'score', 0.5) for s in snippets) / max(total_snippets, 1)
        
        # Sentiment breakdown
        sentiment_counts = {'opportunity': 0, 'neutral': 0, 'risk': 0}
        for snippet in snippets:
            sentiment = getattr(snippet, 'climate_sentiment', 'neutral')
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
        
        # Year range
        years = [int(s.year) for s in snippets if s.year and str(s.year).isdigit()]
        year_range = f"{min(years)}-{max(years)}" if years else "N/A"
        
        summary_data.append({
            'Topic': topic_name,
            'Snippets': total_snippets,
            'Companies': unique_companies,
            'Avg Relevance': f"{avg_relevance:.3f}",
            'Opportunity': sentiment_counts['opportunity'],
            'Neutral': sentiment_counts['neutral'],
            'Risk': sentiment_counts['risk'],
            'Year Range': year_range,
            'Threshold': f"{topic_data['threshold']:.2f}",
            'AI Validated': '✅' if topic_data.get('validated') else '❌'
        })
    
    # Display summary table
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_topics = len(summary_data)
            st.metric("Total Topics", total_topics)
        
        with col2:
            total_snippets = sum(row['Snippets'] for row in summary_data)
            st.metric("Total Snippets", total_snippets)
        
        with col3:
            validated_topics = sum(1 for row in summary_data if row['AI Validated'] == '✅')
            st.metric("AI Validated", f"{validated_topics}/{total_topics}")
        
        with col4:
            avg_snippets = total_snippets / max(total_topics, 1)
            st.metric("Avg Snippets/Topic", f"{avg_snippets:.1f}")
        
        # Export functionality
        st.markdown("#### 📥 Export Topics")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            # Export summary
            summary_csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📋 Download Summary CSV",
                data=summary_csv,
                file_name="manual_topics_summary.csv",
                mime="text/csv"
            )
        
        with col_b:
            # Export all snippets
            if st.button("📄 Export All Snippets"):
                export_all_snippets(manual_topics)

def export_all_snippets(manual_topics):
    """Export all snippets from all manual topics."""
    all_snippets_data = []
    
    for topic_name, topic_data in manual_topics.items():
        for snippet in topic_data['snippets']:
            all_snippets_data.append({
                'topic_name': topic_name,
                'text': snippet.text,
                'company': snippet.company,
                'ticker': snippet.ticker,
                'year': snippet.year,
                'quarter': snippet.quarter,
                'date': snippet.date,
                'speaker': snippet.speaker,
                'profession': snippet.profession,
                'climate_sentiment': snippet.climate_sentiment,
                'relevance_score': getattr(snippet, 'score', 0.5),
                'search_query': topic_data['query'],
                'threshold_used': topic_data['threshold']
            })
    
    if all_snippets_data:
        snippets_df = pd.DataFrame(all_snippets_data)
        snippets_csv = snippets_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download All Snippets CSV",
            data=snippets_csv,
            file_name="manual_topics_all_snippets.csv",
            mime="text/csv"
        )
        
        st.success(f"✅ Exported {len(all_snippets_data)} snippets from {len(manual_topics)} topics")
    else:
        st.warning("No snippets to export")

# Display individual topic details if selected
def display_topic_details():
    """Display detailed view of a selected topic."""
    selected_topic = st.session_state.get('selected_topic_for_view')
    
    if not selected_topic:
        return
    
    manual_topics = st.session_state.get('manual_topics', {})
    
    if selected_topic not in manual_topics:
        st.error(f"Topic '{selected_topic}' not found")
        return
    
    topic_data = manual_topics[selected_topic]
    snippets = topic_data['snippets']
    
    st.subheader(f"📋 Details: {selected_topic}")
    
    # Topic info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Query:** {topic_data['query']}")
        st.write(f"**Threshold:** {topic_data['threshold']:.2f}")
    
    with col2:
        st.write(f"**Total Snippets:** {len(snippets)}")
        st.write(f"**AI Validated:** {'Yes' if topic_data.get('validated') else 'No'}")
    
    with col3:
        if snippets:
            companies = len(set(s.ticker for s in snippets))
            avg_score = sum(getattr(s, 'score', 0.5) for s in snippets) / len(snippets)
            st.write(f"**Companies:** {companies}")
            st.write(f"**Avg Relevance:** {avg_score:.3f}")
    
    # Show sample snippets
    if snippets:
        st.markdown("#### 📝 Sample Snippets")
        
        # Sort by relevance score
        sorted_snippets = sorted(snippets, key=lambda x: getattr(x, 'score', 0.5), reverse=True)
        
        for i, snippet in enumerate(sorted_snippets[:5]):  # Show top 5
            with st.expander(f"Snippet {i+1}: {snippet.company} (Score: {getattr(snippet, 'score', 0.5):.3f})"):
                st.write(f"**Text:** {snippet.text}")
                st.write(f"**Speaker:** {snippet.speaker} ({snippet.profession})")
                st.write(f"**Date:** {snippet.date} | **Sentiment:** {snippet.climate_sentiment}")
        
        if len(snippets) > 5:
            st.info(f"Showing top 5 snippets. Total: {len(snippets)} snippets found.")
    
    # Clear selection button
    if st.button("← Back to Topics List"):
        if 'selected_topic_for_view' in st.session_state:
            del st.session_state.selected_topic_for_view
        st.rerun()