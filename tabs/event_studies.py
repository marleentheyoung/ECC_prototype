# tabs/event_studies.py - Simplified and cleaned event studies functionality
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
from typing import Dict, List

def render_event_studies_tab(rag):
    """Main event studies tab - simplified and streamlined."""
    st.header("📅 Event Studies")
    st.markdown("*Analyze how climate policy events affect firm discussions in earnings calls*")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load market data first using the sidebar")
        return
    
    # Simple 3-step process
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 1️⃣ Select Events")
        selected_events = render_event_selection()
    
    with col2:
        st.markdown("### 2️⃣ Configure Search")
        search_config = render_search_configuration()
    
    with col3:
        st.markdown("### 3️⃣ Run Analysis")
        if st.button("🚀 Analyze Events", type="primary", disabled=not selected_events):
            execute_event_analysis(rag, selected_events, search_config)

def render_event_selection():
    """Simplified event selection."""
    predefined_events = {
        "Paris Agreement": "2015-12-12",
        "Trump Election": "2016-11-08", 
        "Biden Election": "2020-11-07",
        "EU Green Deal": "2019-12-11",
        "US IRA Passage": "2022-08-16",
        "COP21": "2015-11-30",
        "US Paris Withdrawal": "2017-06-01",
        "US Paris Re-entry": "2021-01-20"
    }
    
    selected_events = st.multiselect(
        "Choose events to analyze",
        list(predefined_events.keys()),
        default=["Paris Agreement"],
        help="Select one or more climate policy events"
    )
    
    # Store selected events
    st.session_state.selected_events = {
        name: predefined_events[name] for name in selected_events
    }
    
    # Show selection
    if selected_events:
        st.success(f"✅ {len(selected_events)} event(s) selected")
        for event in selected_events:
            st.caption(f"📅 {event}: {predefined_events[event]}")
    
    return selected_events

def render_search_configuration():
    """Simplified search configuration."""
    st.markdown("#### Search Terms")
    
    # Predefined search queries
    default_queries = [
        "climate policy environmental regulation",
        "paris agreement climate accord", 
        "carbon regulation emissions policy",
        "climate legislation regulatory framework"
    ]
    
    selected_queries = st.multiselect(
        "Select search terms",
        default_queries,
        default=default_queries[:2],
        help="Terms to search for in earnings calls"
    )
    
    # Simple threshold setting
    relevance_threshold = st.slider(
        "Search Precision",
        min_value=0.15,
        max_value=0.50,
        value=0.25,
        step=0.05,
        help="Higher = more precise, Lower = more inclusive"
    )
    
    # Time window
    st.markdown("#### Analysis Window")
    days_around_event = st.selectbox(
        "Days around event",
        [30, 60, 90, 180],
        index=2,  # Default to 90 days
        help="How many days before/after event to analyze"
    )
    
    config = {
        'queries': selected_queries,
        'threshold': relevance_threshold,
        'window_days': days_around_event
    }
    
    # Show configuration summary
    if selected_queries:
        st.info(f"🔍 Using {len(selected_queries)} search terms with {relevance_threshold:.2f} threshold")
    
    return config

def execute_event_analysis(rag, selected_events, search_config):
    """Execute simplified event analysis."""
    if not selected_events:
        st.warning("No events selected")
        return
    
    if not search_config['queries']:
        st.warning("No search terms selected")
        return
    
    # Store config for analysis
    st.session_state.event_study_queries = search_config['queries']
    st.session_state.relevance_threshold = search_config['threshold']
    
    with st.spinner("🔍 Running event analysis..."):
        try:
            results = {}
            
            for event_name, event_date in selected_events.items():
                st.info(f"Analyzing: {event_name}")
                
                # Create simple timeline around event
                timeline_data = create_event_timeline(
                    rag, 
                    event_date, 
                    search_config['queries'], 
                    search_config['threshold'],
                    search_config['window_days']
                )
                
                results[event_name] = {
                    'event_date': event_date,
                    'timeline': timeline_data,
                    'config': search_config
                }
            
            if results:
                display_event_results(results)
                st.session_state.event_study_results = results
                st.success("✅ Event analysis completed!")
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.info("💡 Try adjusting search terms or threshold")

def create_event_timeline(rag, event_date, search_queries, threshold, window_days):
    """Create timeline of mentions around an event."""
    # Convert event date
    event_dt = pd.to_datetime(event_date)
    start_date = event_dt - pd.Timedelta(days=window_days)
    end_date = event_dt + pd.Timedelta(days=window_days)
    
    # Get relevant snippets using semantic search
    all_snippets = []
    
    for query in search_queries:
        search_results = rag.query_embedding_index(
            query, 
            top_k=None,
            relevance_threshold=threshold
        )
        
        # Convert to snippets and filter by date
        for result in search_results:
            for snippet in rag.snippets:
                if (snippet.text == result['text'] and 
                    snippet.company == result['company'] and
                    snippet.ticker == result['ticker']):
                    
                    # Check if snippet is in time window
                    try:
                        snippet_date = pd.to_datetime(snippet.date)
                        if start_date <= snippet_date <= end_date:
                            snippet.relevance_score = result['score']
                            all_snippets.append(snippet)
                    except:
                        continue  # Skip invalid dates
                    break
    
    # Remove duplicates
    unique_snippets = []
    seen = set()
    for snippet in all_snippets:
        key = f"{snippet.text}_{snippet.company}_{snippet.date}"
        if key not in seen:
            seen.add(key)
            unique_snippets.append(snippet)
    
    # Group by week for timeline
    weekly_groups = {}
    
    for snippet in unique_snippets:
        snippet_date = pd.to_datetime(snippet.date)
        
        # Get week start (Monday)
        week_start = snippet_date - pd.Timedelta(days=snippet_date.weekday())
        week_key = week_start.strftime('%Y-%m-%d')
        
        if week_key not in weekly_groups:
            weekly_groups[week_key] = {
                'mentions': 0,
                'companies': set(),
                'date': week_start,
                'days_from_event': (week_start - event_dt).days
            }
        
        weekly_groups[week_key]['mentions'] += 1
        weekly_groups[week_key]['companies'].add(snippet.ticker)
    
    # Convert to timeline data
    timeline_data = []
    for week_key, data in weekly_groups.items():
        timeline_data.append({
            'date': data['date'].strftime('%Y-%m-%d'),
            'mentions': data['mentions'],
            'companies': len(data['companies']),
            'days_from_event': data['days_from_event']
        })
    
    # Sort by date
    timeline_data.sort(key=lambda x: x['date'])
    
    return timeline_data

def display_event_results(results):
    """Display simplified event results."""
    st.markdown("---")
    st.subheader("📊 Event Analysis Results")
    
    # Single event or multiple events
    if len(results) == 1:
        event_name, event_data = list(results.items())[0]
        display_single_event(event_name, event_data)
    else:
        # Create tabs for multiple events
        event_names = list(results.keys())
        event_tabs = st.tabs(event_names)
        
        for i, (event_name, event_data) in enumerate(results.items()):
            with event_tabs[i]:
                display_single_event(event_name, event_data)
        
        # Multi-event comparison
        st.markdown("---")
        st.subheader("🔄 Event Comparison")
        display_event_comparison(results)

def display_single_event(event_name, event_data):
    """Display results for a single event."""
    st.subheader(f"📈 {event_name}")
    
    timeline = event_data['timeline']
    event_date = event_data['event_date']
    
    if not timeline:
        st.warning("No relevant mentions found around this event")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(timeline)
    df['date'] = pd.to_datetime(df['date'])
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_mentions = df['mentions'].sum()
        st.metric("Total Mentions", total_mentions)
    
    with col2:
        peak_mentions = df['mentions'].max()
        st.metric("Peak Weekly Mentions", peak_mentions)
    
    with col3:
        unique_companies = df['companies'].sum()
        st.metric("Company Mentions", unique_companies)
    
    # Timeline chart
    fig = px.line(
        df,
        x='date',
        y='mentions',
        title=f"Climate Discussion Timeline - {event_name}",
        labels={'date': 'Date', 'mentions': 'Weekly Mentions'}
    )
    
    # Add event date marker
    event_dt = pd.to_datetime(event_date)
    fig.add_vline(
        x=event_dt, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Event: {event_name}"
    )
    
    fig.update_traces(line=dict(color='blue', width=2))
    fig.update_layout(height=400, hovermode='x unified')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show peak period info
    peak_week = df.loc[df['mentions'].idxmax()]
    peak_date = peak_week['date'].strftime('%Y-%m-%d')
    days_from_event = peak_week['days_from_event']
    
    if abs(days_from_event) <= 14:  # Within 2 weeks
        st.success(f"🎯 Peak activity was {abs(days_from_event)} days from event ({peak_date})")
    else:
        st.info(f"📅 Peak activity was {abs(days_from_event)} days from event ({peak_date})")

def display_event_comparison(results):
    """Simple comparison across multiple events."""
    comparison_data = []
    
    for event_name, event_data in results.items():
        timeline = event_data['timeline']
        
        if timeline:
            df = pd.DataFrame(timeline)
            total_mentions = df['mentions'].sum()
            peak_mentions = df['mentions'].max()
            
            # Find peak timing relative to event
            peak_idx = df['mentions'].idxmax()
            peak_timing = df.iloc[peak_idx]['days_from_event']
            
        else:
            total_mentions = peak_mentions = peak_timing = 0
        
        comparison_data.append({
            'Event': event_name,
            'Total Mentions': total_mentions,
            'Peak Mentions': peak_mentions,
            'Peak Timing (days)': peak_timing,
            'Event Date': event_data['event_date']
        })
    
    # Comparison table
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Total Mentions', ascending=False)
    
    st.dataframe(comparison_df, use_container_width=True)
    
    # Simple comparison chart
    if len(comparison_data) > 1:
        fig = px.bar(
            comparison_df,
            x='Event',
            y='Total Mentions',
            title="Total Mentions by Event",
            color='Total Mentions',
            color_continuous_scale='viridis'
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Export option
    csv_data = comparison_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Comparison Data",
        data=csv_data,
        file_name="event_comparison.csv",
        mime="text/csv"
    )