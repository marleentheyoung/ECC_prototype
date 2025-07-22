# tabs/event_studies.py - Ultra-simplified event studies
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def render_event_studies_tab(rag):
    """Ultra-simplified event studies tab."""
    st.header("📅 Event Studies")
    st.markdown("*Analyze how climate policy events affect firm discussions in earnings calls*")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load market data first using the sidebar")
        return
    
    # Simple form layout
    with st.form("event_analysis_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📅 Select Events")
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
            
            selected_event_names = st.multiselect(
                "Choose events",
                list(predefined_events.keys()),
                default=["Paris Agreement", "EU Green Deal"]
            )
            
            # Time aggregation
            time_aggregation = st.selectbox(
                "Time Aggregation",
                ["Weekly", "Monthly", "Quarterly"],
                index=0
            )
        
        with col2:
            st.subheader("🔍 Search Query")
            search_query = st.text_area(
                "Enter search terms",
                placeholder="e.g., climate policy regulatory framework, sustainability legislation...",
                height=100
            )
            
            threshold = st.slider(
                "Search Precision",
                0.15, 0.50, 0.30, 0.05
            )
            
            # Time window
            window_days = st.selectbox(
                "Analysis Window (days)",
                [90, 180, 365, 730],
                index=1
            )
            
            # Timeline scope option
            timeline_scope = st.radio(
                "📊 Timeline Scope",
                ["Around Events Only", "Full Timeline"],
                index=0,
                help="Show full timeline or just focus around selected events"
            )
        
        submitted = st.form_submit_button("🚀 Analyze Events", type="primary")
    
    # Store timeline scope in session state OUTSIDE the form
    if submitted:
        st.session_state.timeline_scope = timeline_scope
        
        if search_query and selected_event_names:
            execute_simple_event_analysis(rag, selected_event_names, predefined_events,
                                        search_query, threshold, window_days, time_aggregation)
            
def execute_simple_event_analysis(rag, selected_event_names, predefined_events, 
                                search_query, threshold, window_days, time_aggregation):
    """Execute simplified event analysis."""
    
    timeline_scope = st.session_state.get('timeline_scope', "Around Events Only")
    
    with st.spinner("🔍 Analyzing events..."):
        # Get all relevant snippets
        search_results = rag.query_embedding_index(
            search_query, 
            top_k=None,
            relevance_threshold=threshold
        )
        
        # Convert to snippets with dates
        relevant_snippets = []
        for result in search_results:
            for snippet in rag.snippets:
                if (snippet.text == result['text'] and 
                    snippet.company == result['company'] and
                    snippet.ticker == result['ticker']):
                    
                    try:
                        snippet_date = pd.to_datetime(snippet.date)
                        snippet.parsed_date = snippet_date
                        snippet.relevance_score = result['score']
                        relevant_snippets.append(snippet)
                    except:
                        continue
                    break
        
        if not relevant_snippets:
            st.warning("No relevant snippets found. Try lowering the threshold or changing your query.")
            return
        
        # Create timeline
        timeline_data = create_simple_timeline(relevant_snippets, time_aggregation, timeline_scope, selected_event_names, predefined_events)
        
        # Add normalization option
        normalize_data = st.checkbox("📊 Normalize by total mentions", value=True, help="Show climate mentions as % of total mentions in each period")
        
        if normalize_data:
            timeline_data = normalize_timeline_data(timeline_data, rag, time_aggregation)
        
        # Display results
        display_simple_results(timeline_data, selected_event_names, predefined_events, 
                             search_query, time_aggregation, normalize_data)
        
def create_simple_timeline(snippets, time_aggregation, timeline_scope="Around Events Only", selected_event_names=None, predefined_events=None):
    """Create simple timeline data with datetime x-axis."""
    
    # Filter by timeline scope if needed
    if timeline_scope == "Around Events Only" and selected_event_names and predefined_events:
        # Filter to ±2 years around events
        filtered_snippets = []
        for snippet in snippets:
            try:
                snippet_date = snippet.parsed_date
                
                # Check if within 2 years of any event
                within_range = False
                for event_name in selected_event_names:
                    event_date = pd.to_datetime(predefined_events[event_name])
                    # Fix: Use timedelta from datetime module instead
                    if abs((snippet_date - event_date).days) <= 730:  # 2 years = 730 days
                        within_range = True
                        break
                
                if within_range:
                    filtered_snippets.append(snippet)
            except:
                continue
        
        snippets = filtered_snippets
    
    # Group by time period
    time_groups = {}
    
    # Group by time period
    time_groups = {}
    
    for snippet in snippets:
        date = snippet.parsed_date
        
        # Create period key AND period datetime
        if time_aggregation == "Weekly":
            from datetime import timedelta
            week_start = date - timedelta(days=date.weekday())
            period_key = week_start.strftime('%Y-W%U')
            period_datetime = week_start
        elif time_aggregation == "Monthly":
            month_start = date.replace(day=1)
            period_key = month_start.strftime('%Y-%m')
            period_datetime = month_start
        else:  # Quarterly
            quarter = (date.month - 1) // 3 + 1
            quarter_start = date.replace(month=((quarter-1)*3 + 1), day=1)
            period_key = f"{date.year}-Q{quarter}"
            period_datetime = quarter_start
        
        if period_key not in time_groups:
            time_groups[period_key] = {
                'mentions': 0,
                'companies': set(),
                'period_datetime': period_datetime,
                'period_key': period_key
            }
        
        time_groups[period_key]['mentions'] += 1
        time_groups[period_key]['companies'].add(snippet.ticker)
    
    # Convert to list with datetime objects
    timeline_data = []
    for period_key, data in sorted(time_groups.items()):
        timeline_data.append({
            'period': period_key,
            'period_datetime': data['period_datetime'],
            'mentions': data['mentions'],
            'companies': len(data['companies'])
        })
    
    return timeline_data

def normalize_timeline_data(timeline_data, rag, time_aggregation):
    """Normalize mentions by total mentions in each time period."""
    
    # Get total mentions for each period from the entire dataset
    total_mentions_by_period = {}
    
    for snippet in rag.snippets:
        if not snippet.date or not snippet.year:
            continue
            
        try:
            date = pd.to_datetime(snippet.date)
            
            # Create the same period key as in timeline
            if time_aggregation == "Weekly":
                from datetime import timedelta
                week_start = date - timedelta(days=date.weekday())
                period_key = week_start.strftime('%Y-W%U')
            elif time_aggregation == "Monthly":
                month_start = date.replace(day=1)
                period_key = month_start.strftime('%Y-%m')
            else:  # Quarterly
                quarter = (date.month - 1) // 3 + 1
                quarter_start = date.replace(month=((quarter-1)*3 + 1), day=1)
                period_key = f"{date.year}-Q{quarter}"
            
            if period_key not in total_mentions_by_period:
                total_mentions_by_period[period_key] = 0
            total_mentions_by_period[period_key] += 1
            
        except:
            continue
    
    # Normalize the timeline data
    normalized_data = []
    for item in timeline_data:
        period_key = item['period']
        total_mentions = total_mentions_by_period.get(period_key, 1)  # Avoid division by zero
        
        normalized_item = item.copy()
        normalized_item['mentions'] = (item['mentions'] / total_mentions) * 100  # As percentage
        normalized_item['total_mentions'] = total_mentions  # Keep for reference
        normalized_data.append(normalized_item)
    
    return normalized_data

def display_simple_results(timeline_data, selected_event_names, predefined_events, 
                         search_query, time_aggregation, normalized=False):
    """Display simple results using px.line for both timeline and event lines."""
    
    st.subheader("📊 Event Timeline Results")
    
    if not timeline_data:
        st.warning("No timeline data to display")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(timeline_data)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Mentions", df['mentions'].sum())
    with col2:
        st.metric(f"Peak {time_aggregation} Mentions", df['mentions'].max())
    with col3:
        st.metric("Total Companies", df['companies'].sum())
    
    # Create combined data for plotting
    plot_data = []
    
    # Add timeline data
    for _, row in df.iterrows():
        plot_data.append({
            'date': row['period_datetime'],
            'value': row['mentions'],
            'type': 'Timeline',
            'label': f"{row['mentions']} mentions"
        })
    
    # Add event lines as vertical data points
    max_mentions = df['mentions'].max()
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, event_name in enumerate(selected_event_names):
        try:
            event_date_str = predefined_events[event_name]
            event_date_dt = pd.to_datetime(event_date_str)
                        
            # Add event as vertical line data (create points from 0 to max)
            for y_val in [0, max_mentions]:
                plot_data.append({
                    'date': event_date_dt,
                    'value': y_val,
                    'type': f'Event: {event_name}',
                    'label': event_name
                })
                
        except Exception as e:
            st.warning(f"Could not process event {event_name}: {e}")
    
    # Convert to DataFrame for plotting
    plot_df = pd.DataFrame(plot_data)
    
    # Update labels based on normalization
    if normalized:
        y_label = f'{time_aggregation} Mentions (%)'
        title = f"Climate Discussion Timeline - Normalized ({time_aggregation})"
    else:
        y_label = f'{time_aggregation} Mentions'
        title = f"Climate Discussion Timeline ({time_aggregation})"
    
    # Create the plot with updated labels
    fig = px.line(
        plot_df,
        x='date',
        y='value',
        color='type',
        title=title,
        labels={'date': 'Date', 'value': y_label},
        hover_data=['label']
    )
    
    # [rest of the function stays the same...]

    # Style the timeline differently from event lines
    fig.update_traces(
        line=dict(width=3),
        selector=dict(name='Timeline')
    )

    # Make event lines dashed
    for i, event_name in enumerate(selected_event_names):
        fig.update_traces(
            line=dict(dash='dash', width=2),
            selector=dict(name=f'Event: {event_name}')
        )

    # ADD TEXT ANNOTATIONS FOR EVENTS
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    for i, event_name in enumerate(selected_event_names):
        try:
            event_date_str = predefined_events[event_name]
            event_date_dt = pd.to_datetime(event_date_str)
            color = colors[i % len(colors)]
            
            # Add text annotation at the top of the event line
            fig.add_annotation(
                x=event_date_dt,
                y=max_mentions * 1.1,  # Place slightly above the maximum value
                text=event_name,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=color,
                font=dict(
                    size=10,
                    color=color
                ),
                bgcolor="rgba(255, 255, 255, 0.8)",  # Semi-transparent white background
                bordercolor=color,
                borderwidth=1,
                borderpad=3,
                textangle=0  # Keep text horizontal
            )
            
        except Exception as e:
            st.warning(f"Could not add annotation for {event_name}: {e}")

    fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title=f"{time_aggregation} Mentions",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show events as a simple table
    st.subheader("📅 Selected Events")
    event_data = []
    for event_name in selected_event_names:
        event_data.append({
            'Event': event_name,
            'Date': predefined_events[event_name]
        })
    
    event_df = pd.DataFrame(event_data)
    st.dataframe(event_df, use_container_width=True)
    
    # Show search info
    st.info(f"🔍 Search query: '{search_query}' | {len(selected_event_names)} events analyzed")
    
    # Export option
    export_df = df[['period', 'mentions', 'companies']].copy()
    export_df.columns = ['Time_Period', 'Mentions', 'Companies']
    
    csv_data = export_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Timeline Data",
        data=csv_data,
        file_name="event_timeline.csv",
        mime="text/csv"
    )