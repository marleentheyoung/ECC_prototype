# visualization.py - Visualization and charting functions
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def display_topic_distribution(topic_info, topic_names):
    """Display topic distribution chart."""
    st.subheader("📊 Topic Distribution")
    topic_counts = topic_info[topic_info['Topic'] != -1]['Count'].tolist()
    topic_labels = [topic_names.get(t, f"Topic {t}") for t in topic_info[topic_info['Topic'] != -1]['Topic'].tolist()]
    
    fig = px.bar(
        x=topic_labels, 
        y=topic_counts,
        title="Number of Documents per Topic",
        labels={'x': 'Topic', 'y': 'Document Count'}
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

def display_evolution_charts_simple(evolution_data, topic_name, time_granularity):
    """Display evolution charts for the selected topic."""
    
    if not evolution_data:
        st.warning("No data available for visualization.")
        return
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(evolution_data)
    
    st.subheader(f"📊 Evolution of {topic_name}")
    
    # 1. Topic mention frequency over time
    st.markdown("#### Topic Mention Frequency")
    fig = px.line(
        df, 
        x='period', 
        y='count', 
        title=f"Topic Mentions Over Time ({time_granularity})",
        labels={'count': 'Number of Mentions', 'period': 'Time Period'},
        markers=True
    )
    fig.update_layout(
        xaxis_title="Time Period",
        yaxis_title="Number of Mentions",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 2. Company diversity over time
    st.markdown("#### Company Diversity")
    fig2 = px.line(
        df, 
        x='period', 
        y='companies', 
        title=f"Number of Companies Discussing Topic ({time_granularity})",
        labels={'companies': 'Number of Companies', 'period': 'Time Period'},
        markers=True
    )
    fig2.update_layout(
        xaxis_title="Time Period",
        yaxis_title="Number of Companies",
        hovermode='x unified'
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # 3. Sentiment evolution
    st.markdown("#### Sentiment Evolution")
    
    # Prepare sentiment data
    sentiment_data = []
    for _, row in df.iterrows():
        for sentiment in ['opportunity', 'neutral', 'risk']:
            sentiment_data.append({
                'period': row['period'],
                'sentiment': sentiment.title(),
                'count': row[f'sentiment_{sentiment}']
            })
    
    sentiment_df = pd.DataFrame(sentiment_data)
    
    fig3 = px.bar(
        sentiment_df, 
        x='period', 
        y='count', 
        color='sentiment',
        title="Sentiment Evolution Over Time",
        color_discrete_map={'Opportunity': 'green', 'Neutral': 'yellow', 'Risk': 'red'}
    )
    fig3.update_layout(xaxis_title="Time Period", yaxis_title="Count")
    st.plotly_chart(fig3, use_container_width=True)

def display_all_topics_stacked_chart(evolution_data, valid_topics, topic_names, show_eu, show_us, time_granularity):
    """Display stacked bar chart for all topics evolution."""
    
    if not evolution_data:
        st.warning("No data available for all topics visualization.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(evolution_data)
    
    st.subheader("📊 All Topics Evolution - Stacked Bar Chart")
    
    # Create separate charts for EU and US if both are selected
    if show_eu and show_us:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### EU Market")
            eu_columns = [col for col in df.columns if col.startswith('EU_') and col != 'period']
            if eu_columns:
                # Prepare data for EU stacked bar chart
                eu_data = df[['period'] + eu_columns]
                eu_data.columns = ['period'] + [col.replace('EU_', '') for col in eu_columns]
                
                # Melt the dataframe for plotly stacked bar chart
                eu_melted = eu_data.melt(
                    id_vars=['period'], 
                    var_name='Topic', 
                    value_name='Count'
                )
                
                # Create stacked bar chart
                fig_eu = px.bar(
                    eu_melted,
                    x='period',
                    y='Count',
                    color='Topic',
                    title=f"EU Topics Evolution ({time_granularity})",
                    labels={'Count': 'Number of Mentions', 'period': 'Time Period'},
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_eu.update_layout(
                    xaxis_title="Time Period",
                    yaxis_title="Number of Mentions",
                    legend_title="Topics",
                    hovermode='x unified',
                    height=500
                )
                st.plotly_chart(fig_eu, use_container_width=True)
        
        with col2:
            st.markdown("#### US Market")
            us_columns = [col for col in df.columns if col.startswith('US_') and col != 'period']
            if us_columns:
                # Prepare data for US stacked bar chart
                us_data = df[['period'] + us_columns]
                us_data.columns = ['period'] + [col.replace('US_', '') for col in us_columns]
                
                # Melt the dataframe for plotly stacked bar chart
                us_melted = us_data.melt(
                    id_vars=['period'], 
                    var_name='Topic', 
                    value_name='Count'
                )
                
                # Create stacked bar chart
                fig_us = px.bar(
                    us_melted,
                    x='period',
                    y='Count',
                    color='Topic',
                    title=f"US Topics Evolution ({time_granularity})",
                    labels={'Count': 'Number of Mentions', 'period': 'Time Period'},
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_us.update_layout(
                    xaxis_title="Time Period",
                    yaxis_title="Number of Mentions",
                    legend_title="Topics",
                    hovermode='x unified',
                    height=500
                )
                st.plotly_chart(fig_us, use_container_width=True)
    
    else:
        # Single market view
        market_prefix = "EU_" if show_eu else "US_"
        market_name = "EU" if show_eu else "US"
        
        market_columns = [col for col in df.columns if col.startswith(market_prefix) and col != 'period']
        if market_columns:
            # Prepare data for stacked bar chart
            market_data = df[['period'] + market_columns]
            market_data.columns = ['period'] + [col.replace(market_prefix, '') for col in market_data.columns]
            
            # Melt the dataframe for plotly stacked bar chart
            market_melted = market_data.melt(
                id_vars=['period'], 
                var_name='Topic', 
                value_name='Count'
            )
            
            # Create stacked bar chart
            fig = px.bar(
                market_melted,
                x='period',
                y='Count',
                color='Topic',
                title=f"{market_name} Topics Evolution ({time_granularity})",
                labels={'Count': 'Number of Mentions', 'period': 'Time Period'},
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(
                xaxis_title="Time Period",
                yaxis_title="Number of Mentions",
                legend_title="Topics",
                hovermode='x unified',
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)

def display_evolution_insights_simple(evolution_data, topic_name, time_granularity):
    """Display insights and key metrics about topic evolution."""
    
    if not evolution_data:
        return
    
    df = pd.DataFrame(evolution_data)
    
    st.subheader(f"🔍 Key Insights for {topic_name}")
    
    # Calculate key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_mentions = df['count'].sum()
        st.metric("Total Mentions", total_mentions)
    
    with col2:
        total_companies = df['companies'].sum()
        st.metric("Total Companies", total_companies)
    
    with col3:
        avg_mentions_per_period = df['count'].mean()
        st.metric("Avg Mentions/Period", f"{avg_mentions_per_period:.1f}")
    
    with col4:
        # Calculate trend (simple linear trend)
        if len(df) > 1:
            periods = list(range(len(df)))
            mentions = df['count'].values
            if len(mentions) > 1:
                trend = np.polyfit(periods, mentions, 1)[0]
                trend_emoji = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
                st.metric("Trend", f"{trend_emoji} {trend:.1f}")
            else:
                st.metric("Trend", "N/A")
        else:
            st.metric("Trend", "N/A")
    
    # Period-wise breakdown
    st.markdown("#### Period-wise Breakdown")
    period_summary = df.copy()
    period_summary.columns = ['Period', 'Mentions', 'Companies', 'Opportunity', 'Neutral', 'Risk']
    st.dataframe(period_summary, use_container_width=True)

def display_topics_summary(evolution_data, valid_topics, topic_names, show_eu, show_us):
    """Display summary statistics for all topics."""
    st.subheader("📋 Topics Summary")
    
    df = pd.DataFrame(evolution_data)
    
    # Calculate total mentions per topic across all periods
    topic_totals = {}
    for topic_num in valid_topics:
        topic_name = topic_names.get(topic_num, f"Topic {topic_num}")
        total = 0
        
        if show_eu:
            eu_col = f"EU_{topic_name}"
            if eu_col in df.columns:
                total += df[eu_col].sum()
        
        if show_us:
            us_col = f"US_{topic_name}"
            if us_col in df.columns:
                total += df[us_col].sum()
        
        topic_totals[topic_name] = total
    
    # Display as a simple table
    summary_df = pd.DataFrame(list(topic_totals.items()), columns=['Topic', 'Total Mentions'])
    summary_df = summary_df.sort_values('Total Mentions', ascending=False)
    st.dataframe(summary_df, use_container_width=True)