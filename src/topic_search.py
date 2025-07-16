# topic_search.py - Topic search and manual topic identification functions
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, List, Tuple
from anthropic import Anthropic

def perform_topic_search(rag, search_queries: Dict[str, str], relevance_threshold: float = 0.30) -> Dict[str, List]:
    """
    Perform semantic search for multiple topics and return results.
    
    Args:
        rag: RAG system instance
        search_queries: Dict mapping topic names to search queries
        relevance_threshold: Minimum relevance score for inclusion
    
    Returns:
        Dict mapping topic names to lists of matching snippets
    """
    topic_results = {}
    
    for topic_name, query in search_queries.items():
        try:
            # Perform semantic search
            results = rag.query_embedding_index(
                query, 
                top_k=None,  # Get all results
                relevance_threshold=relevance_threshold
            )
            
            # Convert results to snippet format for consistency
            snippets = []
            for result in results:
                # Find corresponding snippet object
                for snippet in rag.snippets:
                    if (snippet.text == result['text'] and 
                        snippet.company == result['company'] and
                        snippet.ticker == result['ticker']):
                        snippet.score = result['score']  # Add relevance score
                        snippets.append(snippet)
                        break
            
            topic_results[topic_name] = snippets
            
        except Exception as e:
            st.error(f"Error searching for topic '{topic_name}': {str(e)}")
            topic_results[topic_name] = []
    
    return topic_results

def validate_topic_relevance_with_llm(snippets: List, topic_name: str, query: str, api_key: str = None) -> List:
    """
    Use LLM to validate if snippets are actually relevant to the topic.
    
    Args:
        snippets: List of snippet objects
        topic_name: Name of the topic
        query: Original search query
        api_key: Anthropic API key
    
    Returns:
        List of validated snippets with validation scores
    """
    if not api_key:
        st.warning("No API key provided for LLM validation. Skipping validation step.")
        return snippets
    
    try:
        client = Anthropic(api_key=api_key)
        validated_snippets = []
        
        # Process snippets in batches to avoid rate limits
        batch_size = 5
        
        for i in range(0, len(snippets), batch_size):
            batch = snippets[i:i+batch_size]
            
            for snippet in batch:
                prompt = f"""You are analyzing earnings call transcripts for climate investment research.

Topic: {topic_name}
Search Query: {query}

Text to evaluate:
"{snippet.text}"

Question: Is this text actually discussing or related to the topic "{topic_name}"? 

Consider:
1. Does the text mention concepts related to {topic_name}?
2. Is the discussion substantive (not just passing mention)?
3. Does it provide meaningful information about {topic_name}?

Respond with only:
- "RELEVANT" if the text is clearly about this topic
- "PARTIALLY_RELEVANT" if it mentions the topic but is not the main focus
- "NOT_RELEVANT" if it's not actually about this topic

Your response:"""

                try:
                    response = client.messages.create(
                        model="claude-3-5-sonnet-20240620",
                        max_tokens=50,
                        temperature=0.1,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    
                    validation = response.content[0].text.strip()
                    
                    if validation in ["RELEVANT", "PARTIALLY_RELEVANT"]:
                        snippet.llm_validation = validation
                        snippet.llm_confidence = 1.0 if validation == "RELEVANT" else 0.7
                        validated_snippets.append(snippet)
                    
                except Exception as e:
                    st.warning(f"Error validating snippet: {str(e)}")
                    # If validation fails, include the snippet anyway
                    snippet.llm_validation = "VALIDATION_ERROR"
                    snippet.llm_confidence = snippet.score  # Use original similarity score
                    validated_snippets.append(snippet)
        
        return validated_snippets
        
    except Exception as e:
        st.error(f"Error setting up LLM validation: {str(e)}")
        return snippets

def analyze_topic_distribution(topic_results: Dict[str, List]) -> pd.DataFrame:
    """
    Analyze the distribution of topics across companies, years, and sentiments.
    
    Args:
        topic_results: Dict mapping topic names to snippet lists
    
    Returns:
        DataFrame with topic analysis
    """
    analysis_data = []
    
    for topic_name, snippets in topic_results.items():
        if not snippets:
            analysis_data.append({
                'Topic': topic_name,
                'Total_Snippets': 0,
                'Companies': 0,
                'Avg_Relevance': 0,
                'Opportunity_Sentiment': 0,
                'Risk_Sentiment': 0,
                'Neutral_Sentiment': 0,
                'Year_Range': 'N/A'
            })
            continue
        
        # Calculate metrics
        total_snippets = len(snippets)
        unique_companies = len(set(s.ticker for s in snippets))
        avg_relevance = sum(getattr(s, 'score', 0.5) for s in snippets) / total_snippets
        
        # Sentiment analysis
        sentiment_counts = {'opportunity': 0, 'risk': 0, 'neutral': 0}
        for snippet in snippets:
            sentiment = getattr(snippet, 'climate_sentiment', 'neutral')
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
        
        # Year range
        years = [int(s.year) for s in snippets if s.year and str(s.year).isdigit()]
        year_range = f"{min(years)}-{max(years)}" if years else 'N/A'
        
        analysis_data.append({
            'Topic': topic_name,
            'Total_Snippets': total_snippets,
            'Companies': unique_companies,
            'Avg_Relevance': round(avg_relevance, 3),
            'Opportunity_Sentiment': sentiment_counts['opportunity'],
            'Risk_Sentiment': sentiment_counts['risk'],
            'Neutral_Sentiment': sentiment_counts['neutral'],
            'Year_Range': year_range
        })
    
    return pd.DataFrame(analysis_data)

def visualize_topic_comparison(analysis_df: pd.DataFrame):
    """Create visualizations comparing topics."""
    
    if analysis_df.empty:
        st.warning("No data to visualize.")
        return
    
    # Topic size comparison
    st.subheader("📊 Topic Size Comparison")
    fig1 = px.bar(
        analysis_df, 
        x='Topic', 
        y='Total_Snippets',
        title="Number of Snippets per Topic",
        color='Avg_Relevance',
        color_continuous_scale='viridis'
    )
    fig1.update_xaxes(tickangle=45)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Company coverage
    st.subheader("🏢 Company Coverage")
    fig2 = px.bar(
        analysis_df, 
        x='Topic', 
        y='Companies',
        title="Number of Companies Discussing Each Topic",
        color='Companies',
        color_continuous_scale='blues'
    )
    fig2.update_xaxes(tickangle=45)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Sentiment distribution
    st.subheader("💭 Sentiment Distribution")
    sentiment_data = []
    for _, row in analysis_df.iterrows():
        for sentiment in ['Opportunity_Sentiment', 'Risk_Sentiment', 'Neutral_Sentiment']:
            sentiment_data.append({
                'Topic': row['Topic'],
                'Sentiment': sentiment.replace('_Sentiment', ''),
                'Count': row[sentiment]
            })
    
    sentiment_df = pd.DataFrame(sentiment_data)
    if not sentiment_df.empty:
        fig3 = px.bar(
            sentiment_df, 
            x='Topic', 
            y='Count', 
            color='Sentiment',
            title="Sentiment Distribution Across Topics",
            color_discrete_map={
                'Opportunity': 'green', 
                'Risk': 'red', 
                'Neutral': 'gray'
            }
        )
        fig3.update_xaxes(tickangle=45)
        st.plotly_chart(fig3, use_container_width=True)

def export_topic_results(topic_results: Dict[str, List], filename_prefix: str = "topic_search_results"):
    """Export topic search results to CSV."""
    export_data = []
    
    for topic_name, snippets in topic_results.items():
        for snippet in snippets:
            export_data.append({
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
                'llm_validation': getattr(snippet, 'llm_validation', 'N/A'),
                'llm_confidence': getattr(snippet, 'llm_confidence', 0.5)
            })
    
    if export_data:
        df = pd.DataFrame(export_data)
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Topic Search Results as CSV",
            data=csv,
            file_name=f"{filename_prefix}.csv",
            mime="text/csv"
        )
        return df
    else:
        st.warning("No data to export.")
        return None