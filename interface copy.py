import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from green_investment_rag import GreenInvestmentRAG
from helpers import load_data, filter_results

# Page config
st.set_page_config(
    page_title="Green Investment Analyzer",
    page_icon="🌱",
    layout="wide"
)

# ✅ Add the cached loader here
@st.cache_resource
def load_rag():
    rag = GreenInvestmentRAG()
    rag.load_snippets('climate_snippets.json')
    rag.load_index('climate_index.faiss')
    return rag

def main():
    st.title("🌱 Green Investment Analyzer")
    st.subheader("Extract climate investment insights from earnings calls")
    
    # Sidebar for data loading
    with st.sidebar:
        st.header("📊 Data Management")

        # Load prebuilt index from disk using the cached function
        if st.button("Load Prebuilt Index"):
            with st.spinner("Loading prebuilt index..."):
                rag = load_rag()  # ✅ use cached loader
                st.session_state.rag_system = rag
                st.session_state.data_loaded = True
                st.success("Loaded index and snippets from disk.")

        # Build and save index based on current session's RAG
        if st.button("Build and Save Embedding Index"):
            if not st.session_state.data_loaded:
                st.warning("Please load data first!")
            else:
                with st.spinner("Building and saving embedding index..."):
                    rag = st.session_state.rag_system
                    rag.build_embedding_index()
                    rag.save_index('climate_index.faiss')
                    rag.save_snippets('climate_snippets.json')
                    st.success("Embedding index built and saved.")

        if st.button("Load Sample Data"):
            with st.spinner("Loading data and building index..."):
                data = load_data()
                rag = GreenInvestmentRAG()  # ❌ do NOT call load_rag() here
                rag.load_earnings_data(data)
                rag.build_embedding_index()
                
                st.session_state.rag_system = rag
                st.session_state.data_loaded = True
                st.success(f"Loaded {len(rag.snippets)} snippets!")

        uploaded_files = st.file_uploader("Upload JSON files", type=['json'], accept_multiple_files=True)
        
        if uploaded_files:
            with st.spinner("Processing uploaded files..."):
                try:
                    combined_data = []
                    for file in uploaded_files:
                        file_data = json.load(file)
                        combined_data.extend(file_data if isinstance(file_data, list) else [file_data])
                    
                    rag = GreenInvestmentRAG()  # ❌ do NOT call load_rag() here
                    rag.load_earnings_data(combined_data)
                    rag.build_embedding_index()
                    
                    st.session_state.rag_system = rag
                    st.session_state.data_loaded = True
                    st.success(f"Loaded {len(rag.snippets)} snippets from {len(uploaded_files)} file(s)!")
                
                except Exception as e:
                    st.error(f"Error loading files: {e}")

    # Initialize session state keys if not present
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None

    if not st.session_state.data_loaded:
        st.warning("Please load data first using the sidebar.")
        return
    
    rag = st.session_state.rag_system
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "📈 Subtopic identification", "📊 Trends"])
    
    with tab1:
        st.header("Topic Search")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_type = st.radio("Search Type", ["Category", "Custom Query", "Semantic Search"])
            
            if search_type == "Category":
                category = st.selectbox("Select Investment Category", 
                                      list(rag.investment_categories.keys()))
            elif search_type == "Semantic Search":
                query = st.text_input("Enter your semantic query")

            else:
                query = st.text_input("Enter your search query")
        
        with col2:
            st.subheader("Filters")
            
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
                
                print(year_range)
            else:
                year_range = None
        
        # Search button and results (moved outside columns)
        if search_type == "Category":
            if st.button("Search by Category"):
                results = rag.search_by_category(category, top_k=50)  # Get more results before filtering
                # Apply filters
                filtered_results = filter_results(
                    results, 
                    selected_companies=selected_companies if selected_companies else None,
                    sentiment_filter=sentiment_filter,
                    year_range=year_range
                )
                display_results(filtered_results[:10])  # Show top 10 after filtering

        elif st.button("Semantic Search") and query:
                    results = rag.query_embedding_index(query, top_k=50)

                    # Optionally apply company/sentiment/year filters (reuse filter_results())
                    filtered_results = filter_results(
                        results,
                        selected_companies=selected_companies if selected_companies else None,
                        sentiment_filter=sentiment_filter,
                        year_range=year_range
                    )
                    display_results(filtered_results[:40])
        else:
            if st.button("Search") and query:
                results = rag.search_by_query(query, top_k=50)  # Get more results before filtering
                # Apply filters
                filtered_results = filter_results(
                    results, 
                    selected_companies=selected_companies if selected_companies else None,
                    sentiment_filter=sentiment_filter,
                    year_range=year_range
                )
                display_results(filtered_results[:10])  # Show top 10 after filtering
    
    with tab2:
        st.header("Subtopic identification")
        
        # Category overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Category Analysis")
            selected_category = st.selectbox("Analyze Category", 
                                           list(rag.investment_categories.keys()),
                                           key="analytics_category")
            
            if st.button("Generate Summary"):
                summary = rag.get_investment_summary(selected_category)
                
                st.metric("Total Mentions", summary['total_mentions'])
                st.metric("Companies Mentioned", summary['companies_mentioned'])
                
                # Company breakdown chart
                if summary['company_breakdown']:
                    df = pd.DataFrame(list(summary['company_breakdown'].items()), 
                                    columns=['Company', 'Mentions'])
                    fig = px.bar(df, x='Company', y='Mentions', 
                               title=f"{selected_category.title()} Mentions by Company")
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Sentiment Distribution")
            
            # Overall sentiment analysis
            all_sentiments = [s.climate_sentiment for s in rag.snippets if s.climate_sentiment]
            if all_sentiments:
                sentiment_counts = pd.Series(all_sentiments).value_counts()
                
                fig = px.pie(values=sentiment_counts.values, 
                            names=sentiment_counts.index,
                            title="Overall Climate Sentiment Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sentiment data available")
    
    with tab3:
        st.header("Company Comparison")
        
        # Select companies to compare
        companies = st.multiselect("Select Companies to Compare", 
                                 list(set([s.ticker for s in rag.snippets])))
        
        if len(companies) >= 2:
            category_comp = st.selectbox("Compare by Category", 
                                       list(rag.investment_categories.keys()),
                                       key="compare_category")
            
            if st.button("Compare Companies"):
                comparison_data = []
                for company in companies:
                    results = rag.search_by_category(category_comp, top_k=50)
                    company_results = rag.filter_by_company(results, [company])
                    
                    comparison_data.append({
                        'Company': company,
                        'Mentions': len(company_results),
                        'Avg Score': sum(r['score'] for r in company_results) / len(company_results) if company_results else 0
                    })
                
                df = pd.DataFrame(comparison_data)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(df, x='Company', y='Mentions', 
                               title=f"{category_comp.title()} Mentions Comparison")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(df, x='Company', y='Avg Score', 
                               title="Average Relevance Score")
                    st.plotly_chart(fig, use_container_width=True)
        elif companies:
            st.info("Please select at least 2 companies to compare.")
    
    with tab4:
        st.header("Investment Trends")
        
        # Create columns for search input and filters
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Search type selection
            search_type = st.radio("Search Type for Trends", ["Category", "Semantic Search"], key="trend_search_type")
            
            if search_type == "Category":
                trend_category = st.selectbox("Select Category for Trend Analysis",
                                            list(rag.investment_categories.keys()),
                                            key="trend_category")
            else:  # Semantic Search
                trend_query = st.text_input("Enter semantic query for trend analysis", 
                                        placeholder="e.g., renewable energy investments, electric vehicle partnerships",
                                        key="trend_semantic_query")
        
        with col2:
            st.subheader("Filters")
            # Company filter
            all_companies = list(set([s.ticker for s in rag.snippets]))
            selected_companies_trend = st.multiselect("Filter by Company", all_companies, key="trend_company_filter")
            
            # Sentiment filter
            sentiment_filter_trend = st.selectbox("Filter by Sentiment",
                                                ["All", "opportunity", "neutral", "risk"],
                                                key="trend_sentiment_filter")
            
            # Year range
            years = [int(s.year) for s in rag.snippets if s.year and str(s.year).isdigit()]
            if years:
                year_range_trend = st.slider("Year Range",
                                        min_value=min(years),
                                        max_value=max(years),
                                        value=(min(years), max(years)),
                                        key="trend_year_range")
            else:
                year_range_trend = None
        
        # Generate trend analysis button
        if st.button("Generate Trend Analysis", key="trend_analysis_btn"):
            # Perform search based on selected type
            if search_type == "Category":
                results = rag.search_by_category(
                    trend_category, 
                    top_k=50000,
                    selected_companies=selected_companies_trend if selected_companies_trend else None,
                    sentiment_filter=sentiment_filter_trend,
                    year_range=year_range_trend
                )
                analysis_title = trend_category.title()
            else:  # Semantic Search
                if trend_query:
                    results = rag.query_embedding_index(
                        trend_query, 
                        top_k=50000,
                        selected_companies=selected_companies_trend if selected_companies_trend else None,
                        sentiment_filter=sentiment_filter_trend,
                        year_range=year_range_trend
                    )
                    analysis_title = f"'{trend_query}'"
                else:
                    st.warning("Please enter a semantic query for trend analysis.")
                    results = []
            
            if results:
                # Create time series data
                trend_data = []
                for result in results:
                    # Convert year to int for proper sorting
                    year = int(result['year']) if result['year'] and str(result['year']).isdigit() else None
                    if year:
                        trend_data.append({
                            'Date': result['date'],
                            'Year': year,
                            'Quarter': result['quarter'],
                            'Score': result['score'],
                            'Company': result['ticker'],
                            'Sentiment': result.get('climate_sentiment', 'neutral')
                        })
                
                df = pd.DataFrame(trend_data)
                if not df.empty:
                    # Display summary statistics
                    st.subheader("Trend Analysis Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Mentions", len(df))
                    
                    with col2:
                        st.metric("Companies", df['Company'].nunique())
                    
                    with col3:
                        st.metric("Year Range", f"{df['Year'].min()}-{df['Year'].max()}")
                    
                    with col4:
                        avg_score = df['Score'].mean() if 'Score' in df.columns else 0
                        st.metric("Avg. Relevance Score", f"{avg_score:.3f}")
                    
                    # Aggregate by year-quarter
                    df['Year-Quarter'] = df['Year'].astype(str) + '-' + df['Quarter']
                    quarterly_mentions = df.groupby('Year-Quarter').size().reset_index(name='Mentions')
                    
                    # Sort by year-quarter for proper ordering
                    quarterly_mentions['Year'] = quarterly_mentions['Year-Quarter'].str.split('-').str[0].astype(int)
                    quarterly_mentions['Quarter'] = quarterly_mentions['Year-Quarter'].str.split('-').str[1]
                    quarterly_mentions = quarterly_mentions.sort_values(['Year', 'Quarter'])
                    
                    # Overall trend chart
                    fig = px.line(quarterly_mentions, x='Year-Quarter', y='Mentions',
                                title=f"{analysis_title} Investment Mentions Over Time",
                                markers=True)
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Company trend comparison
                    company_trends = df.groupby(['Year-Quarter', 'Company']).size().reset_index(name='Mentions')
                    company_trends['Year'] = company_trends['Year-Quarter'].str.split('-').str[0].astype(int)
                    company_trends['Quarter'] = company_trends['Year-Quarter'].str.split('-').str[1]
                    company_trends = company_trends.sort_values(['Year', 'Quarter'])
                    
                    fig2 = px.line(company_trends, x='Year-Quarter', y='Mentions',
                                color='Company',
                                title=f"{analysis_title} Mentions by Company",
                                markers=True)
                    fig2.update_xaxes(tickangle=45)
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Sentiment analysis over time (if semantic search is used)
                    if search_type == "Semantic Search":
                        sentiment_trends = df.groupby(['Year-Quarter', 'Sentiment']).size().reset_index(name='Mentions')
                        sentiment_trends['Year'] = sentiment_trends['Year-Quarter'].str.split('-').str[0].astype(int)
                        sentiment_trends['Quarter'] = sentiment_trends['Year-Quarter'].str.split('-').str[1]
                        sentiment_trends = sentiment_trends.sort_values(['Year', 'Quarter'])
                        
                        fig3 = px.line(sentiment_trends, x='Year-Quarter', y='Mentions',
                                    color='Sentiment',
                                    title=f"{analysis_title} Sentiment Trends Over Time",
                                    markers=True)
                        fig3.update_xaxes(tickangle=45)
                        st.plotly_chart(fig3, use_container_width=True)
                    
                    # Show top companies table
                    st.subheader("Top Companies by Mentions")
                    top_companies = df.groupby('Company').agg({
                        'Date': 'count',
                        'Score': 'mean',
                        'Sentiment': lambda x: x.mode().iloc[0] if not x.mode().empty else 'neutral'
                    }).reset_index()
                    top_companies.columns = ['Company', 'Total Mentions', 'Avg Score', 'Dominant Sentiment']
                    top_companies = top_companies.sort_values('Total Mentions', ascending=False)
                    st.dataframe(top_companies, use_container_width=True)
                    
                else:
                    st.info("No trend data available for the selected search after applying filters.")
            else:
                st.info("No results found for the selected search criteria.")

def display_results(results):
    """Display search results in a nice format"""
    if not results:
        st.warning("No results found.")
        return
    
    st.subheader(f"Found {len(results)} results")
    
    for i, result in enumerate(results):
        with st.expander(f"Result {i+1}: {result['company']} ({result['score']:.3f})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Text:** {result['text']}")
                st.write(f"**Speaker:** {result['speaker']} ({result['profession']})")
            
            with col2:
                st.metric("Relevance Score", f"{result['score']:.3f}")
                st.write(f"**Date:** {result['date']}")
                st.write(f"**Quarter:** {result['quarter']} {result['year']}")
                if result['climate_sentiment']:
                    sentiment_color = {
                        'opportunity': '🟢',
                        'neutral': '🟡', 
                        'risk': '🔴'
                    }
                    st.write(f"**Sentiment:** {sentiment_color.get(result['climate_sentiment'], '')} {result['climate_sentiment']}")

if __name__ == "__main__":
    main()