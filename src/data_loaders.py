# data_loaders.py - Data loading and caching functions
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
from src.green_investment_rag import GreenInvestmentRAG

@st.cache_resource
def load_eu_rag():
    """Load EU market data."""
    rag = GreenInvestmentRAG()
    rag.load_market_data('EU')
    return rag

@st.cache_resource
def load_us_rag():
    """Load US market data."""
    rag = GreenInvestmentRAG()
    rag.load_market_data('US')
    return rag

@st.cache_resource
def load_combined_rag():
    """Load combined EU + US data (fallback method)."""
    rag = GreenInvestmentRAG()
    rag.load_combined_data()
    return rag

@st.cache_resource
def load_full_rag():
    """Load the full combined index directly (EU + US)."""
    rag = GreenInvestmentRAG()
    try:
        rag.load_market_data('FULL')
        return rag
    except FileNotFoundError:
        st.error("Full index files not found. Please ensure climate_index_full.faiss and climate_snippets_full.json exist.")
        return None

def load_market_data(market_option):
    """Load market data based on selection."""
    try:
        if market_option == "EU (STOXX 600)":
            rag = load_eu_rag()
            market_key = "EU"
        elif market_option == "US (S&P 500)":
            rag = load_us_rag()
            market_key = "US"
        elif market_option == "Full Index (EU + US)":
            rag = load_full_rag()
            if rag is None:
                return None, None, False
            market_key = "Full"
        else:  # Combined (EU + US) - fallback method
            rag = load_combined_rag()
            market_key = "Combined"
        
        return rag, market_key, True
        
    except Exception as e:
        st.error(f"Error loading {market_option} data: {e}")
        return None, None, False