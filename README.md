# Green Investment Analyzer

A Streamlit application for analyzing climate investment insights from earnings calls using topic modeling and evolution analysis.

## Project Structure

```
green-investment-analyzer/
├── main.py                 # Main application entry point
├── config.py              # Configuration and environment setup
├── data_loaders.py        # Data loading and caching functions
├── topic_analysis.py      # Topic modeling and analysis
├── evolution_analysis.py  # Topic evolution over time analysis
├── visualization.py       # Chart and visualization functions
├── ui_components.py       # UI component functions for tabs
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Features

- **Multi-market Analysis**: Support for EU (STOXX 600), US (S&P 500), and combined datasets
- **Topic Modeling**: Advanced topic discovery using BERTopic and UMAP
- **LLM Integration**: Automatic topic naming using Anthropic Claude
- **Evolution Analysis**: Track how topics evol