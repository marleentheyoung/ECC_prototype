# config.py - Configuration and environment setup
import os
import sys
import warnings

def setup_environment():
    """Setup environment for macOS compatibility and suppress warnings."""
    # Force single-threaded execution for macOS stability
    if sys.platform == 'darwin':  # macOS
        os.environ['LOKY_MAX_CPU_COUNT'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'
        os.environ['NUMBA_NUM_THREADS'] = '1'

    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

def check_required_libraries():
    """Check if required libraries are available."""
    try:
        import umap.umap_ as umap
        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer
        return True, None
    except ImportError as e:
        return False, str(e)

# Application constants
APP_CONFIG = {
    'page_title': "Green Investment Analyzer",
    'page_icon': "🌱",
    'layout': "wide",
    'max_topics': 15,
    'default_topics': 6,
    'default_max_snippets': 3000,
    'min_snippets': 100
}

CLAUDE_CONFIG = {
    'model': "claude-3-5-sonnet-20240620",
    'max_tokens': 100,
    'temperature': 0.2
}

TOPIC_ANALYSIS_CONFIG = {
    'top_n_words': 8,
    'umap_n_neighbors': 10,
    'umap_n_components': 3,
    'umap_min_dist': 0.1,
    'umap_spread': 1.0,
    'vectorizer_max_features': 1000,
    'vectorizer_min_df': 3,
    'vectorizer_max_df': 0.8
}