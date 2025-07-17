# event_study_analyzer.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import numpy as np  # Add this line if not already present
from scipy import stats
import streamlit as st
from typing import Dict, List, Tuple, Optional
import re

class EventStudyAnalyzer:
    def __init__(self, rag_system):
        self.rag = rag_system
        self.predefined_events = {
            "Paris Agreement Adoption": "2015-12-12",
            "Trump Election": "2016-11-08", 
            "Biden Election": "2020-11-07",
            "EU Green Deal": "2019-12-11",
            "US IRA Passage": "2022-08-16",
            "COP21 Opening": "2015-11-30",
            "US Paris Withdrawal": "2017-06-01",
            "US Paris Re-entry": "2021-01-20",
            "COP26 Glasgow": "2021-10-31",
            "EU Taxonomy Regulation": "2020-06-18"
        }
    
    # ==========================================
    # CORE EVENT ANALYSIS METHODS
    # ==========================================
    
    def analyze_event_impact(self, event_date: str, event_window: Tuple[int, int], 
                       search_queries: List[str], compare_regions: bool = False) -> Dict:
        """
        Main method to analyze event impact on firm discussions using semantic search.
        
        Args:
            event_date: Date of the event (YYYY-MM-DD)
            event_window: Tuple of (days_before, days_after) event
            search_queries: List of semantic search queries for the event
            compare_regions: Whether to compare EU vs US responses
            
        Returns:
            Dictionary containing analysis results
        """
        event_date_obj = datetime.strptime(event_date, "%Y-%m-%d")
        
        # Define time periods
        baseline_start = event_date_obj + timedelta(days=event_window[0] - 90)
        baseline_end = event_date_obj + timedelta(days=event_window[0])
        event_start = event_date_obj + timedelta(days=event_window[0])
        event_end = event_date_obj + timedelta(days=event_window[1])
        
        # Get relevant snippets using semantic search
        baseline_snippets = self._get_snippets_semantic_search(baseline_start, baseline_end, search_queries)
        event_snippets = self._get_snippets_semantic_search(event_start, event_end, search_queries)

        # Calculate metrics
        results = {
            'event_date': event_date,
            'event_window': event_window,
            'search_queries': search_queries,
            'baseline_period': (baseline_start.strftime('%Y-%m-%d'), baseline_end.strftime('%Y-%m-%d')),
            'event_period': (event_start.strftime('%Y-%m-%d'), event_end.strftime('%Y-%m-%d')),
            'baseline_metrics': self._calculate_period_metrics(baseline_snippets),
            'event_metrics': self._calculate_period_metrics(event_snippets),
            'abnormal_metrics': self._calculate_abnormal_metrics(baseline_snippets, event_snippets),
            'statistical_tests': self._perform_statistical_tests(baseline_snippets, event_snippets),
            'timeline_data': self._create_timeline_data_semantic(event_date_obj, event_window, search_queries)
        }
        
        # Add regional analysis if requested
        if compare_regions:
            results['regional_analysis'] = self._analyze_regional_differences_semantic(
                event_date_obj, event_window, search_queries
            )
        
        return results
    
    def _get_snippets_in_period(self, start_date: datetime, end_date: datetime, 
                               keywords: List[str]) -> List:
        """Get snippets that fall within a time period and contain keywords."""
        relevant_snippets = []
        
        for snippet in self.rag.snippets:
            # Parse snippet date
            snippet_date = self._parse_snippet_date(snippet.date)
            if snippet_date is None:
                continue
                
            # Check if snippet is in time period
            if start_date <= snippet_date <= end_date:
                # Check if snippet contains any keywords
                if self._snippet_contains_keywords(snippet.text, keywords):
                    relevant_snippets.append(snippet)
        
        return relevant_snippets
    
    def _get_snippets_semantic_search(self, start_date: datetime, end_date: datetime, 
                                search_queries: List[str], relevance_threshold: float = 0.40) -> List:
        """Get snippets using semantic search within a time period."""
        all_relevant_snippets = []
        print(search_queries)
        for query in search_queries:
            print("\n\n\n\nwere HEREEEEEEEE\n\n\n")
            # Use existing RAG semantic search
            search_results = self.rag.query_embedding_index(
                query, 
                top_k=None,  # Get all results
                relevance_threshold=relevance_threshold
            )

            # print(search_results)
            
            # Convert results to snippet objects and filter by date
            for result in search_results:
                # Find corresponding snippet
                for snippet in self.rag.snippets:
                    if (snippet.text == result['text'] and 
                        snippet.company == result['company'] and
                        snippet.ticker == result['ticker']):
                        
                        # Check if snippet is in time period
                        snippet_date = self._parse_snippet_date(snippet.date)
                        if snippet_date and start_date <= snippet_date <= end_date:
                            # Add relevance score to snippet
                            snippet.relevance_score = result['score']
                            all_relevant_snippets.append(snippet)
                        break
        print(len(all_relevant_snippets))
        
        # Remove duplicates (same snippet found by multiple queries)
        unique_snippets = []
        seen_texts = set()
        for snippet in all_relevant_snippets:
            text_key = f"{snippet.text}_{snippet.company}_{snippet.date}"
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_snippets.append(snippet)
        
        return unique_snippets

    def _create_timeline_data_semantic(self, event_date: datetime, event_window: Tuple[int, int], 
                                    search_queries: List[str]) -> Dict:
        """Create timeline data using semantic search."""
        start_date = event_date + timedelta(days=event_window[0])
        end_date = event_date + timedelta(days=event_window[1])
        
        # Create daily timeline
        timeline = []
        current_date = start_date
        
        while current_date <= end_date:
            day_end = current_date + timedelta(days=1)
            day_snippets = self._get_snippets_semantic_search(current_date, day_end, search_queries)
            
            # Calculate average relevance score for the day
            avg_relevance = np.mean([getattr(s, 'relevance_score', 0.5) for s in day_snippets]) if day_snippets else 0.0
            
            timeline.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'mentions': len(day_snippets),
                'companies': len(set(s.ticker for s in day_snippets)) if day_snippets else 0,
                'sentiment_score': self._calculate_avg_sentiment(day_snippets),
                'avg_relevance': avg_relevance,
                'days_from_event': (current_date - event_date).days
            })
            
            current_date += timedelta(days=1)
        
        return {
            'daily_timeline': timeline,
            'event_date': event_date.strftime('%Y-%m-%d')
        }

    def _analyze_regional_differences_semantic(self, event_date: datetime, event_window: Tuple[int, int], 
                                            search_queries: List[str]) -> Dict:
        """Analyze regional differences using semantic search."""
        from utils import determine_market
        
        # Get event period snippets using semantic search
        event_start = event_date + timedelta(days=event_window[0])
        event_end = event_date + timedelta(days=event_window[1])
        event_snippets = self._get_snippets_semantic_search(event_start, event_end, search_queries)
        
        # Separate by region
        eu_snippets = []
        us_snippets = []
        
        for snippet in event_snippets:
            market = determine_market(snippet)
            if market == 'EU':
                eu_snippets.append(snippet)
            else:
                us_snippets.append(snippet)
        
        # Calculate metrics for each region
        eu_metrics = self._calculate_period_metrics(eu_snippets)
        us_metrics = self._calculate_period_metrics(us_snippets)
        
        # Add average relevance scores
        eu_metrics['avg_relevance'] = np.mean([getattr(s, 'relevance_score', 0.5) for s in eu_snippets]) if eu_snippets else 0.0
        us_metrics['avg_relevance'] = np.mean([getattr(s, 'relevance_score', 0.5) for s in us_snippets]) if us_snippets else 0.0
        
        # Statistical test for regional differences
        regional_test = self._test_regional_significance(eu_snippets, us_snippets)
        
        return {
            'eu_metrics': eu_metrics,
            'us_metrics': us_metrics,
            'regional_test': regional_test,
            'eu_timeline': self._create_regional_timeline_semantic(eu_snippets, event_date, event_window),
            'us_timeline': self._create_regional_timeline_semantic(us_snippets, event_date, event_window)
        }

    def _create_timeline_data_semantic(self, event_date: datetime, event_window: Tuple[int, int], 
                                    search_queries: List[str]) -> Dict:
        """Create timeline data using semantic search."""
        start_date = event_date + timedelta(days=event_window[0])
        end_date = event_date + timedelta(days=event_window[1])
        
        # Create daily timeline
        timeline = []
        current_date = start_date
        
        while current_date <= end_date:
            day_end = current_date + timedelta(days=1)
            day_snippets = self._get_snippets_semantic_search(current_date, day_end, search_queries)
            
            # Calculate average relevance score for the day
            avg_relevance = np.mean([getattr(s, 'relevance_score', 0.5) for s in day_snippets]) if day_snippets else 0.0
            
            timeline.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'mentions': len(day_snippets),
                'companies': len(set(s.ticker for s in day_snippets)) if day_snippets else 0,
                'sentiment_score': self._calculate_avg_sentiment(day_snippets),
                'avg_relevance': avg_relevance,
                'days_from_event': (current_date - event_date).days
            })
            
            current_date += timedelta(days=1)

        return {
            'daily_timeline': timeline,
            'event_date': event_date.strftime('%Y-%m-%d')
        }

    def _analyze_regional_differences_semantic(self, event_date: datetime, event_window: Tuple[int, int], 
                                            search_queries: List[str]) -> Dict:
        """Analyze regional differences using semantic search."""
        from utils import determine_market
        
        # Get event period snippets using semantic search
        event_start = event_date + timedelta(days=event_window[0])
        event_end = event_date + timedelta(days=event_window[1])
        event_snippets = self._get_snippets_semantic_search(event_start, event_end, search_queries)
        
        # Separate by region
        eu_snippets = []
        us_snippets = []
        
        for snippet in event_snippets:
            market = determine_market(snippet)
            if market == 'EU':
                eu_snippets.append(snippet)
            else:
                us_snippets.append(snippet)
        
        # Calculate metrics for each region
        eu_metrics = self._calculate_period_metrics(eu_snippets)
        us_metrics = self._calculate_period_metrics(us_snippets)
        
        # Add average relevance scores
        eu_metrics['avg_relevance'] = np.mean([getattr(s, 'relevance_score', 0.5) for s in eu_snippets]) if eu_snippets else 0.0
        us_metrics['avg_relevance'] = np.mean([getattr(s, 'relevance_score', 0.5) for s in us_snippets]) if us_snippets else 0.0
        
        # Statistical test for regional differences
        regional_test = self._test_regional_significance(eu_snippets, us_snippets)
        
        return {
            'eu_metrics': eu_metrics,
            'us_metrics': us_metrics,
            'regional_test': regional_test,
            'eu_timeline': self._create_regional_timeline_semantic(eu_snippets, event_date, event_window),
            'us_timeline': self._create_regional_timeline_semantic(us_snippets, event_date, event_window)
        }

    def _create_regional_timeline_semantic(self, snippets: List, event_date: datetime, 
                                        event_window: Tuple[int, int]) -> List[Dict]:
        """Create timeline data for a specific region using semantic search."""
        # Group snippets by date
        date_groups = {}
        for snippet in snippets:
            snippet_date = self._parse_snippet_date(snippet.date)
            if snippet_date:
                date_key = snippet_date.strftime('%Y-%m-%d')
                if date_key not in date_groups:
                    date_groups[date_key] = []
                date_groups[date_key].append(snippet)
        
        # Create timeline
        timeline = []
        for date_key, date_snippets in date_groups.items():
            date_obj = datetime.strptime(date_key, '%Y-%m-%d')
            avg_relevance = np.mean([getattr(s, 'relevance_score', 0.5) for s in date_snippets])
            
            timeline.append({
                'date': date_key,
                'mentions': len(date_snippets),
                'companies': len(set(s.ticker for s in date_snippets)),
                'sentiment_score': self._calculate_avg_sentiment(date_snippets),
                'avg_relevance': avg_relevance,
                'days_from_event': (date_obj - event_date).days
            })
        
        return sorted(timeline, key=lambda x: x['date'])
        
    def _calculate_abnormal_metrics(self, baseline_snippets: List, event_snippets: List) -> Dict:
        """Calculate abnormal attention and other metrics."""
        baseline_metrics = self._calculate_period_metrics(baseline_snippets)
        event_metrics = self._calculate_period_metrics(event_snippets)
        
        # Calculate abnormal attention (similar to abnormal returns)
        baseline_mentions = baseline_metrics['total_mentions']
        event_mentions = event_metrics['total_mentions']
        
        # FIX: Get event window from session state safely
        event_window = st.session_state.get('event_window', (-90, 90))
        event_window_days = abs(event_window[1] - event_window[0])
        
        # FIX: Avoid division by zero
        baseline_daily = baseline_mentions / 90 if baseline_mentions > 0 else 0
        event_daily = event_mentions / event_window_days if event_window_days > 0 else 0
        
        abnormal_attention = event_daily - baseline_daily
        
        # FIX: Safe calculations for other metrics
        abnormal_companies = event_metrics['unique_companies'] - baseline_metrics['unique_companies']
        sentiment_change = event_metrics['avg_sentiment_score'] - baseline_metrics['avg_sentiment_score']
        specificity_change = event_metrics['specificity_score'] - baseline_metrics['specificity_score']
        
        return {
            'abnormal_attention': abnormal_attention,
            'abnormal_companies': abnormal_companies,
            'sentiment_change': sentiment_change,
            'specificity_change': specificity_change,
            'baseline_daily_avg': baseline_daily,
            'event_daily_avg': event_daily
        }

    def _calculate_period_metrics(self, snippets: List) -> Dict:
        """Calculate summary metrics for a period."""
        if not snippets:
            return {
                'total_mentions': 0,
                'unique_companies': 0,
                'avg_sentiment_score': 0.0,
                'forward_looking_ratio': 0.0,
                'specificity_score': 0.0
            }
        
        return {
            'total_mentions': len(snippets),
            'unique_companies': len(set(s.ticker for s in snippets)),
            'avg_sentiment_score': self._calculate_avg_sentiment(snippets),
            'forward_looking_ratio': self._calculate_forward_looking_ratio(snippets),
            'specificity_score': self._calculate_specificity_score(snippets)
        }

    def _calculate_avg_sentiment(self, snippets: List) -> float:
        """Calculate average sentiment score for snippets."""
        if not snippets:
            return 0.0
        
        sentiment_map = {'opportunity': 1.0, 'neutral': 0.0, 'risk': -1.0}
        sentiment_scores = [
            sentiment_map.get(snippet.climate_sentiment, 0.0) 
            for snippet in snippets
        ]
        
        # FIX: Handle empty scores
        if not sentiment_scores:
            return 0.0
        
        return np.mean(sentiment_scores)

    def _calculate_forward_looking_ratio(self, snippets: List) -> float:
        """Calculate ratio of forward-looking statements."""
        if not snippets:
            return 0.0
        
        forward_indicators = [
            'expect', 'anticipate', 'plan', 'will', 'future', 'upcoming', 
            'prepare', 'ready', 'next', 'coming'
        ]
        
        forward_count = 0
        for snippet in snippets:
            text_lower = snippet.text.lower()
            if any(indicator in text_lower for indicator in forward_indicators):
                forward_count += 1
        
        return forward_count / len(snippets)

    def _calculate_specificity_score(self, snippets: List) -> float:
        """Calculate how specific the policy discussions are."""
        if not snippets:
            return 0.0
        
        specific_terms = [
            'paris agreement', 'cop21', 'eu ets', 'clean power plan', 'ira',
            'green deal', 'taxonomy', 'cbam', 'article 6', 'ndcs'
        ]
        
        generic_terms = [
            'climate policy', 'regulation', 'environmental', 'policy',
            'government', 'legislation'
        ]
        
        specific_count = 0
        generic_count = 0
        
        for snippet in snippets:
            text_lower = snippet.text.lower()
            if any(term in text_lower for term in specific_terms):
                specific_count += 1
            elif any(term in text_lower for term in generic_terms):
                generic_count += 1
        
        total_policy_mentions = specific_count + generic_count
        
        # FIX: Avoid division by zero
        if total_policy_mentions == 0:
            return 0.0
        
        return specific_count / total_policy_mentions

    # ==========================================
    # STATISTICAL TESTING METHODS
    # ==========================================
    
    def _perform_statistical_tests(self, baseline_snippets: List, event_snippets: List) -> Dict:
        """Perform statistical tests for event significance."""
        tests_results = {}
        
        # Prepare data for testing
        baseline_mentions = self._aggregate_mentions_by_period(baseline_snippets, 'daily')
        event_mentions = self._aggregate_mentions_by_period(event_snippets, 'daily')
        
        # T-test for difference in means
        if len(baseline_mentions) > 1 and len(event_mentions) > 1:
            t_stat, t_p_value = stats.ttest_ind(event_mentions, baseline_mentions)
            tests_results['t_test'] = {
                'statistic': t_stat,
                'p_value': t_p_value,
                'significant': t_p_value < 0.05
            }
        
        # Mann-Whitney U test (non-parametric)
        if len(baseline_mentions) > 0 and len(event_mentions) > 0:
            try:
                u_stat, u_p_value = stats.mannwhitneyu(event_mentions, baseline_mentions, alternative='two-sided')
                tests_results['mann_whitney'] = {
                    'statistic': u_stat,
                    'p_value': u_p_value,
                    'significant': u_p_value < 0.05
                }
            except ValueError:
                tests_results['mann_whitney'] = {'error': 'Insufficient data for test'}
        
        # Kolmogorov-Smirnov test for distribution differences
        if len(baseline_mentions) > 0 and len(event_mentions) > 0:
            ks_stat, ks_p_value = stats.ks_2samp(baseline_mentions, event_mentions)
            tests_results['kolmogorov_smirnov'] = {
                'statistic': ks_stat,
                'p_value': ks_p_value,
                'significant': ks_p_value < 0.05
            }
        
        return tests_results
    
    def test_parallel_trends(self, event_date: str, keywords: List[str], 
                           pre_event_periods: int = 4) -> Dict:
        """Test parallel trends assumption for DiD-style analysis."""
        event_date_obj = datetime.strptime(event_date, "%Y-%m-%d")
        
        # Create quarterly periods before the event
        quarterly_data = []
        for i in range(pre_event_periods, 0, -1):
            period_end = event_date_obj - timedelta(days=90 * (i-1))
            period_start = period_end - timedelta(days=90)
            
            period_snippets = self._get_snippets_in_period(period_start, period_end, keywords)
            quarterly_data.append({
                'period': f"Q-{i}",
                'mentions': len(period_snippets),
                'companies': len(set(s.ticker for s in period_snippets))
            })
        
        # Test for trend
        periods = list(range(len(quarterly_data)))
        mentions = [d['mentions'] for d in quarterly_data]
        
        if len(mentions) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(periods, mentions)
            
            return {
                'trend_test': {
                    'slope': slope,
                    'p_value': p_value,
                    'parallel_trends_assumption': p_value > 0.05  # Non-significant trend supports assumption
                },
                'quarterly_data': quarterly_data
            }
        
        return {'error': 'Insufficient data for parallel trends test'}
    
    # ==========================================
    # REGIONAL ANALYSIS METHODS
    # ==========================================
    
    def _analyze_regional_differences(self, event_date: datetime, event_window: Tuple[int, int], 
                                    keywords: List[str]) -> Dict:
        """Analyze differences between EU and US firm responses."""
        from utils import determine_market
        
        # Get event period snippets
        event_start = event_date + timedelta(days=event_window[0])
        event_end = event_date + timedelta(days=event_window[1])
        event_snippets = self._get_snippets_in_period(event_start, event_end, keywords)
        
        # Separate by region
        eu_snippets = []
        us_snippets = []
        
        for snippet in event_snippets:
            market = determine_market(snippet)
            if market == 'EU':
                eu_snippets.append(snippet)
            else:
                us_snippets.append(snippet)
        
        # Calculate metrics for each region
        eu_metrics = self._calculate_period_metrics(eu_snippets)
        us_metrics = self._calculate_period_metrics(us_snippets)
        
        # Statistical test for regional differences
        regional_test = self._test_regional_significance(eu_snippets, us_snippets)
        
        return {
            'eu_metrics': eu_metrics,
            'us_metrics': us_metrics,
            'regional_test': regional_test,
            'eu_timeline': self._create_regional_timeline(eu_snippets, event_date, event_window),
            'us_timeline': self._create_regional_timeline(us_snippets, event_date, event_window)
        }
    
    def _test_regional_significance(self, eu_snippets: List, us_snippets: List) -> Dict:
        """Test statistical significance of regional differences."""
        # Aggregate mentions by day for each region
        eu_daily = self._aggregate_mentions_by_period(eu_snippets, 'daily')
        us_daily = self._aggregate_mentions_by_period(us_snippets, 'daily')
        
        if len(eu_daily) > 1 and len(us_daily) > 1:
            t_stat, p_value = stats.ttest_ind(eu_daily, us_daily)
            return {
                'statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'eu_higher': t_stat > 0
            }
        
        return {'error': 'Insufficient data for regional comparison'}
    
    # ==========================================
    # DATA AGGREGATION & TIMELINE METHODS
    # ==========================================
    
    def _create_timeline_data(self, event_date: datetime, event_window: Tuple[int, int], 
                        keywords: List[str]) -> Dict:
        """Create timeline data for visualization."""
        start_date = event_date + timedelta(days=event_window[0])
        end_date = event_date + timedelta(days=event_window[1])
        
        # Create daily timeline
        timeline = []
        current_date = start_date
        
        while current_date <= end_date:
            day_end = current_date + timedelta(days=1)
            day_snippets = self._get_snippets_in_period(current_date, day_end, keywords)
            
            timeline.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'mentions': len(day_snippets),
                'companies': len(set(s.ticker for s in day_snippets)) if day_snippets else 0,
                'sentiment_score': self._calculate_avg_sentiment(day_snippets),
                'days_from_event': (current_date - event_date).days
            })
            
            current_date += timedelta(days=1)
        
        return {
            'daily_timeline': timeline,
            'event_date': event_date.strftime('%Y-%m-%d')
        }
    
    def _aggregate_mentions_by_period(self, snippets: List, frequency: str) -> List[float]:
        """Aggregate mention counts by time period."""
        if not snippets:
            return []
        
        # Group snippets by date
        date_groups = {}
        for snippet in snippets:
            snippet_date = self._parse_snippet_date(snippet.date)
            if snippet_date:
                date_key = snippet_date.strftime('%Y-%m-%d')
                if date_key not in date_groups:
                    date_groups[date_key] = 0
                date_groups[date_key] += 1
        
        return list(date_groups.values())
    
    def _create_regional_timeline(self, snippets: List, event_date: datetime, 
                                event_window: Tuple[int, int]) -> List[Dict]:
        """Create timeline data for a specific region."""
        return self._create_timeline_data(event_date, event_window, [])['daily_timeline']
    
    # ==========================================
    # TEXT ANALYSIS HELPER METHODS
    # ==========================================
    
    def _snippet_contains_keywords(self, text: str, keywords: List[str]) -> bool:
        """Check if snippet contains any of the specified keywords."""
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in keywords)
    
    def _calculate_avg_sentiment(self, snippets: List) -> float:
        """Calculate average sentiment score for snippets."""
        if not snippets:
            return 0.0
        
        sentiment_map = {'opportunity': 1.0, 'neutral': 0.0, 'risk': -1.0}
        sentiment_scores = [
            sentiment_map.get(snippet.climate_sentiment, 0.0) 
            for snippet in snippets
        ]
        
        return np.mean(sentiment_scores)
    
    def _calculate_forward_looking_ratio(self, snippets: List) -> float:
        """Calculate ratio of forward-looking statements."""
        if not snippets:
            return 0.0
        
        forward_indicators = [
            'expect', 'anticipate', 'plan', 'will', 'future', 'upcoming', 
            'prepare', 'ready', 'next', 'coming'
        ]
        
        forward_count = 0
        for snippet in snippets:
            text_lower = snippet.text.lower()
            if any(indicator in text_lower for indicator in forward_indicators):
                forward_count += 1
        
        return forward_count / len(snippets)
    
    # ==========================================
    # UTILITY METHODS
    # ==========================================
    
    def _parse_snippet_date(self, date_str: str) -> Optional[datetime]:
        """Parse snippet date string to datetime object."""
        if not date_str:
            return None
        
        try:
            # Try different date formats
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            return None
        except:
            return None
    
    # ==========================================
    # EXPORT METHODS
    # ==========================================
    
    def export_event_study_data(self, results: Dict, event_name: str) -> pd.DataFrame:
        """Export event study results for econometric analysis."""
        timeline_data = results.get('timeline_data', {}).get('daily_timeline', [])
        
        if not timeline_data:
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(timeline_data)
        df['event_name'] = event_name
        df['abnormal_attention'] = df['mentions'] - results.get('baseline_metrics', {}).get('total_mentions', 0) / 90
        
        return df
    
    def create_placebo_tests(self, real_event_date: str, keywords: List[str], 
                           num_placebo: int = 5) -> Dict:
        """Create placebo tests with fake event dates."""
        real_date = datetime.strptime(real_event_date, "%Y-%m-%d")
        placebo_results = {}
        
        # Generate random dates within 2 years before real event
        for i in range(num_placebo):
            # Random date between 2 years and 6 months before real event
            days_back = np.random.randint(180, 730)
            placebo_date = real_date - timedelta(days=days_back)
            
            placebo_results[f'Placebo_{i+1}'] = self.analyze_event_impact(
                placebo_date.strftime('%Y-%m-%d'),
                (-30, 30),  # Shorter window for placebo
                keywords,
                compare_regions=False
            )
        
        return placebo_results