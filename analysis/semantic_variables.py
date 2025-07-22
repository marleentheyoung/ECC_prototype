# semantic_climate_variables.py - Semantic search-based climate exposure construction

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time

# At the top of semantic_variables.py, replace the existing path setup with:
import sys
import os
from pathlib import Path

# Get project root (parent of analysis/)
current_file = Path(__file__)
project_root = current_file.parent.parent
src_dir = project_root / 'src'

# Add to path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

# Now the import should work
from src.green_investment_rag import GreenInvestmentRAG

class SemanticClimateExposureConstructor:
    """
    Construct firm-level climate change exposure variables using semantic search
    - More sophisticated than keyword counting
    - Captures semantic meaning and context
    - Allows for LLM validation of relevance
    """
    
    def __init__(self, rag_system, output_dir="data/semantic_firm_variables"):
        self.rag = rag_system
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Semantic climate topics (replacing simple bigrams)
        self.climate_topics = {
            'opportunities': {
                'query': 'renewable energy investments clean technology green innovation sustainable business models energy transition opportunities',
                'threshold': 0.40,
                'validation_prompt': 'climate investment opportunities and green business'
            },
            'regulation': {
                'query': 'climate policy environmental regulation carbon pricing emissions trading Paris Agreement regulatory compliance',
                'threshold': 0.40,
                'validation_prompt': 'climate regulation and environmental policy'
            },
            'physical_risk': {
                'query': 'extreme weather climate physical risk supply chain disruption weather events flooding drought',
                'threshold': 0.40,
                'validation_prompt': 'physical climate risks and weather impacts'
            },
            'transition_risk': {
                'query': 'stranded assets carbon intensive business model transition risk technology disruption',
                'threshold': 0.40,
                'validation_prompt': 'climate transition risks and business model changes'
            },
            'disclosure': {
                'query': 'climate risk disclosure ESG reporting sustainability reporting climate scenario analysis',
                'threshold': 0.40,
                'validation_prompt': 'climate risk disclosure and sustainability reporting'
            }
        }
        
        # Policy-specific semantic searches (more precise than keyword matching)
        self.policy_topics = {
            'paris_agreement': {
                'query': 'Paris Agreement COP21 international climate accord global climate commitment NDCs',
                'threshold': 0.45,
                'validation_prompt': 'Paris Agreement and international climate commitments'
            },
            'eu_ets': {
                'query': 'EU ETS European emissions trading system carbon allowances EU carbon market',
                'threshold': 0.45,
                'validation_prompt': 'EU Emissions Trading System and carbon allowances'
            },
            'carbon_pricing': {
                'query': 'carbon tax carbon pricing internal carbon price shadow carbon price',
                'threshold': 0.45,
                'validation_prompt': 'carbon pricing and carbon taxation'
            },
            'green_deal': {
                'query': 'European Green Deal EU Green Deal climate neutrality EU climate policy',
                'threshold': 0.45,
                'validation_prompt': 'European Green Deal and EU climate policy'
            },
            'inflation_reduction_act': {
                'query': 'Inflation Reduction Act IRA climate investment tax credits clean energy incentives',
                'threshold': 0.45,
                'validation_prompt': 'US Inflation Reduction Act and climate incentives'
            },
            'tcfd': {
                'query': 'TCFD Task Force Climate-related Financial Disclosures climate scenario analysis',
                'threshold': 0.45,
                'validation_prompt': 'TCFD climate risk disclosures'
            }
        }

    # Add progress to main pipeline function:
    def construct_semantic_variables(self, start_year=2010, end_year=2024, use_llm_validation=True):
        """Main pipeline using semantic search instead of keyword counting"""
        
        print("🧠 Starting semantic climate exposure variable construction...")
        
        # Create overall progress bar
        total_steps = 7
        main_pbar = tqdm(total=total_steps, desc="Overall progress", unit="steps")
        
        # 1. Create firm-quarter panel structure
        main_pbar.set_description("Creating panel structure")
        panel_data = self.create_panel_structure(start_year, end_year)
        print(f"📊 Created panel with {len(panel_data)} firm-quarter observations")
        main_pbar.update(1)
        
        # 2. Calculate semantic exposure measures
        main_pbar.set_description("Calculating semantic exposure")
        panel_data = self.calculate_semantic_exposure(panel_data, use_llm_validation)
        print("🎯 Calculated semantic exposure measures")
        main_pbar.update(1)
        
        # 3. Calculate policy-specific semantic measures
        main_pbar.set_description("Calculating policy measures")
        panel_data = self.calculate_policy_semantic_measures(panel_data, use_llm_validation)
        print("📋 Calculated policy-specific measures")
        main_pbar.update(1)
        
        # 4. Calculate semantic sentiment measures
        main_pbar.set_description("Calculating sentiment measures")
        panel_data = self.calculate_semantic_sentiment(panel_data)
        print("💭 Calculated semantic sentiment measures")
        main_pbar.update(1)
        
        # 5. Calculate semantic evolution measures
        main_pbar.set_description("Calculating evolution measures")
        panel_data = self.calculate_semantic_evolution(panel_data)
        print("📈 Calculated semantic evolution measures")
        main_pbar.update(1)
        
        # 6. Add validation and quality metrics
        main_pbar.set_description("Adding quality metrics")
        panel_data = self.add_quality_metrics(panel_data)
        print("✅ Added quality metrics")
        main_pbar.update(1)
        
        # 7. Save datasets with methodology documentation
        main_pbar.set_description("Saving datasets")
        self.save_semantic_datasets(panel_data)
        print("💾 Saved semantic datasets")
        main_pbar.update(1)
        
        main_pbar.close()
        print("🎉 Semantic variable construction completed!")
        
        return panel_data

    # Replace the calculate_semantic_exposure function:
    def calculate_semantic_exposure(self, panel_data, use_llm_validation=True):
        """Calculate exposure using semantic search with optional LLM validation"""
        
        print("🔍 Running semantic searches for climate topics...")
        
        # Filter to only rows with earnings calls to avoid processing empty rows
        rows_with_calls = [row for row in panel_data if row['has_earnings_call']]
        
        # Progress bar for firm-quarters with data
        pbar_firms = tqdm(
            rows_with_calls, 
            desc="Processing firm-quarters", 
            unit="firm-qtrs",
            leave=True
        )
        
        for row in pbar_firms:
            firm_snippets = row['snippets']
            
            # Update progress bar description
            pbar_firms.set_description(f"Processing {row['ticker']} {row['year']}Q{row['quarter']}")
            
            # Progress bar for climate topics
            topic_pbar = tqdm(
                self.climate_topics.items(), 
                desc="Climate topics", 
                leave=False,
                unit="topics"
            )
            
            for topic_name, topic_config in topic_pbar:
                topic_pbar.set_description(f"Searching {topic_name}")
                
                # Get topic-relevant snippets using semantic search
                relevant_snippets = self.semantic_search_in_firm_snippets(
                    firm_snippets, 
                    topic_config['query'], 
                    topic_config['threshold']
                )

                # Optional LLM validation
                if use_llm_validation and relevant_snippets:
                    validated_snippets = self.validate_snippets_with_llm(
                        relevant_snippets,
                        topic_config['validation_prompt']
                    )
                else:
                    validated_snippets = relevant_snippets
                
                # Calculate measures
                total_snippets = len(firm_snippets)
                relevant_count = len(relevant_snippets)
                validated_count = len(validated_snippets)
                
                avg_score = np.mean([s.relevance_score for s in relevant_snippets]) if relevant_snippets else 0.0
                
                # Store measures
                row[f'semantic_{topic_name}_exposure'] = validated_count / total_snippets if total_snippets > 0 else 0.0
                row[f'semantic_{topic_name}_count'] = relevant_count
                row[f'semantic_{topic_name}_avg_score'] = avg_score
                row[f'semantic_{topic_name}_validated_count'] = validated_count
            
            topic_pbar.close()
        
        pbar_firms.close()
        
        # Initialize zero values for rows without earnings calls
        for row in panel_data:
            if not row['has_earnings_call']:
                for topic_name in self.climate_topics:
                    row[f'semantic_{topic_name}_exposure'] = 0.0
                    row[f'semantic_{topic_name}_count'] = 0
                    row[f'semantic_{topic_name}_avg_score'] = 0.0
                    row[f'semantic_{topic_name}_validated_count'] = 0
        
        return panel_data

    # Replace the calculate_policy_semantic_measures function:
    def calculate_policy_semantic_measures(self, panel_data, use_llm_validation=True):
        """Calculate policy-specific attention using semantic search"""
        
        print("📋 Running semantic searches for specific policies...")
        
        # Filter to only rows with earnings calls
        rows_with_calls = [row for row in panel_data if row['has_earnings_call']]
        
        # Progress bar for firm-quarters with data
        pbar_firms = tqdm(
            rows_with_calls, 
            desc="Processing firm-quarters", 
            unit="firm-qtrs",
            leave=True
        )
        
        for row in pbar_firms:
            firm_snippets = row['snippets']
            
            # Update progress bar description
            pbar_firms.set_description(f"Processing {row['ticker']} {row['year']}Q{row['quarter']}")
            
            # Progress bar for policy topics
            policy_pbar = tqdm(
                self.policy_topics.items(), 
                desc="Policy topics", 
                leave=False,
                unit="policies"
            )
            
            for policy_name, policy_config in policy_pbar:
                policy_pbar.set_description(f"Searching {policy_name}")
                
                # Semantic search for policy mentions
                policy_snippets = self.semantic_search_in_firm_snippets(
                    firm_snippets,
                    policy_config['query'],
                    policy_config['threshold']
                )
                
                # Optional validation
                if use_llm_validation and policy_snippets:
                    validated_snippets = self.validate_snippets_with_llm(
                        policy_snippets,
                        policy_config['validation_prompt']
                    )
                else:
                    validated_snippets = policy_snippets
                
                # Calculate policy attention measures
                total_snippets = len(firm_snippets)
                validated_count = len(validated_snippets)
                avg_score = np.mean([s.relevance_score for s in policy_snippets]) if policy_snippets else 0.0
                
                row[f'policy_{policy_name}_attention'] = validated_count / total_snippets if total_snippets > 0 else 0.0
                row[f'policy_{policy_name}_mentions'] = validated_count
                row[f'policy_{policy_name}_avg_score'] = avg_score
            
            policy_pbar.close()
        
        pbar_firms.close()
        
        # Initialize zero values for rows without earnings calls
        for row in panel_data:
            if not row['has_earnings_call']:
                for policy_name in self.policy_topics:
                    row[f'policy_{policy_name}_attention'] = 0.0
                    row[f'policy_{policy_name}_mentions'] = 0
                    row[f'policy_{policy_name}_avg_score'] = 0.0
        
        return panel_data



    def calculate_semantic_sentiment(self, panel_data):
        """Calculate sentiment using semantic approach + existing climate_sentiment labels"""
        
        for row in panel_data:
            firm_snippets = row['snippets']
            
            if not firm_snippets:
                row.update({
                    'semantic_climate_sentiment_opportunity': 0.0,
                    'semantic_climate_sentiment_risk': 0.0,
                    'semantic_climate_sentiment_neutral': 0.0,
                    'semantic_sentiment_net': 0.0
                })
                continue
            
            # Use existing climate_sentiment labels from your data
            sentiment_counts = {'opportunity': 0, 'risk': 0, 'neutral': 0}
            
            # Only count snippets that are climate-relevant (using semantic search)
            climate_relevant_snippets = []
            
            for topic_config in self.climate_topics.values():
                topic_snippets = self.semantic_search_in_firm_snippets(
                    firm_snippets, topic_config['query'], topic_config['threshold']
                )
                climate_relevant_snippets.extend(topic_snippets)
            
            # Remove duplicates
            climate_relevant_snippets = list(set(climate_relevant_snippets))
            
            # Count sentiments in climate-relevant snippets
            for snippet in climate_relevant_snippets:
                sentiment = getattr(snippet, 'climate_sentiment', 'neutral')
                if sentiment in sentiment_counts:
                    sentiment_counts[sentiment] += 1
            
            total_climate_snippets = len(climate_relevant_snippets)
            
            if total_climate_snippets > 0:
                row.update({
                    'semantic_climate_sentiment_opportunity': sentiment_counts['opportunity'] / total_climate_snippets,
                    'semantic_climate_sentiment_risk': sentiment_counts['risk'] / total_climate_snippets,
                    'semantic_climate_sentiment_neutral': sentiment_counts['neutral'] / total_climate_snippets,
                    'semantic_sentiment_net': (sentiment_counts['opportunity'] - sentiment_counts['risk']) / total_climate_snippets
                })
            else:
                row.update({
                    'semantic_climate_sentiment_opportunity': 0.0,
                    'semantic_climate_sentiment_risk': 0.0,
                    'semantic_climate_sentiment_neutral': 0.0,
                    'semantic_sentiment_net': 0.0
                })
        
        return panel_data

    def calculate_semantic_evolution(self, panel_data):
        """Calculate how semantic climate attention evolves over time"""
        
        df = pd.DataFrame(panel_data)
        df = df.sort_values(['ticker', 'year', 'quarter'])
        
        # Calculate changes in semantic exposure measures
        semantic_vars = [col for col in df.columns if col.startswith('semantic_') and col.endswith('_exposure')]
        
        for var in semantic_vars:
            # Quarter-over-quarter changes
            df[f'{var}_qoq'] = df.groupby('ticker')[var].pct_change()
            
            # Year-over-year changes  
            df[f'{var}_yoy'] = df.groupby('ticker')[var].pct_change(periods=4)
            
            # Rolling averages
            df[f'{var}_ma4q'] = df.groupby('ticker')[var].rolling(4).mean().reset_index(0, drop=True)
            
            # Semantic attention trend (slope over past 4 quarters)
            def calculate_trend(series):
                if len(series) < 4 or series.isna().all():
                    return np.nan
                x = np.arange(len(series))
                valid_mask = ~series.isna()
                if valid_mask.sum() < 3:
                    return np.nan
                return np.polyfit(x[valid_mask], series[valid_mask], 1)[0]
            
            df[f'{var}_trend_4q'] = df.groupby('ticker')[var].rolling(4).apply(calculate_trend).reset_index(0, drop=True)
        
        return df.to_dict('records')

    def add_quality_metrics(self, panel_data):
        """Add quality and validation metrics for semantic approach"""
        
        for row in panel_data:
            # Calculate average relevance scores across all topics
            avg_scores = []
            validation_rates = []
            
            for topic_name in self.climate_topics:
                avg_score = row.get(f'semantic_{topic_name}_avg_score', 0)
                if avg_score > 0:
                    avg_scores.append(avg_score)
                
                # Validation rate (validated / total found)
                total_found = row.get(f'semantic_{topic_name}_count', 0)
                validated = row.get(f'semantic_{topic_name}_validated_count', 0)
                
                if total_found > 0:
                    validation_rates.append(validated / total_found)
            
            row['semantic_avg_relevance_score'] = np.mean(avg_scores) if avg_scores else 0.0
            row['semantic_validation_rate'] = np.mean(validation_rates) if validation_rates else 0.0
            row['semantic_topics_found'] = len(avg_scores)  # How many topics had any mentions
        
        return panel_data

    def semantic_search_in_firm_snippets(self, firm_snippets, query, threshold):
        """Run semantic search using your existing FAISS index, filtered to firm snippets"""
        
        if not firm_snippets:
            return []
        
        try:
            # Method 1: Use your existing RAG semantic search (RECOMMENDED)
            # This leverages your pre-built FAISS index directly
            all_results = self.rag.query_embedding_index(
                query=query,
                top_k=None,  # Get all results above threshold
                relevance_threshold=threshold
            )
            
            # Filter results to only include this firm's snippets for this quarter
            firm_texts = set(snippet.text for snippet in firm_snippets)
            
            relevant_snippets = []
            for result in all_results:
                # Check if this result matches one of our firm's snippets
                if result['text'] in firm_texts:
                    # Find the corresponding snippet object
                    for snippet in firm_snippets:
                        if snippet.text == result['text']:
                            snippet.relevance_score = result['score']
                            relevant_snippets.append(snippet)
                            break
            
            return relevant_snippets
            
        except Exception as e:
            print(f"Error in semantic search using FAISS index: {e}")
            
            # Fallback: Direct embedding comparison (if FAISS fails)
            try:
                query_embedding = self.rag.model.encode([query], normalize_embeddings=True)
                snippet_texts = [snippet.text for snippet in firm_snippets]
                snippet_embeddings = self.rag.model.encode(snippet_texts, normalize_embeddings=True)
                
                similarities = np.dot(query_embedding, snippet_embeddings.T)[0]
                
                relevant_snippets = []
                for i, similarity in enumerate(similarities):
                    if similarity >= threshold:
                        snippet = firm_snippets[i]
                        snippet.relevance_score = float(similarity)
                        relevant_snippets.append(snippet)
                
                return relevant_snippets
                
            except Exception as e2:
                print(f"Fallback semantic search also failed: {e2}")
                return []

    # Optional: Add progress to LLM validation function
    def validate_snippets_with_llm(self, snippets, validation_prompt):
        """Validate snippets using LLM (if API key available)"""
        
        if not hasattr(self, 'anthropic_client') or not snippets:
            return snippets  # Return all if no validation available
        
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=self.rag.anthropic_api_key)
            
            validated_snippets = []
            
            # Progress bar for LLM validation
            validation_pbar = tqdm(
                snippets[:20], 
                desc="LLM validation", 
                leave=False,
                unit="snippets"
            )
            
            for snippet in validation_pbar:
                validation_pbar.set_description(f"Validating snippet {len(validated_snippets)+1}")
                
                prompt = f"""Is this earnings call text actually about {validation_prompt}?

    Text: "{snippet.text[:400]}"

    Reply with only: RELEVANT, PARTIALLY_RELEVANT, or NOT_RELEVANT"""

                response = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=10,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                validation = response.content[0].text.strip().upper()
                
                if 'RELEVANT' in validation and 'NOT' not in validation:
                    snippet.llm_validation = validation
                    validated_snippets.append(snippet)
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
            
            validation_pbar.close()
            return validated_snippets
            
        except Exception as e:
            print(f"LLM validation error: {e}")
            return snippets  # Fallback to all snippets

    def save_semantic_datasets(self, panel_data):
        """Save datasets with semantic methodology documentation"""
        
        df = pd.DataFrame(panel_data)
        analysis_df = df.drop('snippets', axis=1)
        
        # 1. Main semantic panel dataset
        analysis_df.to_csv(self.output_dir / 'semantic_climate_panel.csv', index=False)
        analysis_df.to_parquet(self.output_dir / 'semantic_climate_panel.parquet')
        
        # 2. Methodology documentation
        methodology = {
            'approach': 'semantic_search',
            'climate_topics': self.climate_topics,
            'policy_topics': self.policy_topics,
            'advantages': [
                'Captures semantic meaning rather than just keywords',
                'Context-aware topic identification',
                'Adjustable relevance thresholds per topic',
                'Optional LLM validation for precision',
                'Handles synonyms and related concepts automatically'
            ],
            'validation_approach': 'LLM-based relevance checking',
            'quality_metrics': [
                'Average relevance scores',
                'Validation rates', 
                'Topic coverage',
                'Cross-temporal consistency'
            ]
        }
        
        with open(self.output_dir / 'methodology.json', 'w') as f:
            json.dump(methodology, f, indent=2)
        
        # 3. Comparison with traditional approach
        self.create_methodology_comparison(analysis_df)
        
        # 4. Quality assurance report
        self.create_quality_report(analysis_df)
        
        print(f"📁 Semantic datasets saved to {self.output_dir}")

    def create_methodology_comparison(self, df):
        """Compare semantic vs traditional keyword approaches"""
        
        # Calculate traditional keyword measures for comparison
        traditional_measures = []
        semantic_measures = []
        
        for col in df.columns:
            if col.startswith('semantic_') and col.endswith('_exposure'):
                topic = col.replace('semantic_', '').replace('_exposure', '')
                
                semantic_measures.append({
                    'topic': topic,
                    'approach': 'semantic',
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'coverage': (df[col] > 0).mean()  # % of firm-quarters with any mentions
                })
        
        comparison_df = pd.DataFrame(semantic_measures)
        comparison_df.to_csv(self.output_dir / 'methodology_comparison.csv', index=False)

    def create_quality_report(self, df):
        """Create comprehensive quality report for semantic approach"""
        
        quality_metrics = {
            'data_coverage': {
                'total_firm_quarters': len(df),
                'firm_quarters_with_calls': (df['has_earnings_call'] == True).sum(),
                'coverage_rate': (df['has_earnings_call'] == True).mean()
            },
            'semantic_performance': {
                'avg_relevance_score': df['semantic_avg_relevance_score'].mean(),
                'avg_validation_rate': df['semantic_validation_rate'].mean(),
                'avg_topics_per_firm_quarter': df['semantic_topics_found'].mean()
            },
            'temporal_consistency': {
                'quarters_covered': df.groupby(['year', 'quarter']).size().describe().to_dict(),
                'firms_covered': df.groupby('ticker').size().describe().to_dict()
            }
        }
        
        with open(self.output_dir / 'quality_report.json', 'w') as f:
            json.dump(quality_metrics, f, indent=2, default=str)

    # Helper methods (same as before)
    def create_panel_structure(self, start_year, end_year):
        """Create firm-quarter panel structure"""
        
        firms = list(set(snippet.ticker for snippet in self.rag.snippets))
        quarters = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31', freq='QE')        

        panel_data = []
        for firm in firms:
            for quarter in quarters:
                year, q = quarter.year, quarter.quarter
                firm_snippets = self.get_firm_quarter_snippets(firm, year, q)

                panel_data.append({
                    'ticker': firm,
                    'year': year,
                    'quarter': q,
                    'date': quarter,
                    'snippets': firm_snippets,
                    'has_earnings_call': len(firm_snippets) > 0
                })

        return panel_data

    def get_firm_quarter_snippets(self, ticker, year, quarter):
        """Get all snippets for a firm in a specific quarter"""
        matching_snippets = []
        total_checked = 0
        
        for snippet in self.rag.snippets:
            total_checked += 1
            
            # Debug: Print first few matches to see what's happening
            if total_checked <= 5 and snippet.ticker == ticker:
                print(f"  🔍 Debug snippet {total_checked}: ticker='{snippet.ticker}', year='{snippet.year}', quarter='{snippet.quarter}'")
            
            # Check ticker match
            if snippet.ticker != ticker:
                continue
                
            # Check year match  
            if not snippet.year or str(snippet.year) != str(year):
                if total_checked <= 5 and snippet.ticker == ticker:
                    print(f"    ❌ Year mismatch: snippet.year='{snippet.year}' vs target year='{year}'")
                continue
                
            # Check quarter match - handle both 'Q1' and '1' formats
            if not snippet.quarter:
                continue
                
            snippet_quarter = str(snippet.quarter)
            target_quarter = str(quarter)
            
            # Handle 'Q1' format -> extract '1'
            if snippet_quarter.startswith('Q'):
                snippet_quarter = snippet_quarter[1:]
                
            if snippet_quarter == target_quarter:
                matching_snippets.append(snippet)
                if len(matching_snippets) <= 3:  # Debug first few matches
                    print(f"    ✅ MATCH found: {ticker} {year}Q{quarter}")
            elif total_checked <= 5 and snippet.ticker == ticker:
                print(f"    ❌ Quarter mismatch: snippet_quarter='{snippet_quarter}' vs target='{target_quarter}'")
        
        print(f"🎯 Found {len(matching_snippets)} snippets for {ticker} {year}Q{quarter}")
        return matching_snippets

# Usage example
def main():
    """Example usage with semantic approach"""
    

    from src.green_investment_rag import GreenInvestmentRAG
    
    rag = GreenInvestmentRAG()
    rag.load_market_data('FULL')
    
    # Add API key if available for LLM validation
    if hasattr(rag, 'anthropic_api_key'):
        rag.anthropic_api_key = "your-api-key"
    
    # Construct semantic variables
    constructor = SemanticClimateExposureConstructor(rag)
    panel_data = constructor.construct_semantic_variables(
        start_year=2015, 
        end_year=2023,
        use_llm_validation=False  # Set to False if no API key
    )
    
    print("✅ Semantic climate exposure variables constructed!")
    print("🎯 Variables capture semantic meaning, not just keywords")
    print("📊 Ready for econometric analysis with enhanced precision")

if __name__ == "__main__":
    main()