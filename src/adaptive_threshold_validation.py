# adaptive_threshold_validation.py - Adaptive threshold validation with user feedback
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
import pandas as pd
from typing import List, Dict, Tuple, Optional
from anthropic import Anthropic
import time

class AdaptiveThresholdValidator:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or st.session_state.get('anthropic_api_key')
        self.client = Anthropic(api_key=self.api_key) if self.api_key else None
        
        # Configuration
        self.sample_size = 20
        self.irrelevant_threshold = 0.25  # 25% irrelevant = threshold too low
        self.threshold_step = 0.05
        self.max_threshold = 0.80
        self.max_iterations = 6
        
    def adaptive_threshold_search(self, rag, topic_name: str, query: str, 
                                initial_threshold: float = 0.30) -> Dict:
        """
        Perform adaptive threshold search with LLM validation and user feedback.
        
        Returns:
            Dict with final_threshold, validated_snippets, validation_history
        """
        if not self.client:
            st.error("No API key provided for LLM validation.")
            return None
        
        current_threshold = initial_threshold
        validation_history = []
        iteration = 0
        
        st.subheader(f"🔍 Adaptive Threshold Search for: {topic_name}")
        progress_placeholder = st.empty()
        
        while iteration < self.max_iterations and current_threshold <= self.max_threshold:
            iteration += 1
            
            with progress_placeholder.container():
                st.write(f"**Iteration {iteration}**: Testing threshold {current_threshold:.2f}")
                
                # Get snippets at current threshold
                search_results = rag.query_embedding_index(
                    query, 
                    top_k=None, 
                    relevance_threshold=current_threshold
                )
                
                if len(search_results) < 5:
                    st.warning(f"Only {len(search_results)} snippets found at threshold {current_threshold:.2f}. Threshold too high.")
                    if iteration > 1:
                        # Revert to previous threshold
                        current_threshold -= self.threshold_step
                        st.info(f"Reverting to threshold {current_threshold:.2f}")
                    break
                
                # Get boundary snippets (lowest scoring ones that passed threshold)
                sorted_results = sorted(search_results, key=lambda x: x['score'])
                boundary_sample = sorted_results[:min(self.sample_size, len(sorted_results))]
                
                st.write(f"Found {len(search_results)} snippets. Validating {len(boundary_sample)} boundary snippets...")
                
                # LLM validation of boundary snippets
                validation_results = self._validate_boundary_snippets(
                    boundary_sample, topic_name, query
                )
                
                # Calculate irrelevant percentage
                irrelevant_count = sum(1 for result in validation_results 
                                     if result['validation'] == 'NOT_RELEVANT')
                irrelevant_percentage = irrelevant_count / len(validation_results)
                
                # Store iteration results
                iteration_data = {
                    'iteration': iteration,
                    'threshold': current_threshold,
                    'total_snippets': len(search_results),
                    'validated_sample': len(boundary_sample),
                    'irrelevant_count': irrelevant_count,
                    'irrelevant_percentage': irrelevant_percentage,
                    'validation_results': validation_results
                }
                validation_history.append(iteration_data)
                
                # Display results and get user feedback
                decision = self._display_validation_results_and_get_feedback(
                    iteration_data, topic_name
                )
                
                if decision == "threshold_good":
                    st.success(f"✅ Threshold {current_threshold:.2f} accepted!")
                    break
                elif decision == "increase_threshold":
                    current_threshold += self.threshold_step
                    st.info(f"⬆️ Increasing threshold to {current_threshold:.2f}")
                elif decision == "manual_override":
                    st.info("🔧 User manually selected threshold")
                    break
                else:
                    st.error("Unexpected decision. Stopping.")
                    break
        
        # Final search with selected threshold
        final_results = rag.query_embedding_index(
            query, 
            top_k=None, 
            relevance_threshold=current_threshold
        )
        
        # Convert to snippets
        final_snippets = self._convert_results_to_snippets(rag, final_results)
        
        return {
            'final_threshold': current_threshold,
            'validated_snippets': final_snippets,
            'validation_history': validation_history,
            'total_api_calls': sum(len(h['validation_results']) for h in validation_history)
        }
    
    def _validate_boundary_snippets(self, snippets: List[Dict], topic_name: str, 
                                   query: str) -> List[Dict]:
        """Validate boundary snippets using LLM."""
        validation_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, snippet in enumerate(snippets):
            # Update progress
            progress_bar.progress((i + 1) / len(snippets))
            status_text.text(f"Validating snippet {i+1}/{len(snippets)}...")
            
            validation = self._validate_single_snippet(snippet, topic_name, query)
            
            validation_results.append({
                'snippet': snippet,
                'validation': validation,
                'user_override': None  # Will be set by user feedback
            })
            
            # Longer delay to avoid rate limiting
            time.sleep(0.5)
        
        progress_bar.empty()
        status_text.empty()
        return validation_results
    
    def _validate_single_snippet(self, snippet: Dict, topic_name: str, query: str) -> str:
        """Validate a single snippet using Claude."""
        prompt = f"""Analyze this earnings call text for relevance to "{topic_name}":

Text: "{snippet['text'][:400]}"
Company: {snippet['company']}

Is this text about {topic_name}? Reply with ONLY one word:
RELEVANT
PARTIALLY_RELEVANT
NOT_RELEVANT"""

        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=10,  # Much lower limit
                temperature=0.0,  # No randomness
                messages=[{"role": "user", "content": prompt}]
            )
            
            validation = response.content[0].text.strip()
            
            # Extract just the first word/line
            validation = validation.split('\n')[0].split(' ')[0].upper().strip()
            
            # Handle common variations
            if 'RELEVANT' in validation and 'NOT' not in validation and 'PARTIALLY' not in validation:
                return "RELEVANT"
            elif 'PARTIALLY' in validation or 'PARTIAL' in validation:
                return "PARTIALLY_RELEVANT"
            elif 'NOT' in validation or validation.startswith('N'):
                return "NOT_RELEVANT"
            
            # If still unclear, default based on content
            full_response = response.content[0].text.upper()
            if 'NOT_RELEVANT' in full_response or 'NOT RELATED' in full_response:
                return "NOT_RELEVANT"
            elif 'PARTIALLY' in full_response:
                return "PARTIALLY_RELEVANT"
            elif 'RELEVANT' in full_response:
                return "RELEVANT"
            
            print(f"Unclear validation response: '{validation}' -> defaulting to NOT_RELEVANT")
            return "NOT_RELEVANT"
            
        except Exception as e:
            print(f"Error validating snippet: {str(e)}")
            return "VALIDATION_ERROR"
    
    def _display_validation_results_and_get_feedback(self, iteration_data: Dict, 
                                                   topic_name: str) -> str:
        """Display validation results and get user feedback."""
        st.markdown("---")
        st.subheader(f"📊 Validation Results - Iteration {iteration_data['iteration']}")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Threshold", f"{iteration_data['threshold']:.2f}")
        with col2:
            st.metric("Total Snippets", iteration_data['total_snippets'])
        with col3:
            st.metric("Irrelevant", f"{iteration_data['irrelevant_count']}/{iteration_data['validated_sample']}")
        with col4:
            st.metric("Irrelevant %", f"{iteration_data['irrelevant_percentage']:.1%}")
        
        # Quality assessment
        if iteration_data['irrelevant_percentage'] <= self.irrelevant_threshold:
            st.success(f"✅ Quality check passed! ≤{self.irrelevant_threshold:.0%} irrelevant snippets")
            recommended_action = "threshold_good"
        else:
            st.warning(f"⚠️ Quality check failed. >{self.irrelevant_threshold:.0%} irrelevant snippets")
            recommended_action = "increase_threshold"
        
        # Show detailed validation results
        st.subheader("🔍 Sample Validation Details")
        
        validation_df = []
        for i, result in enumerate(iteration_data['validation_results']):
            snippet = result['snippet']
            validation_df.append({
                '#': i + 1,
                'Score': f"{snippet['score']:.3f}",
                'LLM Validation': result['validation'],
                'Company': snippet['company'],
                'Text Preview': snippet['text'][:100] + "..." if len(snippet['text']) > 100 else snippet['text']
            })
        
        df = pd.DataFrame(validation_df)
        st.dataframe(df, use_container_width=True)
        
        # User feedback section
        st.subheader("👤 User Feedback & Decision")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Review the validation results above. What would you like to do?**")
            
            feedback_options = [
                "Follow LLM recommendation",
                "Override: Accept current threshold anyway",
                "Override: Force threshold increase",
                "Stop and use current threshold"
            ]
            
            user_choice = st.radio(
                "Your decision:",
                feedback_options,
                index=0 if recommended_action == "threshold_good" else 1,
                key=f"decision_iteration_{iteration_data['iteration']}"
            )
        
        with col2:
            st.write("**LLM Recommendation:**")
            if recommended_action == "threshold_good":
                st.success("✅ Accept threshold")
            else:
                st.warning("⬆️ Increase threshold")
        
        # Process user decision
        if user_choice == "Follow LLM recommendation":
            return recommended_action
        elif user_choice == "Override: Accept current threshold anyway":
            return "manual_override"
        elif user_choice == "Override: Force threshold increase":
            return "increase_threshold"
        elif user_choice == "Stop and use current threshold":
            return "manual_override"
        
        return recommended_action
    

    
    def _convert_results_to_snippets(self, rag, results: List[Dict]) -> List:
        """Convert search results back to snippet objects."""
        snippets = []
        for result in results:
            for snippet in rag.snippets:
                if (snippet.text == result['text'] and 
                    snippet.company == result['company'] and
                    snippet.ticker == result['ticker']):
                    snippet.score = result['score']
                    snippets.append(snippet)
                    break
        return snippets

def display_threshold_search_results(validation_result: Dict):
    """Display the final results of adaptive threshold search."""
    if not validation_result:
        return
    
    st.markdown("---")
    st.subheader("🎯 Final Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Final Threshold", f"{validation_result['final_threshold']:.2f}")
    with col2:
        st.metric("Final Snippets", len(validation_result['validated_snippets']))
    with col3:
        st.metric("Total API Calls", validation_result['total_api_calls'])
    
    # History table
    st.subheader("📊 Validation History")
    history_data = []
    for h in validation_result['validation_history']:
        history_data.append({
            'Iteration': h['iteration'],
            'Threshold': f"{h['threshold']:.2f}",
            'Total Snippets': h['total_snippets'],
            'Validated Sample': h['validated_sample'],
            'Irrelevant': f"{h['irrelevant_count']}/{h['validated_sample']}",
            'Irrelevant %': f"{h['irrelevant_percentage']:.1%}",
            'Quality Check': "✅ Pass" if h['irrelevant_percentage'] <= 0.25 else "❌ Fail"
        })
    
    history_df = pd.DataFrame(history_data)
    st.dataframe(history_df, use_container_width=True)
    
    # Export option
    st.subheader("📥 Export Results")
    if st.button("Download Validated Snippets"):
        export_data = []
        for snippet in validation_result['validated_snippets']:
            export_data.append({
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
                'final_threshold': validation_result['final_threshold']
            })
        
        df = pd.DataFrame(export_data)
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Validated Results as CSV",
            data=csv,
            file_name=f"adaptive_threshold_results_{validation_result['final_threshold']:.2f}.csv",
            mime="text/csv"
        )