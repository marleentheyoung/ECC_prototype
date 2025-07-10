# green_investment_rag.py
# author details

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json

class Snippet:
    def __init__(self, company, ticker, year, quarter, text, speaker, profession, date, climate_sentiment, score=1.0):
        self.company = company
        self.ticker = ticker
        self.year = year
        self.quarter = quarter
        self.text = text
        self.speaker = speaker
        self.profession = profession
        self.date = date
        self.climate_sentiment = climate_sentiment
        self.score = score  # fake relevance score for now
    
    def to_dict(self):
        return {
            "text": self.text,
            "company": self.company,
            "ticker": self.ticker,
            "speaker": self.speaker,
            "profession": self.profession,
            "date": self.date,
            "quarter": self.quarter,
            "year": self.year,
            "climate_sentiment": self.climate_sentiment
        }
    
    @staticmethod
    def from_dict(d):
        print(d)
        return Snippet(
            company=d.get("company"),
            ticker=d.get("ticker"),
            year=d.get("year"),
            quarter=d.get("quarter"),
            text=d.get("text"),
            speaker=d.get("speaker"),
            profession=d.get("profession"),
            date=d.get("date"),
            climate_sentiment=d.get("climate_sentiment")
        )


class GreenInvestmentRAG:
    def __init__(self):
        self.snippets = []
        self.investment_categories = {
        "Renewable Energy": ["solar", "wind", "hydro", "renewable"],
        "Electric Vehicles": ["EV", "electric vehicle", "battery"],
        "Energy Efficiency": ["efficiency", "savings", "optimization"],
        "Climate Strategy": ["climate", "sustainability", "net zero", "ESG"],
        "Green Finance": [
            "green bond", "sustainability-linked bond", "sustainability-linked loan", "sustainable finance",
            "green finance", "ESG financing", "climate finance", "transition finance",
            "green investment", "sustainable investment", "responsible investment",
            "climate risk disclosure", "climate reporting", "ESG reporting", "non-financial disclosure",
            "carbon pricing", "internal carbon price", "carbon credits", "emissions trading",
            "taxonomy aligned", "EU taxonomy", "sustainable debt", "green capital",
            "climate resilience finance", "impact investing", "social bond", "blue bond"
        ]
    }

    def load_earnings_data(self, data):
        for doc in data:
            for text_data in doc['texts']:
                snippet = Snippet(
                    company=doc['company_name'],
                    ticker=doc['ticker'],
                    year=doc['year'],
                    quarter=doc['quarter'],
                    text=text_data['text'],
                    speaker=text_data['speaker'],
                    profession=text_data['profession'],
                    date=doc['date'],
                    climate_sentiment=text_data.get('climate_sentiment', 'neutral'),
                )
                self.snippets.append(snippet)

    def build_embedding_index(self, embedding_model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """Build a FAISS index for semantic search over snippets."""
        print(f"Building embedding index using model: {embedding_model_name}")
        self.model = SentenceTransformer(embedding_model_name)

        # Get the text of each snippet
        texts = [s.text for s in self.snippets]

        # Compute dense embeddings
        self.embeddings = self.model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        
        # Build FAISS index (cosine similarity via inner product)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        print(f"FAISS index built with {self.index.ntotal} snippets.")

    def _apply_global_filters(self, results, selected_companies=None, sentiment_filter="All", year_range=None):
        """Apply global filtering to any result set."""
        if not results:
            return []
        
        filtered = results.copy()
        
        # Filter by company
        if selected_companies:
            filtered = [r for r in filtered if r.get('ticker') in selected_companies]
        
        # Filter by sentiment
        if sentiment_filter and sentiment_filter != "All":
            filtered = [r for r in filtered if r.get('climate_sentiment') == sentiment_filter]
        
        # Filter by year range
        if year_range:
            filtered = [
                r for r in filtered
                if r.get('year') and str(r['year']).isdigit() and
                year_range[0] <= int(r['year']) <= year_range[1]
            ]
        
        return filtered

    def query_embedding_index(self, query, top_k=10, selected_companies=None, sentiment_filter="All", year_range=None, relevance_threshold=0.45):
        """Semantic search for a query in the embedding index with relevance threshold and global filtering."""
        if not hasattr(self, 'index') or not hasattr(self, 'model'):
            raise RuntimeError("You must build the embedding index first using build_embedding_index().")

        # Embed the query
        query_emb = self.model.encode([query], normalize_embeddings=True)

        # Search in the FAISS index - get all results if top_k is None
        if top_k is None:
            search_k = len(self.snippets)  # Get all snippets
        else:
            search_k = min(len(self.snippets), top_k * 10)  # Get more results to account for filtering
        
        scores, indices = self.index.search(np.array(query_emb), search_k)

        # Format results and apply relevance threshold
        results = []
        for idx, score in zip(indices[0], scores[0]):
            # Only include results above the relevance threshold
            if float(score) >= relevance_threshold:
                snippet = self.snippets[idx]
                results.append({
                    'company': snippet.company,
                    'ticker': snippet.ticker,
                    'text': snippet.text,
                    'speaker': snippet.speaker,
                    'profession': snippet.profession,
                    'score': float(score),
                    'date': snippet.date,
                    'quarter': snippet.quarter,
                    'year': snippet.year,
                    'climate_sentiment': snippet.climate_sentiment,
                })

        # Apply global filtering
        filtered_results = self._apply_global_filters(results, selected_companies, sentiment_filter, year_range)
        
        # Return all results if top_k is None, otherwise limit
        if top_k is None:
            return filtered_results
        else:
            return filtered_results[:top_k]

    def search_by_category(self, category, top_k=10, selected_companies=None, sentiment_filter="All", year_range=None):
        """Search by category with global filtering."""
        keywords = self.investment_categories.get(category, [])
        results = [
            {
                'company': s.company,
                'ticker': s.ticker,
                'text': s.text,
                'speaker': s.speaker,
                'profession': s.profession,
                'score': s.score,
                'date': s.date,
                'quarter': s.quarter,
                'year': s.year,
                'climate_sentiment': s.climate_sentiment,
            }
            for s in self.snippets if any(k.lower() in s.text.lower() for k in keywords)
        ]
        
        # Apply global filtering
        filtered_results = self._apply_global_filters(results, selected_companies, sentiment_filter, year_range)
        
        # Return all results if top_k is None, otherwise limit
        if top_k is None:
            return filtered_results
        else:
            return filtered_results[:top_k]

    def search_by_query(self, query, top_k=10, selected_companies=None, sentiment_filter="All", year_range=None):
        """Search by query with global filtering."""
        results = [
            {
                'company': s.company,
                'ticker': s.ticker,
                'text': s.text,
                'speaker': s.speaker,
                'profession': s.profession,
                'score': s.score,
                'date': s.date,
                'quarter': s.quarter,
                'year': s.year,
                'climate_sentiment': s.climate_sentiment,
            }
            for s in self.snippets if query.lower() in s.text.lower()
        ]
        
        # Apply global filtering
        filtered_results = self._apply_global_filters(results, selected_companies, sentiment_filter, year_range)
        
        # Return all results if top_k is None, otherwise limit
        if top_k is None:
            return filtered_results
        else:
            return filtered_results[:top_k]

    def get_investment_summary(self, category, selected_companies=None, sentiment_filter="All", year_range=None):
        """Get investment summary with global filtering."""
        keywords = self.investment_categories.get(category, [])
        filtered = [s for s in self.snippets if any(k.lower() in s.text.lower() for k in keywords)]
        
        # Convert snippets to dict format for filtering
        results = [
            {
                'company': s.company,
                'ticker': s.ticker,
                'text': s.text,
                'speaker': s.speaker,
                'profession': s.profession,
                'score': s.score,
                'date': s.date,
                'quarter': s.quarter,
                'year': s.year,
                'climate_sentiment': s.climate_sentiment,
            }
            for s in filtered
        ]
        
        # Apply global filtering
        filtered_results = self._apply_global_filters(results, selected_companies, sentiment_filter, year_range)
        
        companies = {}
        for r in filtered_results:
            companies[r['ticker']] = companies.get(r['ticker'], 0) + 1
        
        return {
            'total_mentions': len(filtered_results),
            'companies_mentioned': len(companies),
            'company_breakdown': companies
        }

    def filter_by_company(self, results, companies):
        """Legacy method - now use global filtering in search methods."""
        return [r for r in results if r['ticker'] in companies]
    
    def save_index(self, path='climate_index.faiss'):
        faiss.write_index(self.index, path)

    def load_index(self, path='climate_index.faiss', embedding_model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(embedding_model_name)
        self.index = faiss.read_index(path)

    def save_snippets(self, path='climate_snippets.json'):
        with open(path, 'w') as f:
            json.dump([s.to_dict() for s in self.snippets], f)

    def load_snippets(self, path='climate_snippets.json'):
        with open(path) as f:
            data = json.load(f)
        self.snippets = [Snippet.from_dict(d) for d in data]

    def load_market_data(self, market='EU'):
        """Load market-specific data and index files."""
        if market == 'EU':
            index_path = 'climate_index_STOXX600.faiss'
            snippets_path = 'climate_snippets_STOXX600.json'
        elif market == 'US':
            index_path = 'climate_index_SP500.faiss'
            snippets_path = 'climate_snippets_SP500.json'
        else:
            raise ValueError("Market must be 'EU' or 'US'")
        
        self.load_snippets(snippets_path)
        self.load_index(index_path)

    def load_combined_data(self):
        """Load both EU and US market data."""
        # Load EU data
        eu_rag = GreenInvestmentRAG()
        eu_rag.load_market_data('EU')
        
        # Load US data
        us_rag = GreenInvestmentRAG()
        us_rag.load_market_data('US')
        
        # Combine snippets
        self.snippets = eu_rag.snippets + us_rag.snippets
        
        # For combined data, we need to rebuild the index since we're combining two indices
        self.build_embedding_index()