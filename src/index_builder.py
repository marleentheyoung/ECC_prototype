from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
import os

# Path to your folder
path = '/Users/marleendejonge/Library/CloudStorage/OneDrive-UvA/PhD/PhD planning/Papers/PaperI/ECC transcripts/ECC_climate_risk/data/climate_segments/BERT/SP500'

# Load your model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Collect all passages AND metadata
passages = []
snippet_data = []

for ext in ['SP500', 'STOXX600']:
    path = f'/Users/marleendejonge/Library/CloudStorage/OneDrive-UvA/PhD/PhD planning/Papers/PaperI/ECC transcripts/ECC_climate_risk/data/climate_segments/BERT/{ext}'

    for filename in os.listdir(path):
        if filename.endswith('.json'):
            full_path = os.path.join(path, filename)
            # Open and load the JSON file
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for transcript in data:
                snippets = transcript.get('texts', [])
                
                # Extract metadata from transcript level
                company = transcript.get('company_name', '')
                ticker = transcript.get('ticker', '')
                year = transcript.get('year', '')
                quarter = transcript.get('quarter', '')
                date = transcript.get('date', '')
                
                for snipp in snippets:
                    text = snipp.get('text', '')
                    if text:  # Only add non-empty texts
                        passages.append(text)
                        
                        # Create complete snippet data
                        snippet_dict = {
                            'text': text,
                            'company': company,
                            'ticker': ticker,
                            'year': year,
                            'quarter': quarter,
                            'date': date,
                            'speaker': snipp.get('speaker', ''),
                            'profession': snipp.get('profession', ''),
                            'climate_sentiment': snipp.get('climate_sentiment', 'neutral')
                        }
                        snippet_data.append(snippet_dict)

print(f"Collected {len(passages)} passages with metadata")

# Encode all passages in one go
embeddings = model.encode(passages, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

# Save embeddings and complete snippet data
np.save('data/climate_passages_full.npy', embeddings)
with open('data/climate_snippets_full.json', 'w') as f:
    json.dump(snippet_data, f, indent=2)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Use inner product for cosine similarity (with normalized embeddings)
index.add(embeddings)

# Save the index
faiss.write_index(index, 'data/climate_index_full.faiss')

print(f"Saved {len(snippet_data)} snippets with embeddings and FAISS index")