# analysis/run_semantic_variables.py - Runner script with proper path setup

import sys
import os
from pathlib import Path

# Get the project root directory (parent of analysis/)
project_root = Path(__file__).parent.parent
src_dir = project_root / 'src'

# Add both project root and src to Python path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

print(f"Project root: {project_root}")
print(f"Source directory: {src_dir}")
print(f"Python path: {sys.path[:3]}")  # Show first 3 entries

# Now import your modules
try:
    from src.green_investment_rag import GreenInvestmentRAG
    print("✅ Successfully imported GreenInvestmentRAG")
    
    # Import from the semantic_variables.py file in the analysis directory
    import semantic_variables
    SemanticClimateExposureConstructor = semantic_variables.SemanticClimateExposureConstructor
    print("✅ Successfully imported SemanticClimateExposureConstructor")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📁 Available files in src/:")
    if src_dir.exists():
        for file in src_dir.glob("*.py"):
            print(f"  - {file.name}")
    else:
        print("  - src/ directory not found!")
    sys.exit(1)

def main():
    """Run semantic variable construction with proper path setup"""
    
    print("🌱 Green Investment Analyzer - Semantic Variables")
    print("=" * 50)
    
    # Configuration
    MARKET = 'FULL'  # Options: 'FULL', 'EU', 'US'
    START_YEAR = 2009
    END_YEAR = 2025
    USE_LLM_VALIDATION = False  # Set to True if you have API key
    MAX_TEST_FIRMS = 3          # NEW: Limit to 3 firms
    MAX_SNIPPETS_PER_FIRM = 5  # NEW: Max 5 snippets per firm

    # Step 1: Load RAG system
    print(f"🔄 Loading {MARKET} market data...")
    
    try:
        rag = GreenInvestmentRAG()
        rag.load_market_data(MARKET)
        print(f"✅ Loaded {len(rag.snippets):,} snippets")
        
        # LIMIT DATA FOR TESTING
        # print(f"🧪 TEST MODE: Limiting to {MAX_TEST_FIRMS} firms with max {MAX_SNIPPETS_PER_FIRM} snippets each...")
        # unique_firms = list(set(s.ticker for s in rag.snippets))[:MAX_TEST_FIRMS]
        unique_firms = list(set(s.ticker for s in rag.snippets))
        test_snippets = []
        for firm in unique_firms:
            # firm_snippets = [s for s in rag.snippets if s.ticker == firm][:MAX_SNIPPETS_PER_FIRM]
            firm_snippets = [s for s in rag.snippets if s.ticker == firm]
            test_snippets.extend(firm_snippets)
        rag.snippets = test_snippets
        print(f"✅ Reduced to {len(rag.snippets)} snippets from firms: {unique_firms}")

        # 🔍 DEBUG: Check snippet structure
        if rag.snippets:            
            # Check for empty tickers
            empty_tickers = [s for s in rag.snippets[:10] if not s.ticker or s.ticker.strip() == '']
            print(f"  Empty tickers in first 10: {len(empty_tickers)}")
            
            # Check unique tickers
            unique_tickers = set(s.ticker for s in rag.snippets[:100] if s.ticker)
            print(f"  Sample tickers: {list(unique_tickers)[:10]}")
            
            # Check year/quarter format
            year_quarter_samples = [(s.year, s.quarter) for s in rag.snippets[:5]]
            print(f"  Year-quarter samples: {year_quarter_samples}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("🔍 Check if these files exist:")
        data_dir = project_root / 'data'
        if data_dir.exists():
            print(f"  - {data_dir}/climate_index_full.faiss")
            print(f"  - {data_dir}/climate_snippets_full.json")
        return
    
    # Step 2: Setup output directory
    output_dir = project_root / 'output' / 'firm_variables'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Step 3: Initialize constructor
    print("🏗️ Initializing semantic constructor...")
    constructor = SemanticClimateExposureConstructor(rag, output_dir=output_dir)
    
    # Step 4: Run construction
    print("🧠 Running semantic variable construction...")
    try:
        panel_data = constructor.construct_semantic_variables(
            start_year=START_YEAR,
            end_year=END_YEAR,
            use_llm_validation=USE_LLM_VALIDATION
        )
        
        print("✅ Success!")
        print(f"📊 Generated {len(panel_data):,} firm-quarter observations")
        print(f"📁 Files saved to: {output_dir}")
        
        # Show some sample data
        if panel_data:
            sample = panel_data[0]
            print(f"\n📝 Sample row for {sample['ticker']} {sample['year']}Q{sample['quarter']}:")
            for key, value in sample.items():
                if key != 'snippets' and isinstance(value, (int, float)):
                    print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error in construction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()