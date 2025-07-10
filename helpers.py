def load_data():
    """Load your JSON data"""
    # Replace with your actual data loading logic
    # For now, using the sample from your document
    sample_data = [
        {
            "file": "Norfolk_Southern_Q3_2010.pdf",
            "company_name": "Norfolk Southern Corp.",
            "ticker": "NSC",
            "quarter": "Q3",
            "year": "2010",
            "texts": [
                {
                    "speaker": "Operator",
                    "profession": "Operator",
                    "text": "Coal burn increased during the summer during well above average temperatures in the East and improved economic activity.",
                    "climate_sentiment": "opportunity"
                },
                {
                    "speaker": "Operator",
                    "profession": "Operator",
                    "text": "Our average price per gallon of diesel fuel was 2.19 an 18% increase compared with the third quarter of 2009.",
                    "climate_sentiment": "opportunity"
                }
            ],
            "climate_risk_sentences": {
                "physical": ["Coal burn increased during the summer during well above average temperatures in the East and improved economic activity."],
                "transition": ["Our average price per gallon of diesel fuel was 2.19 an 18% increase compared with the third quarter of 2009."]
            },
            "date": "2010-10-27"
        },
        {
            "file": "GM_Q3_2011.pdf",
            "company_name": "General Motors Co. (GM)",
            "ticker": "GM",
            "quarter": "Q3",
            "year": "2011",
            "texts": [
                {
                    "speaker": "Daniel F. Akerson",
                    "profession": "Chairman & CEO",
                    "text": "We signed a new agreement with SAIC for electric vehicle development. We intend to lead the way in advanced technology.",
                    "climate_sentiment": "opportunity"
                },
                {
                    "speaker": "Daniel F. Akerson",
                    "profession": "Chairman & CEO",
                    "text": "We also signed an agreement with LG Group to jointly design and engineer future electric vehicles. This will help us expand the number and types of electric vehicles we can offer by leveraging LG's proven expertise in batteries.",
                    "climate_sentiment": "opportunity"
                }
            ],
            "climate_risk_sentences": {
                "physical": [],
                "transition": []
            },
            "date": "2011-11-09"
        }
    ]
    return sample_data


def filter_results(results, selected_companies=None, sentiment_filter="All", year_range=None):
    """Filter the results based on company, sentiment, and year range."""
    # This function is now deprecated - use the global filtering in the RAG class methods
    if not results:
        return []
    
    filtered = results.copy()
    
    # Filter by company
    if selected_companies:
        filtered = [r for r in filtered if r.get('ticker') in selected_companies]
    
    # Filter by sentiment
    if sentiment_filter and sentiment_filter != "All":
        filtered = [r for r in filtered if r.get('climate_sentiment') == sentiment_filter]
    print(len(filtered))
    # Filter by year range
    if year_range:
        filtered = [
            r for r in filtered
            if r.get('year') and str(r['year']).isdigit() and
            year_range[0] <= int(r['year']) <= year_range[1]
        ]
    print(len(filtered))
    
    return filtered