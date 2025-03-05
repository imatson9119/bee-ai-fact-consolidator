#!/usr/bin/env python3
"""
Test script for the fact consolidator without requiring API credentials.
Uses a set of sample facts to demonstrate clustering and consolidation.
"""

import json
import sys
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fact-consolidator-test")

# Sample facts
SAMPLE_FACTS = [
    {"id": 1, "text": "Ian is friends with someone named Scott"},
    {"id": 2, "text": "Ian has a friend named Scott"},
    {"id": 3, "text": "Scott is friends with Ian"},
    {"id": 4, "text": "Ian's friend Scott is the best man at his wedding"},
    {"id": 5, "text": "Ian lives in Seattle"},
    {"id": 6, "text": "Ian is from Seattle, WA"},
    {"id": 7, "text": "Ian is a resident of Seattle"},
    {"id": 8, "text": "Ian loves coffee"},
    {"id": 9, "text": "Ian enjoys drinking coffee"},
    {"id": 10, "text": "Ian is a coffee enthusiast"},
    {"id": 11, "text": "Ian works as a software engineer"},
    {"id": 12, "text": "Ian's job is software engineering"},
    {"id": 13, "text": "Ian is a programmer by profession"},
    {"id": 14, "text": "Ian plays the guitar"},
    {"id": 15, "text": "Ian knows how to play guitar"}
]


def mock_llm_client():
    """Creates a mock version of the LLM client that returns predefined responses."""
    class MockLLMClient:
        def consolidate_facts(self, facts):
            if any("Scott" in fact for fact in facts):
                return "- Ian is close friends with Scott\n- Ian's friend Scott is the best man at his wedding"
            elif any("Seattle" in fact for fact in facts):
                return "- Ian lives in Seattle, Washington"
            elif any("coffee" in fact for fact in facts):
                return "- Ian is an avid coffee enthusiast"
            elif any("software" in fact for fact in facts or "engineer" in fact for fact in facts):
                return "- Ian works as a software engineer"
            elif any("guitar" in fact for fact in facts):
                return "- Ian plays the guitar"
            else:
                return "\n".join([f"- {fact}" for fact in facts])
    
    return MockLLMClient()


def run_test():
    """Run a test of the fact consolidation process using sample data."""
    try:
        # Try to import the FactConsolidator class
        from fact_consolidator import FactConsolidator
        import text_similarity
    except ImportError:
        logger.error("Could not import FactConsolidator. Make sure fact_consolidator.py is in the current directory.")
        sys.exit(1)
    
    # Create a mock fact client
    class MockFactClient:
        pass
    
    # Get a mock LLM client
    llm_client = mock_llm_client()
    
    # Create the consolidator
    consolidator = FactConsolidator(MockFactClient(), llm_client)
    
    # Test with agglomerative clustering (default)
    print("=== Testing with Agglomerative Clustering ===")
    clusters = consolidator.cluster_facts(SAMPLE_FACTS, use_agglomerative=True, similarity_threshold=0.5)
    print(f"Found {len(clusters)} clusters")
    
    for i, (cluster_id, cluster_facts) in enumerate(clusters.items()):
        print(f"\nCluster {i+1}: {len(cluster_facts)} facts")
        for fact in cluster_facts:
            print(f"- {fact['text']}")
        
        # Get consolidated facts for this cluster if it has more than 1 fact
        if len(cluster_facts) > 1:
            fact_texts = [fact["text"] for fact in cluster_facts]
            consolidated = llm_client.consolidate_facts(fact_texts)
            print("\nConsolidated:")
            print(consolidated)
    
    # Test with k-means clustering
    print("\n\n=== Testing with K-means Clustering ===")
    clusters = consolidator.cluster_facts(SAMPLE_FACTS, use_agglomerative=False, n_clusters=5)
    print(f"Found {len(clusters)} clusters")
    
    for i, (cluster_id, cluster_facts) in enumerate(clusters.items()):
        print(f"\nCluster {i+1}: {len(cluster_facts)} facts")
        for fact in cluster_facts:
            print(f"- {fact['text']}")
        
        # Get consolidated facts for this cluster if it has more than 1 fact
        if len(cluster_facts) > 1:
            fact_texts = [fact["text"] for fact in cluster_facts]
            consolidated = llm_client.consolidate_facts(fact_texts)
            print("\nConsolidated:")
            print(consolidated)


if __name__ == "__main__":
    run_test() 