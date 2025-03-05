#!/usr/bin/env python3
"""
Test script for the LLM integration with fact consolidation.
This script sends sample facts directly to the local LLM to test the integration.
"""

import os
import json
import argparse
from openai import OpenAI
import logging

# Default LLM API endpoint
DEFAULT_LLM_API_URL = "http://localhost:1234/v1"


def test_llm_connection(api_url=DEFAULT_LLM_API_URL):
    """Test the connection to the LLM."""
    client = OpenAI(base_url=api_url, api_key="not-needed")
    
    try:
        # Simple test message
        response = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, are you connected and working?"}
            ],
            temperature=0.3,
            max_tokens=100
        )
        print("LLM Connection Test:")
        print(f"API URL: {api_url}")
        print(f"Response: {response.choices[0].message.content}")
        print("Connection successful!\n")
        return True
    except Exception as e:
        print(f"Error connecting to LLM: {str(e)}")
        print("Please make sure LM Studio is running and the API is accessible.")
        return False


def test_fact_consolidation(facts, api_url=DEFAULT_LLM_API_URL):
    """Test the fact consolidation by sending sample facts to the LLM."""
    client = OpenAI(base_url=api_url, api_key="not-needed")
    
    facts_text = "\n".join([f"- {fact['text']}" for fact in facts])
    
    prompt = f"""
    I have a set of similar personal facts that need to be consolidated 
    and deduplicated. Please combine these facts into one or more consolidated, 
    accurate statements that preserve all the important information.
    
    Original facts:
    {facts_text}
    
    Consolidated fact(s):
    """
    
    try:
        print("Sending facts for consolidation...")
        response = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that consolidates similar facts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        print("\nOriginal Facts:")
        for i, fact in enumerate(facts):
            print(f"{i+1}. {fact['text']}")
        
        print("\nConsolidated Facts:")
        consolidated = response.choices[0].message.content.strip()
        print(consolidated)
        
        return consolidated
    except Exception as e:
        print(f"Error during fact consolidation: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Test LLM integration for fact consolidation")
    parser.add_argument("--sample-file", type=str, default="sample_facts.json", help="JSON file with sample facts")
    parser.add_argument("--api-url", type=str, default=DEFAULT_LLM_API_URL, help="URL for the local LLM API")
    parser.add_argument("--topic", type=str, help="Test with a specific topic of facts (friends, location, coffee, etc.)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Test connection
    if not test_llm_connection(args.api_url):
        return 1
    
    try:
        # Load sample facts
        if os.path.exists(args.sample_file):
            with open(args.sample_file, 'r') as f:
                all_facts = json.load(f)
        else:
            # If sample file doesn't exist, generate it
            print(f"Sample file {args.sample_file} not found, generating it...")
            from generate_sample_facts import generate_facts
            all_facts = generate_facts(output_file=args.sample_file)

        # If topic specified, filter facts
        if args.topic:
            from generate_sample_facts import SAMPLE_FACTS
            if args.topic in SAMPLE_FACTS:
                topic_facts = SAMPLE_FACTS[args.topic]
                facts = [fact for fact in all_facts if any(tf in fact['text'] for tf in topic_facts)]
                if not facts:
                    print(f"No facts found for topic '{args.topic}'")
                    return 1
            else:
                print(f"Topic '{args.topic}' not recognized. Available topics: {', '.join(SAMPLE_FACTS.keys())}")
                return 1
        else:
            # Group facts by clustering
            try:
                from fact_consolidator import FactConsolidator
                import text_similarity
                
                # Mock fact client
                class MockFactClient:
                    pass
                
                # Create LLM client
                class LLMClient:
                    def __init__(self, api_url):
                        self.client = OpenAI(base_url=api_url, api_key="not-needed")
                    
                    def consolidate_facts(self, facts):
                        facts_text = "\n".join([f"- {fact}" for fact in facts])
                        
                        prompt = f"""
                        I have a set of similar personal facts that need to be consolidated 
                        and deduplicated. Please combine these facts into one or more consolidated, 
                        accurate statements that preserve all the important information.
                        
                        Original facts:
                        {facts_text}
                        
                        Consolidated fact(s):
                        """
                        
                        response = self.client.chat.completions.create(
                            model="local-model",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant that consolidates similar facts."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.3,
                            max_tokens=300
                        )
                        
                        return response.choices[0].message.content.strip()
                
                print("\nClustering facts and testing consolidation...\n")
                
                llm_client = LLMClient(args.api_url)
                consolidator = FactConsolidator(MockFactClient(), llm_client)
                
                # Cluster the facts
                clusters = consolidator.cluster_facts(all_facts, use_agglomerative=True, similarity_threshold=0.5)
                logger.info(f"Found {len(clusters)} clusters")
                
                # Process each cluster
                for i, (cluster_id, cluster_facts) in enumerate(clusters.items()):
                    if len(cluster_facts) > 1:  # Only process clusters with multiple facts
                        print(f"\n--- Cluster {i+1} ---")
                        
                        # Show original facts
                        print("\nOriginal facts:")
                        for j, fact in enumerate(cluster_facts):
                            print(f"{j+1}. {fact['text']}")
                        
                        # Get consolidated facts
                        fact_texts = [fact["text"] for fact in cluster_facts]
                        consolidated_text = llm_client.consolidate_facts(fact_texts)
                        
                        # Show consolidated facts
                        print("\nConsolidated facts:")
                        print(consolidated_text)
            except ImportError as e:
                # Fall back to simple test if import fails
                logger.error(f"Could not import required modules: {str(e)}")
                print("Could not import FactConsolidator, falling back to simple test.")
                test_fact_consolidation(all_facts[:5], args.api_url)
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        print(f"Error: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    main() 