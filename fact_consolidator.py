#!/usr/bin/env python3
"""
Fact Consolidator

A tool to consolidate and deduplicate facts from Bee AI using clustering
and a local LLM for consolidation.
"""

import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
import time

import numpy as np
import requests
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
import click
from tqdm import tqdm
from dotenv import load_dotenv

# Import our text similarity utilities
import text_similarity

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fact-consolidator")

# Constants
API_BASE_URL = os.getenv("BEE_API_URL", "https://api.bee.computer")
API_TOKEN = os.getenv("BEE_API_TOKEN", "")
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:1234/v1")


class FactClient:
    """Client for interacting with the Bee AI Facts API."""
    
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.headers = {
            "x-api-key": f"{token}",
            "Content-Type": "application/json"
        }
    
    def get_facts(self, confirmed: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Get all facts for the current user."""
        url = f"{self.base_url}/v1/me/facts"
        params = {}
        if confirmed is not None:
            params["confirmed"] = str(confirmed).lower()
        
        all_facts = []
        page = 1
        limit = 100
        
        while True:
            params.update({"page": page, "limit": limit})
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            response_data = response.json()
            facts = response_data["facts"]
            total_pages = response_data.get("totalPages", 1)
            logger.debug(f"Received page {page}/{total_pages} with {len(facts)} facts")
            
            if not facts:
                break
                
            all_facts.extend(facts)
            
            # If we have pagination info, use it to determine if we should continue
            if isinstance(response_data, dict) and "currentPage" in response_data and "totalPages" in response_data:
                current_page = response_data["currentPage"]
                total_pages = response_data["totalPages"]
                
                if current_page >= total_pages:
                    break
                page += 1
            else:
                # If we don't have pagination info, just check if we got fewer than requested
                if len(facts) < limit:
                    break
                page += 1
        
        logger.debug(f"Total facts retrieved: {len(all_facts)}")
        return all_facts
    
    def create_fact(self, text: str) -> Dict[str, Any]:
        """Create a new fact."""
        url = f"{self.base_url}/v1/me/facts"
        payload = {"text": text}
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()
    
    def delete_fact(self, fact_id: int) -> Dict[str, Any]:
        """Delete a fact by ID."""
        url = f"{self.base_url}/v1/me/facts/{fact_id}"
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()
        return response.json()


class LLMClient:
    """Client for interacting with the local LLM API."""
    
    def __init__(self, api_url: str):
        self.client = OpenAI(base_url=api_url, api_key="not-needed")
    
    def consolidate_facts(self, facts: List[str]) -> List[str]:
        """
        Use the LLM to consolidate a group of similar facts into
        one or more consolidated facts.
        """
        # Create an array of the facts
        facts_text = "[\n\t" + ",\n\t".join([f'"{fact}"' for fact in facts]) + "\n]"
        
        prompt = f"""
        I have a set of similar personal facts that need to be consolidated 
        and deduplicated. Please combine these facts into one or more consolidated, 
        accurate statements that preserve all the important information.
        
        Please adhere to the following rules:
        1. Do not make up new facts, only consolidate existing ones.
        2. Facts should be relatively short, no more than one sentence.
        3. All output facts must begin with the name of the person they are about.
        4. Do not merge facts with distinct implications or subjects.
        5. If possible, do not remove information when consolidating facts.
        6. Respond ONLY with a valid array containing consolidated / de-duplicated facts.

        Below is an example of how you should consolidate facts:
        
        # Example Input:
        [
            "Ian is friends with someone named Scott",
            "Ian has a friend named Scott",
            "Ian has a friend named Scott",
            "Scott is friends with Ian"
        ]

        # Example Output:
        [
            "Ian is close friends with someone named Scott",
            "Ian has a friend named Scott that is the best man at his wedding"
        ]

    
        
        # Input:
        {facts_text}
        
        # Output:
        """
        
        try:
            response = self.client.chat.completions.create(
                model="local-model",  # This is ignored by the local LLM server
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that consolidates similar facts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )

            output = response.choices[0].message.content
            # Find the last array in the response
            logger.info(f"\nInput:\n{facts_text}\nOutput:\n{output}")
            return json.loads(output)
        
        except Exception as e:
            logger.error(f"Error calling LLM API: {str(e)}")
            return []


class FactConsolidator:
    """Main class for consolidating facts using clustering and LLM."""
    
    def __init__(self, fact_client: FactClient, llm_client: LLMClient):
        self.fact_client = fact_client
        self.llm_client = llm_client
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def cluster_facts_kmeans(self, facts: List[Dict[str, Any]], n_clusters: int = 0) -> Dict[int, List[Dict[str, Any]]]:
        """
        Cluster facts based on their text similarity using k-means.
        If n_clusters is 0, automatically determine the number of clusters.
        """
        if len(facts) <= 1:
            return {0: facts}
        
        # Extract fact texts
        fact_texts = [fact["text"] for fact in facts]
        
        # Create TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform(fact_texts)
        
        # Determine number of clusters if not specified
        if n_clusters <= 0:
            # Heuristic: sqrt of number of data points, capped between 2 and 10
            n_clusters = min(10, max(2, int(np.sqrt(len(facts)))))
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Group facts by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(facts[i])
        
        return clusters
    
    def cluster_facts_agglomerative(self, facts: List[Dict[str, Any]], similarity_threshold: float = 0.5) -> Dict[int, List[Dict[str, Any]]]:
        """
        Cluster facts based on their text similarity using agglomerative clustering.
        """
        if len(facts) <= 1:
            return {0: facts}
        
        # Extract fact texts
        fact_texts = [fact["text"] for fact in facts]
        
        # Create TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform(fact_texts)
        
        # Apply agglomerative clustering
        return text_similarity.agglomerative_clustering(facts, tfidf_matrix, similarity_threshold)
    
    def cluster_facts(self, facts: List[Dict[str, Any]], 
                     n_clusters: int = 0, 
                     similarity_threshold: float = 0.5,
                     use_agglomerative: bool = True) -> Dict[int, List[Dict[str, Any]]]:
        """
        Cluster facts based on their text similarity.
        Chooses between k-means and agglomerative clustering based on parameters.
        """
        if use_agglomerative:
            return self.cluster_facts_agglomerative(facts, similarity_threshold)
        else:
            return self.cluster_facts_kmeans(facts, n_clusters)
    
    def process_facts(self, facts: List[Dict[str, Any]], 
                     min_cluster_size: int = 2,
                     similarity_threshold: float = 0.5,
                     use_agglomerative: bool = True) -> List[Tuple[List[Dict[str, Any]], List[str]]]:
        """
        Process facts:
        1. Cluster similar facts
        2. For each cluster with at least min_cluster_size facts, get consolidated facts
        
        Returns a list of tuples: (original_facts, consolidated_facts)
        """
        results = []
        
        # Validate input
        if not facts:
            logger.warning("No facts provided to process")
            return results
            
        # Check if facts have the expected structure
        if not all(isinstance(fact, dict) and "text" in fact for fact in facts):
            # Log detailed info about the first few facts to help diagnose issues
            sample_facts = facts[:3]
            logger.error(f"Facts don't have the expected structure. Sample: {sample_facts}")
            raise ValueError("Facts must be dictionaries with a 'text' field")
        
        try:
            # Cluster the facts
            logger.info(f"Clustering {len(facts)} facts using {'agglomerative' if use_agglomerative else 'k-means'} clustering")
            clusters = self.cluster_facts(facts, 
                                        similarity_threshold=similarity_threshold,
                                        use_agglomerative=use_agglomerative)
            
            logger.info(f"Found {len(clusters)} clusters")
            
            # Process each cluster
            for cluster_id, cluster_facts in clusters.items():
                logger.debug(f"Cluster {cluster_id}: {len(cluster_facts)} facts")
                
                if len(cluster_facts) >= min_cluster_size:
                    # Get the text of the facts in this cluster
                    fact_texts = [fact["text"] for fact in cluster_facts]
                    
                    # Get consolidated facts
                    logger.debug(f"Sending {len(fact_texts)} facts to LLM for consolidation")
                    consolidated_facts = self.llm_client.consolidate_facts(fact_texts)
                    
                    if consolidated_facts:
                        logger.debug(f"Received {len(consolidated_facts)} consolidated facts")
                        results.append((cluster_facts, consolidated_facts))
                    else:
                        logger.warning(f"No consolidated facts returned for cluster {cluster_id}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing facts: {str(e)}")
            raise


@click.command()
@click.option("--min-cluster-size", default=2, help="Minimum number of facts in a cluster to consider for consolidation")
@click.option("--confirmed-only", is_flag=True, help="Only process confirmed facts")
@click.option("--auto-approve", is_flag=True, help="Automatically approve all consolidations without prompting")
@click.option("--dry-run", is_flag=True, help="Don't make any changes, just show what would be done")
@click.option("--similarity-threshold", default=0.5, help="Similarity threshold for clustering (0.0-1.0)")
@click.option("--use-kmeans", is_flag=True, help="Use k-means instead of agglomerative clustering")
@click.option("--debug", is_flag=True, help="Enable debug logging")
def main(min_cluster_size: int, confirmed_only: bool, auto_approve: bool, dry_run: bool, 
         similarity_threshold: float, use_kmeans: bool, debug: bool):
    """Consolidate similar facts from Bee AI using clustering and LLM assistance."""
    
    # Configure logging level
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    if not API_TOKEN:
        logger.error("BEE_API_TOKEN environment variable is not set")
        return
    
    try:
        # Initialize clients
        fact_client = FactClient(API_BASE_URL, API_TOKEN)
        llm_client = LLMClient(LLM_API_URL)
        consolidator = FactConsolidator(fact_client, llm_client)
        
        # Get facts
        confirmed_param = True if confirmed_only else None
        logger.info(f"Fetching facts (confirmed_only={confirmed_only})...")
        facts = fact_client.get_facts(confirmed=confirmed_param)
        logger.info(f"Retrieved {len(facts)} facts")
        
        if not facts:
            logger.info("No facts to process")
            return
        
        # Process facts
        logger.info(f"Clustering facts with minimum cluster size {min_cluster_size}...")
        consolidation_results = consolidator.process_facts(
            facts, 
            min_cluster_size=min_cluster_size,
            similarity_threshold=similarity_threshold,
            use_agglomerative=not use_kmeans
        )
        logger.info(f"Found {len(consolidation_results)} clusters to consolidate")
        
        if not consolidation_results:
            logger.info("No consolidation candidates found")
            return
        
        # Process each consolidation candidate
        for i, (original_facts, consolidated_facts) in enumerate(consolidation_results):
            print(f"\n--- Consolidation {i+1}/{len(consolidation_results)} ---")
            
            # Show original facts
            print("\nOriginal facts:")
            for j, fact in enumerate(original_facts):
                print(f"{j+1}. {fact['text']}")
            
            # Show consolidated facts
            print("\nConsolidated facts:")
            for j, fact in enumerate(consolidated_facts):
                print(f"{j+1}. {fact}")
            
            # Get user approval
            if not auto_approve and not dry_run:
                while True:
                    response = input("\nApprove this consolidation? [y/n]: ").strip().lower()
                    if response in ['y', 'n']:
                        break
                    print("Please enter 'y' or 'n'")
                
                approved = response == 'y'
            else:
                approved = auto_approve
            
            # Process approval
            if approved and not dry_run:
                print("Approved. Updating facts...")
                
                try:
                    # Delete original facts
                    for fact in original_facts:
                        print(f"Deleting fact {fact['id']}: {fact['text']}")
                        fact_client.delete_fact(fact['id'])
                        # Small delay to avoid overwhelming the API
                        time.sleep(0.5)
                    
                    # Create new consolidated facts
                    for fact_text in consolidated_facts:
                        print(f"Creating new fact: {fact_text}")
                        fact_client.create_fact(fact_text)
                        # Small delay to avoid overwhelming the API
                        time.sleep(0.5)
                except Exception as e:
                    logger.error(f"Error updating facts: {str(e)}")
                    print(f"Error: {str(e)}")
                    if debug:
                        import traceback
                        traceback.print_exc()
            elif dry_run:
                print("Dry run - no changes made")
            else:
                print("Skipped")
        
        print("\nFact consolidation complete!")
    
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"Error: {str(e)}")
        if debug:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    main() 