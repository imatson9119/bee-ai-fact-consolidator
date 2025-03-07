"""
Utility functions for text similarity and clustering enhancement.
"""

from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Any


def compute_similarity_matrix(tfidf_matrix):
    """
    Compute the cosine similarity matrix from TF-IDF vectors.
    
    Args:
        tfidf_matrix: The TF-IDF matrix from sklearn
        
    Returns:
        A similarity matrix with values between 0 and 1
    """
    return cosine_similarity(tfidf_matrix)


def find_similar_facts(facts: List[Dict[str, Any]], 
                       tfidf_matrix, 
                       similarity_threshold: float = 0.5) -> List[Tuple[int, int, float]]:
    """
    Find pairs of facts that exceed a similarity threshold.
    
    Args:
        facts: List of fact dictionaries
        tfidf_matrix: TF-IDF matrix from sklearn
        similarity_threshold: Minimum similarity to consider (between 0 and 1)
        
    Returns:
        List of tuples containing (fact1_index, fact2_index, similarity_score)
    """
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(tfidf_matrix)
    
    # Find pairs above threshold
    similar_pairs = []
    for i in range(len(facts)):
        for j in range(i+1, len(facts)):
            similarity = similarity_matrix[i, j]
            if similarity >= similarity_threshold:
                similar_pairs.append((i, j, similarity))
    
    # Sort by similarity (highest first)
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    
    return similar_pairs


def agglomerative_clustering(facts: List[Dict[str, Any]], 
                             tfidf_matrix,
                             similarity_threshold: float = 0.4) -> Dict[int, List[Dict[str, Any]]]:
    """
    Perform agglomerative clustering on facts based on similarity.
    This is an alternative to k-means that doesn't require specifying the number of clusters.
    
    Args:
        facts: List of fact dictionaries
        tfidf_matrix: TF-IDF matrix from sklearn
        similarity_threshold: Minimum similarity to consider (between 0 and 1)
        
    Returns:
        Dictionary mapping cluster IDs to lists of facts
    """
    if len(facts) <= 1:
        return {0: facts}
    
    # Get similarity pairs
    similar_pairs = find_similar_facts(facts, tfidf_matrix, similarity_threshold)
    
    # Initialize clusters (one fact per cluster initially)
    clusters = {i: [fact] for i, fact in enumerate(facts)}
    fact_to_cluster = {i: i for i in range(len(facts))}
    
    # Merge clusters based on similarity
    for fact1_idx, fact2_idx, similarity in similar_pairs:
        # Get current clusters
        cluster1 = fact_to_cluster[fact1_idx]
        cluster2 = fact_to_cluster[fact2_idx]
        
        # Skip if already in the same cluster
        if cluster1 == cluster2:
            continue
        
        # Merge cluster2 into cluster1
        clusters[cluster1].extend(clusters[cluster2])
        
        # Update fact_to_cluster mapping
        for fact_idx in range(len(facts)):
            if fact_to_cluster[fact_idx] == cluster2:
                fact_to_cluster[fact_idx] = cluster1
        
        # Remove empty cluster
        del clusters[cluster2]
    
    # Renumber clusters for cleaner output
    result = {}
    for i, (cluster_id, cluster_facts) in enumerate(clusters.items()):
        result[i] = cluster_facts
    
    return result 