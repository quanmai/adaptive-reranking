import numpy as np
import torch
import time
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from models import FlanT5, GPT

class EmbeddingModel:
    """Efficient embedding model for generating dense vector representations."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into dense vectors.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text into a dense vector.
        
        Args:
            text: Text string to encode
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        return self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score between -1 and 1
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def calculate_uncertainty(doc_vector: np.ndarray, query_vector: np.ndarray) -> float:
    """
    Calculate uncertainty score for a document given the query.
    
    Args:
        doc_vector: Document embedding vector
        query_vector: Query embedding vector
        
    Returns:
        Uncertainty score (higher means more uncertain)
    """
    similarity = cosine_similarity(doc_vector, query_vector)
    return 1 - abs(similarity)

def normalize_scores(scores: List[float], target_min: float = 0.0, target_max: float = 1.0) -> List[float]:
    """
    Normalize scores to a target range.
    
    Args:
        scores: List of scores to normalize
        target_min: Target minimum value
        target_max: Target maximum value
        
    Returns:
        List of normalized scores
    """
    if not scores:
        return scores
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [target_min + (target_max - target_min) / 2] * len(scores)
    
    normalized = []
    for score in scores:
        norm_score = (score - min_score) / (max_score - min_score)
        norm_score = target_min + norm_score * (target_max - target_min)
        normalized.append(norm_score)
    
    return normalized

async def quantum_inspired_adr(
    data: List[Dict], 
    llm: str, 
    order: str = "bm25",
    batch_size: int = 5,
    llm_budget: int = 20,
    top_k: int = 10,
    learning_rate: float = 0.1,
    embedding_model_name: str = "all-MiniLM-L6-v2"
) -> Dict[str, List[str]]:
    """
    Quantum-Inspired Adaptive Document Reranking algorithm.
    
    Args:
        data: List of queries with candidate documents
        llm: LLM model name
        order: Initial ordering strategy
        batch_size: Number of documents to evaluate in each iteration
        llm_budget: Maximum number of LLM evaluations
        top_k: Number of top documents to return
        learning_rate: Learning rate for state vector updates
        embedding_model_name: Name of the embedding model
        
    Returns:
        Dictionary mapping query IDs to ranked document IDs
    """
    
    # Initialize LLM ranker
    if "gpt" in llm:
        ranker = GPT(model=llm)
    else:
        ranker = FlanT5(model=llm)
    
    # Initialize embedding model
    embedding_model = EmbeddingModel(embedding_model_name)
    
    # Statistics tracking
    total_llm_calls = 0
    total_tokens = 0
    total_time = 0
    
    results = {}
    
    for entry in tqdm(data, desc="Processing queries with QI-ADR", unit="query"):
        query = entry['query']
        docs = entry['hits']
        
        if len(docs) == 0:
            continue
            
        query_id = str(docs[0]['qid'])
        
        # Apply initial ordering
        if order == "random":
            np.random.shuffle(docs)
        elif order == "inverse":
            docs = docs[::-1]
        
        start_time = time.perf_counter()
        
        # Stage 1: Probabilistic State Representation (Initialization)
        print(f"Stage 1: Initializing state vectors for query {query_id}")
        
        # Generate query vector
        query_vector = embedding_model.encode_single(query)
        
        # Generate document vectors and initialize data structures
        doc_texts = [doc['content'] for doc in docs]
        doc_vectors = embedding_model.encode(doc_texts)
        
        # Initialize data structures
        relevance_states = {}
        candidate_pool = {}
        ranked_list = {}
        
        for i, doc in enumerate(docs):
            doc_id = doc['docid']
            relevance_states[doc_id] = doc_vectors[i].copy()
            candidate_pool[doc_id] = {
                'doc': doc,
                'vector': doc_vectors[i].copy(),
                'initial_similarity': cosine_similarity(doc_vectors[i], query_vector)
            }
        
        current_budget = llm_budget
        
        # Stage 2 & 3: Iterative Refinement Loop
        iteration = 0
        while current_budget > 0 and len(ranked_list) < top_k and len(candidate_pool) > 0:
            iteration += 1
            print(f"Iteration {iteration}: Budget remaining: {current_budget}")
            
            # Stage 2: Uncertainty Quantification
            uncertainties = {}
            for doc_id, doc_info in candidate_pool.items():
                uncertainty = calculate_uncertainty(doc_info['vector'], query_vector)
                uncertainties[doc_id] = uncertainty
            
            # Select most uncertain documents for evaluation
            sorted_uncertain = sorted(uncertainties.items(), key=lambda x: x[1], reverse=True)
            batch_to_evaluate = sorted_uncertain[:min(batch_size, current_budget, len(candidate_pool))]
            
            if not batch_to_evaluate:
                break
            
            print(f"  Evaluating {len(batch_to_evaluate)} documents with LLM")
            
            # Stage 3: Perform "Measurement" with LLM
            evaluated_docs = []
            for doc_id, uncertainty in batch_to_evaluate:
                doc_info = candidate_pool[doc_id]
                doc_content = doc_info['doc']['content']
                
                # Get LLM score for this document
                tokens, scores = ranker.generate(query, [doc_content])
                total_tokens += tokens
                total_llm_calls += 1
                
                # Convert LLM output to relevance score (0-1 scale)
                if "gpt" in llm:
                    # For GPT, the score is already a logit
                    llm_score = 1.0 / (1.0 + np.exp(-scores[0]))  # Sigmoid
                else:
                    # For FlanT5, apply temperature scaling
                    llm_score = 1.0 / (1.0 + np.exp(-scores[0] / 34))  # Temperature = 34
                
                evaluated_docs.append({
                    'doc_id': doc_id,
                    'doc_info': doc_info,
                    'llm_score': llm_score,
                    'vector': doc_info['vector']
                })
                
                # Add to ranked list
                ranked_list[doc_id] = {
                    'doc': doc_info['doc'],
                    'score': llm_score,
                    'type': 'llm'
                }
                
                # Remove from candidate pool
                del candidate_pool[doc_id]
                current_budget -= 1
            
            # Propagate Knowledge (Bayesian-style Update)
            print(f"  Propagating knowledge to {len(candidate_pool)} remaining documents")
            
            for eval_doc in evaluated_docs:
                eval_vector = eval_doc['vector']
                eval_score = eval_doc['llm_score']
                eval_initial_sim = eval_doc['doc_info']['initial_similarity']
                
                # Update state vectors of remaining documents
                for doc_id, doc_info in candidate_pool.items():
                    # Calculate similarity between evaluated and unevaluated documents
                    sim = cosine_similarity(eval_vector, doc_info['vector'])
                    
                    # Update the state vector
                    score_diff = eval_score - doc_info['initial_similarity']
                    update = learning_rate * sim * score_diff * eval_vector
                    doc_info['vector'] = doc_info['vector'] + update
                    
                    # Normalize the vector to maintain unit length
                    norm = np.linalg.norm(doc_info['vector'])
                    if norm > 0:
                        doc_info['vector'] = doc_info['vector'] / norm
        
        # Stage 4: Final Ranking Aggregation
        print(f"Stage 4: Final ranking for query {query_id}")
        
        # Generate final inferred scores for remaining documents
        for doc_id, doc_info in candidate_pool.items():
            final_score = cosine_similarity(doc_info['vector'], query_vector)
            ranked_list[doc_id] = {
                'doc': doc_info['doc'],
                'score': final_score,
                'type': 'inferred'
            }
        
        # Separate LLM and inferred scores for normalization
        llm_scores = [item['score'] for item in ranked_list.values() if item['type'] == 'llm']
        inferred_scores = [item['score'] for item in ranked_list.values() if item['type'] == 'inferred']
        
        # Normalize scores to common scale
        if llm_scores:
            normalized_llm = normalize_scores(llm_scores, 0.0, 1.0)
        else:
            normalized_llm = []
            
        if inferred_scores:
            normalized_inferred = normalize_scores(inferred_scores, 0.0, 1.0)
        else:
            normalized_inferred = []
        
        # Update scores with normalized values
        llm_idx = 0
        inferred_idx = 0
        for doc_id, item in ranked_list.items():
            if item['type'] == 'llm':
                item['normalized_score'] = normalized_llm[llm_idx]
                llm_idx += 1
            else:
                item['normalized_score'] = normalized_inferred[inferred_idx]
                inferred_idx += 1
        
        # Sort by normalized scores and get top-k
        sorted_docs = sorted(ranked_list.items(), 
                           key=lambda x: x[1]['normalized_score'], 
                           reverse=True)
        
        # Return top documents
        results[query_id] = [doc_id for doc_id, _ in sorted_docs[:top_k]]
        
        end_time = time.perf_counter()
        total_time += end_time - start_time
    
    # Print statistics
    num_queries = len(data)
    if num_queries > 0:
        print(f"\nQI-ADR Statistics:")
        print(f"Average LLM calls per query: {total_llm_calls / num_queries:.2f}")
        print(f"Average tokens per query: {total_tokens / num_queries:.2f}")
        print(f"Average latency per query: {total_time / num_queries:.3f}s")
    
    return results