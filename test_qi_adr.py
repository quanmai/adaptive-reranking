#!/usr/bin/env python3
"""
Simple test script for the Quantum-Inspired ADR implementation.
"""

import json
import asyncio
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from quantum_inspired_adr import quantum_inspired_adr, EmbeddingModel, cosine_similarity

def create_test_data():
    """Create simple test data for validation."""
    return [
        {
            'query': 'machine learning algorithms',
            'hits': [
                {'qid': 1, 'docid': 'doc1', 'content': 'Neural networks are a type of machine learning algorithm'},
                {'qid': 1, 'docid': 'doc2', 'content': 'Linear regression is a simple machine learning technique'},
                {'qid': 1, 'docid': 'doc3', 'content': 'Decision trees are interpretable machine learning models'},
                {'qid': 1, 'docid': 'doc4', 'content': 'Random forests combine multiple decision trees'},
                {'qid': 1, 'docid': 'doc5', 'content': 'Support vector machines find optimal decision boundaries'},
                {'qid': 1, 'docid': 'doc6', 'content': 'The weather is sunny today with clear skies'},
                {'qid': 1, 'docid': 'doc7', 'content': 'Cooking pasta requires boiling water and salt'},
                {'qid': 1, 'docid': 'doc8', 'content': 'Basketball is played with two teams of five players'},
                {'qid': 1, 'docid': 'doc9', 'content': 'Deep learning uses neural networks with many layers'},
                {'qid': 1, 'docid': 'doc10', 'content': 'Gradient descent optimizes machine learning model parameters'}
            ]
        }
    ]

def test_embedding_model():
    """Test the embedding model functionality."""
    print("Testing EmbeddingModel...")
    
    try:
        model = EmbeddingModel("all-MiniLM-L6-v2")
        
        # Test single encoding
        text = "This is a test sentence"
        vector = model.encode_single(text)
        print(f"✓ Single encoding works. Vector shape: {vector.shape}")
        
        # Test batch encoding
        texts = ["First sentence", "Second sentence", "Third sentence"]
        vectors = model.encode(texts)
        print(f"✓ Batch encoding works. Vectors shape: {vectors.shape}")
        
        # Test cosine similarity
        sim = cosine_similarity(vectors[0], vectors[1])
        print(f"✓ Cosine similarity works. Similarity: {sim:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ EmbeddingModel test failed: {e}")
        return False

async def test_qi_adr():
    """Test the full QI-ADR algorithm with mock data."""
    print("\nTesting QI-ADR algorithm...")
    
    try:
        test_data = create_test_data()
        
        # Use a small configuration for testing
        result = await quantum_inspired_adr(
            data=test_data,
            llm="google/flan-t5-large",  # You might want to mock this
            order="bm25",
            batch_size=2,
            llm_budget=4,  # Small budget for testing
            top_k=5,
            learning_rate=0.1,
            embedding_model_name="all-MiniLM-L6-v2"
        )
        
        print(f"✓ QI-ADR completed successfully")
        print(f"✓ Results: {result}")
        
        # Verify result structure
        if '1' in result and len(result['1']) <= 5:
            print("✓ Result structure is correct")
            return True
        else:
            print("✗ Result structure is incorrect")
            return False
            
    except Exception as e:
        print(f"✗ QI-ADR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Running QI-ADR Implementation Tests")
    print("=" * 40)
    
    # Test 1: Embedding model
    embedding_test_passed = test_embedding_model()
    
    # Test 2: Full algorithm (if embedding test passed)
    if embedding_test_passed:
        try:
            qi_adr_test_passed = asyncio.run(test_qi_adr())
        except Exception as e:
            print(f"✗ Could not run QI-ADR test: {e}")
            qi_adr_test_passed = False
    else:
        qi_adr_test_passed = False
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Summary:")
    print(f"Embedding Model: {'✓ PASS' if embedding_test_passed else '✗ FAIL'}")
    print(f"QI-ADR Algorithm: {'✓ PASS' if qi_adr_test_passed else '✗ FAIL'}")
    
    if embedding_test_passed and qi_adr_test_passed:
        print("\n🎉 All tests passed! QI-ADR implementation is ready.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())