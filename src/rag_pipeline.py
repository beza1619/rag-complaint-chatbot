# rag_pipeline.py - Task 3 functions
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class RAGSystem:
    def __init__(self, embeddings_path):
        """Initialize RAG system with pre-built embeddings"""
        print("Loading RAG system...")
        
        # Load embeddings
        self.chunks_df = pd.read_parquet(embeddings_path)
        self.embeddings_array = np.stack(self.chunks_df['embedding'].values)
        
        # Load embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(f"✅ System loaded: {len(self.chunks_df)} chunks available")
    
    def retrieve_chunks(self, query, k=5):
        """Retrieve top-k most relevant chunks"""
        # Embed the query
        query_embedding = self.model.encode(query)
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.embeddings_array)[0]
        
        # Get top-k indices
        top_indices = similarities.argsort()[-k:][::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            results.append({
                'text': self.chunks_df.iloc[idx]['text'],
                'product': self.chunks_df.iloc[idx]['product'],
                'similarity': float(similarities[idx]),
                'chunk_index': int(self.chunks_df.iloc[idx]['chunk_index'])
            })
        
        return results
    
    def generate_answer(self, question, chunks):
        """Generate answer based on retrieved chunks"""
        # Analyze chunks
        products = {}
        issues = []
        
        for chunk in chunks:
            product = chunk['product']
            products[product] = products.get(product, 0) + 1
            
            # Check for common issues
            text_lower = chunk['text'].lower()
            issue_keywords = [
                ('billing', ['billing', 'charge', 'fee', 'overcharge']),
                ('access', ['access', 'login', 'password', 'locked']),
                ('service', ['service', 'customer', 'representative', 'support']),
                ('fraud', ['fraud', 'unauthorized', 'theft', 'scam']),
                ('payment', ['payment', 'transaction', 'transfer', 'withdrawal']),
                ('credit', ['credit', 'score', 'report', 'rating']),
                ('loan', ['loan', 'interest', 'repayment', 'debt'])
            ]
            
            for issue_name, keywords in issue_keywords:
                if any(keyword in text_lower for keyword in keywords):
                    if issue_name not in issues:
                        issues.append(issue_name)
        
        # Build answer
        answer_parts = [f"**Analysis of customer complaints for:** '{question}'"]
        answer_parts.append("---")
        
        if products:
            answer_parts.append(f"**Found {len(chunks)} relevant complaint excerpts**")
            answer_parts.append("**Product distribution:**")
            for product, count in products.items():
                answer_parts.append(f"- {product}: {count} complaints")
        
        if issues:
            answer_parts.append(f"\n**Common issues:** {', '.join(issues)}")
        
        # Add source examples
        answer_parts.append("\n**Top complaint excerpts:**")
        for i, chunk in enumerate(chunks[:3]):  # Show top 3
            excerpt = chunk['text']
            if len(excerpt) > 150:
                excerpt = excerpt[:150] + "..."
            answer_parts.append(f"{i+1}. ({chunk['product']}, similarity: {chunk['similarity']:.2f})")
            answer_parts.append(f"   '{excerpt}'")
        
        return "\n".join(answer_parts), chunks

# Test function
def test_rag_system():
    """Test the RAG system with example questions"""
    # Initialize system
    rag = RAGSystem('../vector_store/sample_embeddings.parquet')
    
    # Test questions
    test_questions = [
        "What are common credit card issues?",
        "Are there problems with savings accounts?",
        "What billing complaints do customers have?",
        "Tell me about customer service complaints",
        "Are there any fraud-related complaints?"
    ]
    
    print("\n🧪 Testing RAG System with 5 questions...")
    print("=" * 60)
    
    results = []
    for question in test_questions:
        print(f"\nQuestion: {question}")
        
        # Retrieve chunks
        chunks = rag.retrieve_chunks(question, k=4)
        
        # Generate answer
        answer, _ = rag.generate_answer(question, chunks)
        
        # Store results for evaluation
        results.append({
            'question': question,
            'answer': answer,
            'num_chunks': len(chunks),
            'top_similarity': chunks[0]['similarity'] if chunks else 0
        })
        
        # Print summary
        print(f"Retrieved: {len(chunks)} chunks")
        print(f"Top similarity: {chunks[0]['similarity']:.3f}" if chunks else "No chunks")
    
    return results

if __name__ == "__main__":
    # Run test
    test_results = test_rag_system()
    
    # Save results to CSV for evaluation table
    results_df = pd.DataFrame(test_results)
    results_df.to_csv('../data/processed/test_results.csv', index=False)
    print(f"\n✅ Saved test results to: ../data/processed/test_results.csv")