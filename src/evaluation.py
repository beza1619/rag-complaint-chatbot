# evaluation.py - Task 3 evaluation functions
import pandas as pd
from rag_pipeline import RAGSystem

def create_evaluation_table():
    """Create evaluation table with test questions and answers"""
    
    # Initialize RAG system
    rag = RAGSystem('../vector_store/sample_embeddings.parquet')
    
    # Define test questions (5-10 as required)
    test_questions = [
        {
            'question': 'What are common credit card issues?',
            'expected_focus': ['credit card', 'billing', 'service', 'payment']
        },
        {
            'question': 'Are there problems with savings accounts?',
            'expected_focus': ['savings', 'checking', 'account', 'access']
        },
        {
            'question': 'What billing complaints do customers have?',
            'expected_focus': ['billing', 'charge', 'fee', 'payment']
        },
        {
            'question': 'Tell me about customer service complaints',
            'expected_focus': ['service', 'customer', 'representative', 'support']
        },
        {
            'question': 'Are there any fraud-related complaints?',
            'expected_focus': ['fraud', 'unauthorized', 'theft', 'scam']
        },
        {
            'question': 'What issues exist with money transfers?',
            'expected_focus': ['transfer', 'money', 'transaction', 'wire']
        },
        {
            'question': 'How many complaints mention late payments?',
            'expected_focus': ['late', 'payment', 'delay', 'overdue']
        },
        {
            'question': 'What problems do customers report with loans?',
            'expected_focus': ['loan', 'interest', 'debt', 'repayment']
        }
    ]
    
    print("Running evaluation on 8 test questions...")
    print("=" * 70)
    
    evaluation_results = []
    
    for i, test in enumerate(test_questions, 1):
        question = test['question']
        expected = test['expected_focus']
        
        print(f"\n{i}. Question: {question}")
        print(f"   Expected focus: {', '.join(expected)}")
        
        # Retrieve chunks
        chunks = rag.retrieve_chunks(question, k=5)
        
        # Generate answer
        answer, sources = rag.generate_answer(question, chunks)
        
        # Calculate quality score (simple heuristic)
        quality_score = 0
        
        # Check if answer mentions expected keywords
        answer_lower = answer.lower()
        keywords_found = sum(1 for keyword in expected if keyword in answer_lower)
        
        # Score based on keywords found and chunk relevance
        if keywords_found >= 3:
            quality_score += 2
        elif keywords_found >= 1:
            quality_score += 1
        
        # Score based on chunk similarity
        if chunks and chunks[0]['similarity'] > 0.5:
            quality_score += 2
        elif chunks and chunks[0]['similarity'] > 0.3:
            quality_score += 1
        
        # Ensure score is 1-5
        quality_score = max(1, min(5, quality_score))
        
        # Get top 2 sources for display
        top_sources = []
        for j, chunk in enumerate(chunks[:2]):
            text_preview = chunk['text'][:80] + "..." if len(chunk['text']) > 80 else chunk['text']
            top_sources.append(f"[{j+1}] {text_preview}")
        
        # Store results
        evaluation_results.append({
            'Question': question,
            'Generated Answer': answer[:150] + "..." if len(answer) > 150 else answer,
            'Retrieved Sources': "; ".join(top_sources),
            'Quality Score (1-5)': quality_score,
            'Comments/Analysis': f"Found {keywords_found}/4 expected keywords. Top similarity: {chunks[0]['similarity']:.3f}" if chunks else "No relevant chunks found"
        })
        
        print(f"   Quality score: {quality_score}/5")
    
    # Create DataFrame
    eval_df = pd.DataFrame(evaluation_results)
    
    # Save to CSV
    output_path = '../data/processed/evaluation_table.csv'
    eval_df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 70)
    print(f"✅ Evaluation complete! Saved to: {output_path}")
    
    # Display markdown table
    print("\n📊 EVALUATION TABLE (Markdown format for report):")
    print("| Question | Generated Answer | Retrieved Sources | Quality Score | Comments/Analysis |")
    print("|----------|-----------------|-------------------|---------------|-------------------|")
    
    for result in evaluation_results:
        # Truncate for display
        answer_display = result['Generated Answer'].replace('\n', ' ')[:80] + "..."
        sources_display = result['Retrieved Sources'].replace('\n', ' ')[:60] + "..."
        
        print(f"| {result['Question']} | {answer_display} | {sources_display} | {result['Quality Score (1-5)']} | {result['Comments/Analysis'][:50]}... |")
    
    return eval_df

def generate_markdown_table():
    """Generate markdown evaluation table for final report"""
    eval_df = pd.read_csv('../data/processed/evaluation_table.csv')
    
    markdown_lines = [
        "## Evaluation Results",
        "",
        "| Question | Generated Answer | Retrieved Sources | Quality Score (1-5) | Comments/Analysis |",
        "|----------|-----------------|-------------------|---------------------|-------------------|"
    ]
    
    for _, row in eval_df.iterrows():
        # Format for markdown
        question = str(row['Question'])
        answer = str(row['Generated Answer']).replace('\n', ' ')[:100]
        sources = str(row['Retrieved Sources']).replace('\n', ' ')[:80]
        score = str(row['Quality Score (1-5)'])
        comments = str(row['Comments/Analysis'])[:80]
        
        markdown_lines.append(f"| {question} | {answer}... | {sources}... | {score} | {comments}... |")
    
    # Save markdown
    with open('../data/processed/evaluation_markdown.md', 'w') as f:
        f.write('\n'.join(markdown_lines))
    
    print("✅ Markdown evaluation table saved for report inclusion")
    return '\n'.join(markdown_lines)

if __name__ == "__main__":
    # Run evaluation
    eval_df = create_evaluation_table()
    
    # Generate markdown for report
    markdown_table = generate_markdown_table()
    
    # Print sample
    print("\n📋 Sample from evaluation table:")
    print(eval_df[['Question', 'Quality Score (1-5)']].head(3))