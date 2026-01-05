import gradio as gr
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading complaint analysis system...")

# Load pre-processed complaint embeddings
try:
    sample_embeddings = pd.read_parquet('vector_store/sample_embeddings.parquet')
    embeddings_array = np.stack(sample_embeddings['embedding'].values)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ System loaded successfully")
except Exception as e:
    print(f"❌ Error loading: {e}")
    raise

def search_complaints(query, k=5):
    """Search for similar complaint chunks"""
    query_embedding = model.encode(query)
    similarities = cosine_similarity([query_embedding], embeddings_array)[0]
    top_indices = similarities.argsort()[-k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'text': sample_embeddings.iloc[idx]['text'],
            'product': sample_embeddings.iloc[idx]['product'],
            'similarity': float(similarities[idx]),
            'chunk_index': int(sample_embeddings.iloc[idx]['chunk_index'])
        })
    return results

def analyze_complaints(question):
    """Analyze complaints and generate answer"""
    # Search for relevant complaints
    chunks = search_complaints(question, k=4)
    
    # Analyze patterns
    products = {}
    issues = []
    
    for chunk in chunks:
        product = chunk['product']
        products[product] = products.get(product, 0) + 1
        
        text_lower = chunk['text'].lower()
        issue_categories = [
            ('billing', ['billing', 'charge', 'fee', 'overcharge']),
            ('access', ['access', 'login', 'password', 'locked']),
            ('service', ['service', 'customer', 'representative', 'support']),
            ('fraud', ['fraud', 'unauthorized', 'theft', 'scam']),
            ('payment', ['payment', 'transaction', 'transfer', 'withdrawal'])
        ]
        
        for issue_name, keywords in issue_categories:
            if any(keyword in text_lower for keyword in keywords):
                if issue_name not in issues:
                    issues.append(issue_name)
    
    # Build answer
    answer_parts = [f"## Analysis of customer complaints for: '{question}'"]
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
    for i, chunk in enumerate(chunks[:3]):
        excerpt = chunk['text']
        if len(excerpt) > 150:
            excerpt = excerpt[:150] + "..."
        answer_parts.append(f"{i+1}. ({chunk['product']}, similarity: {chunk['similarity']:.2f})")
        answer_parts.append(f"   '{excerpt}'")
    
    return "\n".join(answer_parts), chunks

# Create Gradio interface
with gr.Blocks(title="CrediTrust Complaint Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏦 CrediTrust Financial Complaint Analyzer")
    gr.Markdown("### AI-Powered Complaint Analysis for Financial Services")
    gr.Markdown("Ask natural language questions about customer complaints across credit cards, loans, savings accounts, and money transfers.")
    
    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Ask a question about complaints",
                placeholder="e.g., What are common credit card issues?",
                lines=2
            )
            submit_btn = gr.Button("🔍 Analyze Complaints", variant="primary")
            clear_btn = gr.Button("🔄 Clear")
        
        with gr.Column(scale=3):
            answer_output = gr.Markdown(label="Analysis Results")
            
            with gr.Accordion("📄 View Source Complaints", open=False):
                sources_output = gr.JSON(label="Retrieved Complaint Excerpts")
    
    def process_question(question):
        answer, sources = analyze_complaints(question)
        return answer, sources
    
    submit_btn.click(
        fn=process_question,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )
    
    def clear_all():
        return "", None, None
    
    clear_btn.click(
        fn=clear_all,
        outputs=[question_input, answer_output, sources_output]
    )
    
    gr.Markdown("### 💡 Try these example questions:")
    examples = gr.Examples(
        examples=[
            ["What credit card problems do customers report?"],
            ["Are there issues with savings accounts?"],
            ["What billing complaints exist?"],
            ["Find complaints about customer service"]
        ],
        inputs=question_input
    )

if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)
