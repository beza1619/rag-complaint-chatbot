# app.py - Gradio interface for CrediTrust Complaint Analyzer
import gradio as gr
import sys
import os

# Add src folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from rag_pipeline import RAGSystem
    print("✅ RAG system imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have the required packages installed")
    raise

# Initialize RAG system
def initialize_system():
    """Initialize the RAG system"""
    try:
        embeddings_path = 'vector_store/sample_embeddings.parquet'
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        
        rag = RAGSystem(embeddings_path)
        print("✅ RAG system initialized")
        return rag
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        raise

# Initialize
rag_system = initialize_system()

def analyze_complaint(question):
    """Main function to analyze complaints"""
    try:
        # Validate input
        if not question or len(question.strip()) < 3:
            return "Please enter a valid question (at least 3 characters).", []
        
        print(f"Processing question: {question}")
        
        # Retrieve relevant chunks
        chunks = rag_system.retrieve_chunks(question, k=5)
        
        if not chunks:
            return "No relevant complaints found for your question. Try a different question.", []
        
        # Generate answer
        answer, sources = rag_system.generate_answer(question, chunks)
        
        # Format sources for display
        formatted_sources = []
        for i, chunk in enumerate(chunks[:3]):  # Show top 3 sources
            formatted_sources.append({
                "rank": i + 1,
                "product": chunk['product'],
                "similarity": f"{chunk['similarity']:.3f}",
                "text": chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
            })
        
        return answer, formatted_sources
    
    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        print(error_msg)
        return error_msg, []

# Create Gradio interface
with gr.Blocks(title="CrediTrust Complaint Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏦 CrediTrust Financial Complaint Analyzer")
    gr.Markdown("### AI-Powered Analysis of Customer Complaints")
    gr.Markdown("Ask questions about complaints across Credit Cards, Personal Loans, Savings Accounts, and Money Transfers.")
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📝 Ask Your Question")
            question_input = gr.Textbox(
                label="",
                placeholder="Example: 'What are common credit card issues?' or 'Tell me about billing complaints'",
                lines=3,
                max_lines=3
            )
            
            with gr.Row():
                submit_btn = gr.Button("🔍 Analyze Complaints", variant="primary", size="lg")
                clear_btn = gr.Button("🔄 Clear", size="lg")
        
        with gr.Column(scale=3):
            gr.Markdown("### 📊 Analysis Results")
            answer_output = gr.Markdown(
                label="",
                value="*Your analysis will appear here...*"
            )
            
            with gr.Accordion("📄 View Source Complaints (Click to Expand)", open=False):
                sources_output = gr.JSON(
                    label="Retrieved Complaint Excerpts",
                    value=[]
                )
    
    # Example questions
    gr.Markdown("### 💡 Try These Example Questions:")
    
    examples = gr.Examples(
        examples=[
            ["What are common credit card issues?"],
            ["Are there problems with savings accounts?"],
            ["What billing complaints do customers have?"],
            ["Tell me about customer service complaints"],
            ["Are there any fraud-related complaints?"]
        ],
        inputs=question_input,
        label="Click any example to try:"
    )
    
    # Footer
    gr.Markdown("---")
    gr.Markdown(
        """
        **How it works:**
        1. Enter your question about customer complaints
        2. System finds similar complaint excerpts using semantic search
        3. Analyzes patterns across products and issues
        4. Provides evidence-based answer with source citations
        
        *Built for CrediTrust Financial's AI Innovation Challenge*
        """
    )
    
    # Button actions
    def process_and_display(question):
        answer, sources = analyze_complaint(question)
        return answer, sources
    
    submit_btn.click(
        fn=process_and_display,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )
    
    def clear_all():
        return "", "*Your analysis will appear here...*", []
    
    clear_btn.click(
        fn=clear_all,
        outputs=[question_input, answer_output, sources_output]
    )

# Launch application
if __name__ == "__main__":
    print("🚀 Starting CrediTrust Complaint Analyzer...")
    print("🌐 Opening web interface at http://localhost:7860")
    print("⏳ Please wait a moment for the interface to load...")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )