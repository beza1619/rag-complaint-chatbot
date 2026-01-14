# chunking_embedding.py - Task 2 functions
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer

def load_filtered_data(file_path):
    """Load the filtered complaints data"""
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} filtered complaints")
    return df

def create_stratified_sample(df, sample_size=500, random_state=42):
    """Create stratified sample ensuring all products represented"""
    # Get product counts
    product_counts = df['Product'].value_counts()
    
    # Calculate sample per product (proportional)
    samples_per_product = {}
    for product, count in product_counts.items():
        proportion = count / len(df)
        samples_per_product[product] = max(1, int(sample_size * proportion))
    
    # Adjust if total doesn't match sample_size
    total_samples = sum(samples_per_product.values())
    if total_samples != sample_size:
        # Adjust the largest product
        largest_product = max(samples_per_product, key=samples_per_product.get)
        samples_per_product[largest_product] += (sample_size - total_samples)
    
    # Create sample
    sampled_dfs = []
    for product, n_samples in samples_per_product.items():
        product_df = df[df['Product'] == product]
        if len(product_df) >= n_samples:
            sampled_dfs.append(product_df.sample(n=n_samples, random_state=random_state))
        else:
            sampled_dfs.append(product_df)
    
    sample_df = pd.concat(sampled_dfs, ignore_index=True)
    print(f"Created stratified sample of {len(sample_df)} complaints")
    return sample_df

def chunk_text(text, chunk_size=500, overlap=50):
    """Simple text chunking function"""
    if not text or len(text) < chunk_size:
        return [text] if text else []
    
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(text):
            break
    
    return chunks

def create_embeddings(sample_df, output_path):
    """Create embeddings for text chunks"""
    print("Creating embeddings...")
    
    # Initialize model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create chunks
    all_chunks = []
    metadata = []
    
    for idx, row in sample_df.iterrows():
        text = str(row['Consumer complaint narrative']).strip()
        if not text or len(text) < 20:
            continue
        
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata.append({
                "complaint_id": idx,
                "product": row['Product'],
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks)
            })
    
    print(f"Created {len(all_chunks)} chunks from {len(sample_df)} complaints")
    
    # Create embeddings in batches
    batch_size = 100
    embeddings = []
    
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Save to file
    chunks_df = pd.DataFrame({
        'text': all_chunks,
        'embedding': list(embeddings),
        'complaint_id': [m['complaint_id'] for m in metadata],
        'product': [m['product'] for m in metadata],
        'chunk_index': [m['chunk_index'] for m in metadata],
        'total_chunks': [m['total_chunks'] for m in metadata]
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    chunks_df.to_parquet(output_path)
    print(f"Saved embeddings to: {output_path}")
    print(f"File size: {os.path.getsize(output_path)/(1024*1024):.2f} MB")
    
    return chunks_df

if __name__ == "__main__":
    # Example usage
    df = load_filtered_data('../data/processed/filtered_complaints.csv')
    sample_df = create_stratified_sample(df, sample_size=500)
    chunks_df = create_embeddings(sample_df, '../vector_store/sample_embeddings.parquet')