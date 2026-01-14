# data_processing.py - Task 1 functions
import pandas as pd
import os

def load_and_filter_data(file_path):
    """Load CFPB data and filter for target products"""
    print("Loading data from:", file_path)
    
    df = pd.read_csv(file_path)
    print(f"Original data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Filter for 4 target products
    target_products = [
        'credit card',
        'personal loan', 
        'savings account',
        'money transfer'
    ]
    
    # Create filter mask (case-insensitive)
    mask = pd.Series(False, index=df.index)
    for product in target_products:
        mask |= df['Product'].str.contains(product, case=False, na=False)
    
    filtered_df = df[mask].copy()
    print(f"After filtering: {len(filtered_df)} rows")
    
    # Remove empty narratives
    initial_count = len(filtered_df)
    filtered_df = filtered_df[filtered_df['Consumer complaint narrative'].notna()].copy()
    print(f"After removing empty narratives: {len(filtered_df)} rows")
    print(f"Removed {initial_count - len(filtered_df)} empty narratives")
    
    return filtered_df

def save_filtered_data(df, output_path):
    """Save filtered data to CSV"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved filtered data to: {output_path}")
    print(f"File size: {os.path.getsize(output_path)/(1024*1024):.2f} MB")
    
if __name__ == "__main__":
    # Example usage
    filtered_df = load_and_filter_data('../data/raw/complaints.csv')
    save_filtered_data(filtered_df, '../data/processed/filtered_complaints.csv')