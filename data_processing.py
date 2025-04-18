#!/usr/bin/env python3
"""
Direct analysis of TCGA data for ML model creation.
This simplified script:
1. Processes the provided gene expression data directly
2. Processes mutation data to create simple mutational pattern features
3. Combines these for ML training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import re
import os

# We'll work directly with gene expression instead of specific immune markers
# since the exact immune markers might not be available in the dataset

def load_expression_data(filename):
    """Load gene expression data from the provided file"""
    print(f"Loading expression data from {filename}...")
    
    try:
        # Determine file format and load accordingly
        if filename.endswith('.csv'):
            expr_data = pd.read_csv(filename, index_col=0)
        else:
            # Try tab-delimited format
            expr_data = pd.read_csv(filename, index_col=0, sep='\t')
        
        print(f"Loaded expression data with shape: {expr_data.shape}")
        return expr_data
    
    except Exception as e:
        print(f"Error loading expression data: {e}")
        raise

def prepare_expression_data(expr_data, n_genes=100):
    """
    Prepare gene expression data for ML analysis
    This function:
    1. Selects the top most variable genes for feature reduction
    2. Standardizes the expression values
    3. Returns a transpose of the data (samples × genes)
    """
    print("Preparing gene expression data...")
    
    # Calculate variance of each gene across samples
    gene_variance = expr_data.var(axis=1).sort_values(ascending=False)
    
    # Select top most variable genes
    top_genes = gene_variance.index[:n_genes]
    print(f"Selected {len(top_genes)} most variable genes out of {expr_data.shape[0]} total genes")
    
    # Subset the expression data
    expr_subset = expr_data.loc[top_genes]
    
    # Standardize the data
    scaler = StandardScaler()
    expr_scaled = pd.DataFrame(
        scaler.fit_transform(expr_subset.T),
        index=expr_subset.columns,
        columns=expr_subset.index
    )
    
    # Convert sample IDs to patient IDs (first 12 characters)
    expr_scaled.index = [idx[:12] if idx.startswith('TCGA') else idx for idx in expr_scaled.index]
    
    print(f"Prepared expression data with shape: {expr_scaled.shape}")
    return expr_scaled

def load_mutations(filename):
    """Load mutation data from the provided file"""
    print(f"Loading mutation data from {filename}...")
    
    try:
        # Determine file format and load
        if filename.endswith('.csv'):
            mutations = pd.read_csv(filename)
        else:
            # Try tab-delimited format
            mutations = pd.read_csv(filename, sep='\t')
        
        print(f"Loaded mutation data with {mutations.shape[0]} mutations")
        return mutations
    
    except Exception as e:
        print(f"Error loading mutation data: {e}")
        raise

def extract_mutation_patterns(mutations):
    """
    Extract simplified mutation patterns from mutation data:
    - Count mutations per patient
    - Group mutations by type (missense, silent, etc.)
    - Calculate basic trinucleotide context patterns
    
    Returns DataFrame with patients as rows and patterns as columns
    """
    print("Extracting mutation patterns...")
    
    # Extract patient IDs (first 12 characters of sample barcode)
    mutations['Patient_ID'] = mutations['Tumor_Sample_Barcode'].apply(lambda x: str(x)[:12])
    
    # Count mutations per patient
    mutation_counts = mutations.groupby('Patient_ID').size().to_frame('Total_Mutations')
    
    # Count mutation types per patient
    if 'Variant_Classification' in mutations.columns:
        type_counts = mutations.groupby(['Patient_ID', 'Variant_Classification']).size().unstack(fill_value=0)
        type_counts.columns = [f'Type_{col}' for col in type_counts.columns]
        mutation_counts = mutation_counts.join(type_counts)
    
    # Extract basic nucleotide context
    if 'Reference_Allele' in mutations.columns and 'Tumor_Seq_Allele2' in mutations.columns:
        # Create SNV type column (e.g., C>T, G>A)
        snv_mask = (mutations['Reference_Allele'].str.len() == 1) & (mutations['Tumor_Seq_Allele2'].str.len() == 1)
        snv_mutations = mutations[snv_mask].copy()
        
        if not snv_mutations.empty:
            snv_mutations['SNV_Type'] = snv_mutations['Reference_Allele'] + '>' + snv_mutations['Tumor_Seq_Allele2']
            
            # Count SNV types per patient
            snv_counts = snv_mutations.groupby(['Patient_ID', 'SNV_Type']).size().unstack(fill_value=0)
            snv_counts.columns = [f'SNV_{col}' for col in snv_counts.columns]
            mutation_counts = mutation_counts.join(snv_counts)
    
    # Normalize counts to create pattern "signatures"
    # (dividing each type by total mutations)
    pattern_cols = [col for col in mutation_counts.columns if col != 'Total_Mutations']
    
    if pattern_cols:
        for col in pattern_cols:
            mutation_counts[f'Norm_{col}'] = mutation_counts[col] / mutation_counts['Total_Mutations']
        mutation_counts.fillna(0, inplace=True)
    
    print(f"Created mutation patterns with {len(mutation_counts)} patients and {len(mutation_counts.columns)} features")
    return mutation_counts

def create_ml_dataset(expr_file, mut_file, output_file="ml_dataset.csv", n_expr_genes=100):
    """
    Create a machine learning dataset by combining:
    1. Gene expression data (top variable genes)
    2. Mutation patterns from mutation data
    """
    # Step 1: Process expression data
    expr_data = load_expression_data(expr_file)
    prepared_expr = prepare_expression_data(expr_data, n_genes=n_expr_genes)
    
    # Step 2: Process mutation data to get mutation patterns
    mutations = load_mutations(mut_file)
    mutation_patterns = extract_mutation_patterns(mutations)
    
    # Step 3: Merge datasets
    common_patients = list(set(prepared_expr.index) & set(mutation_patterns.index))
    print(f"Found {len(common_patients)} patients with both mutation and expression data")
    
    if not common_patients:
        print("Error: No common patients found between datasets")
        return None
    
    ml_data = pd.merge(
        mutation_patterns.loc[common_patients],
        prepared_expr.loc[common_patients],
        left_index=True,
        right_index=True
    )
    
    # Step 4: Save dataset
    ml_data.to_csv(output_file)
    print(f"Saved ML dataset to {output_file} with {ml_data.shape[0]} patients and {ml_data.shape[1]} features")
    
    # Print summary
    print("\nDataset Summary:")
    mutation_features = [col for col in ml_data.columns if col.startswith(('Type_', 'SNV_', 'Norm_')) or col == 'Total_Mutations']
    expression_features = [col for col in ml_data.columns if col not in mutation_features]
    
    print(f"- Mutation features: {len(mutation_features)}")
    print(f"- Gene expression features: {len(expression_features)}")
    
    # Preview
    print("\nFeature preview (first 5 columns):")
    print(ml_data.iloc[:5, :5])
    
    return ml_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create ML dataset from gene expression and mutation data')
    parser.add_argument('--expression', required=True, help='Path to gene expression data file')
    parser.add_argument('--mutations', required=True, help='Path to mutation data file')
    parser.add_argument('--output', default='ml_dataset.csv', help='Output file path')
    parser.add_argument('--n_genes', type=int, default=100, help='Number of top variable genes to include')
    
    args = parser.parse_args()
    
    create_ml_dataset(
        args.expression, 
        args.mutations, 
        args.output,
        n_expr_genes=args.n_genes
    )