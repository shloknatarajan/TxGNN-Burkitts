"""
Function to find disease index by name in TxGNN-Burkitts
"""

import os
import pandas as pd
import numpy as np
from txgnn import TxData

def find_disease_index_by_name(disease_name, use_disease_files=True):
    """Find the disease index for a given disease name in TxGNN.
    
    Args:
        disease_name (str): Name of the disease to look up
        use_disease_files (bool): Whether to use the disease_files CSVs for lookup first
        
    Returns:
        float: The disease index if found, None otherwise
        str: The disease ID if found, None otherwise
        str: The exact disease name as found in the database
    """
    # Normalize disease name for comparison
    normalized_name = disease_name.lower().replace("'s", "").replace("-", " ")
    
    # First, try looking in the disease_files directory
    if use_disease_files:
        disease_files_dir = os.path.join('./data/disease_files')
        if os.path.exists(disease_files_dir):
            for file_name in os.listdir(disease_files_dir):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(disease_files_dir, file_name)
                    df = pd.read_csv(file_path)
                    
                    # Check if the file has the required columns
                    if 'node_name' in df.columns and 'node_id' in df.columns:
                        index_col = next((col for col in df.columns if col == 'node_index' or col == ''), None)
                        
                        # Search for disease by name
                        for idx, row in df.iterrows():
                            db_name = str(row['node_name']).lower()
                            
                            # Check for exact match or substring
                            if db_name == normalized_name or normalized_name in db_name or \
                               any(term in db_name for term in normalized_name.split()):
                                if index_col:
                                    return float(row[index_col] if index_col != '' else idx), str(row['node_id']), row['node_name']
                                else:
                                    # If no index column is present, use the row index
                                    return float(idx), str(row['node_id']), row['node_name']
    
    # If not found in disease files, try using TxData
    try:
        # Initialize TxData which will load the knowledge graph
        data = TxData('./data')
        
        # Prepare the data split (required before accessing mappings)
        data.prepare_split(split='random', seed=42)
        
        # Get the ID mappings
        mappings = data.retrieve_id_mapping()
        id2name_disease = mappings['id2name_disease']
        idx2id_disease = mappings['idx2id_disease']
        
        # Find the disease ID by matching the name
        disease_id = None
        exact_name = None
        
        for id_, name in id2name_disease.items():
            db_name = str(name).lower()
            
            # Check for exact match or substring
            if db_name == normalized_name or normalized_name in db_name or \
               any(term in db_name for term in normalized_name.split()):
                disease_id = id_
                exact_name = name
                break
                
        if disease_id is None:
            print(f"Disease '{disease_name}' not found in the knowledge graph")
            return None, None, None
            
        # Find the disease's index
        for idx, id_ in idx2id_disease.items():
            if id_ == disease_id:
                return float(idx), disease_id, exact_name
                
        print(f"Could not find index for disease '{disease_name}' (ID: {disease_id})")
        return None, disease_id, exact_name
        
    except Exception as e:
        print(f"Error finding disease index: {e}")
        return None, None, None

def find_burkitts_lymphoma_index():
    """
    Special case function for directly finding the Burkitt's Lymphoma index.
    Based on direct investigation, Burkitt lymphoma has ID 7243.0 and should be 
    in cell_proliferation.csv
    
    Returns:
        float: The disease index if found, None otherwise
        str: The disease ID if found, None otherwise
        str: The exact disease name as found in the database
    """
    # Known Burkitt lymphoma ID
    burkitt_id = 7243.0
    
    # First check specifically in cell_proliferation.csv where we know it exists
    cell_prolif_path = './data/disease_files/cell_proliferation.csv'
    if os.path.exists(cell_prolif_path):
        df = pd.read_csv(cell_prolif_path)
        if 'node_id' in df.columns:
            match = df[df['node_id'] == burkitt_id]
            if not match.empty:
                index_col = next((col for col in df.columns if col == 'node_index' or col == ''), None)
                if index_col:
                    index_val = match[index_col].values[0] if index_col != '' else match.index[0]
                else:
                    index_val = match.index[0]
                return float(index_val), str(burkitt_id), match['node_name'].values[0]
    
    # If not found in cell_proliferation.csv, check all other files
    for file_name in os.listdir('./data/disease_files'):
        if file_name.endswith('.csv'):
            file_path = os.path.join('./data/disease_files', file_name)
            df = pd.read_csv(file_path)
            if 'node_id' in df.columns:
                match = df[df['node_id'] == burkitt_id]
                if not match.empty:
                    index_col = next((col for col in df.columns if col == 'node_index' or col == ''), None)
                    if index_col:
                        index_val = match[index_col].values[0] if index_col != '' else match.index[0]
                    else:
                        index_val = match.index[0]
                    return float(index_val), str(burkitt_id), match['node_name'].values[0]
    
    # If still not found, fall back to the general function
    return find_disease_index_by_name("Burkitt lymphoma")

if __name__ == "__main__":
    # Test with Burkitt's Lymphoma
    disease_name = "Burkitt's Lymphoma"
    
    print(f"Searching for: {disease_name}")
    print("Method 1: Direct lookup of Burkitt's Lymphoma by ID")
    disease_idx, disease_id, exact_name = find_burkitts_lymphoma_index()
    
    if disease_idx is not None:
        print(f"Found disease: {exact_name}")
        print(f"ID: {disease_id}")
        print(f"Index: {disease_idx}")
    else:
        print(f"Could not find Burkitt's Lymphoma by direct ID lookup")
    
    print("\nMethod 2: General name-based lookup")
    general_idx, general_id, general_name = find_disease_index_by_name(disease_name)
    
    if general_idx is not None:
        print(f"Found disease: {general_name}")
        print(f"ID: {general_id}")
        print(f"Index: {general_idx}")
    else:
        print(f"Could not find '{disease_name}' by general name lookup")