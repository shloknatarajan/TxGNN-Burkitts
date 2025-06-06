"""
Function to find disease index by name in TxGNN-Burkitts
"""

import os
import pandas as pd
import numpy as np
from txgnn import TxData
from typing import List, Tuple, Optional
from fuzzywuzzy import fuzz, process
from loguru import logger

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

def get_all_disease_names_from_files():
    """Get all disease names from the CSV files"""
    disease_files = [
        'anemia.csv',
        'adrenal_gland.csv',
        'autoimmune.csv',
        'cardiovascular.csv',
        'cell_proliferation.csv',
        'diabetes.csv',
        'mental_health.csv',
        'metabolic_disorder.csv',
        'neurodigenerative.csv'
    ]
    base_path = './data/disease_files'
    all_disease_names = set()
    for file_name in disease_files:
        file_path = os.path.join(base_path, file_name)
        df = pd.read_csv(file_path)
        all_disease_names.update(df['node_name'].unique())
    
    return list(all_disease_names)

def get_all_disease_names_from_graph(refresh: bool = False):
    """Get all disease names from the graph"""
    if not refresh and os.path.exists('./data/disease_files/all_disease_names.txt'):
        with open('./data/disease_files/all_disease_names.txt', 'r') as f:
            logger.info("Found all disease names in all_disease_names.txt")
            return [line.strip() for line in f]

    logger.info("Calculating all disease names")
    logger.info("Constructing graph")
    data = TxData('./data')
    data.prepare_split(split='full_graph', seed=42)
    logger.info("Retrieving mappings")
    mappings = data.retrieve_id_mapping()
    id2name_disease = mappings['id2name_disease']
    logger.info("Done getting all disease names from graph")

    # Save to file
    with open('./data/disease_files/all_disease_names.txt', 'w') as f:
        logger.info("Writing disease names to all_disease_names.txt")
        for name in id2name_disease.values():
            f.write(f"{name}\n")
    
    return list(id2name_disease.values())

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


def fuzzy_search_list(query: str, 
                      name_list: List[str],
                      method: str = "fuzzywuzzy",
                      threshold: float = 70.0,
                      limit: Optional[int] = 10) -> List[Tuple[str, float]]:
    """
    Search for a name using fuzzy matching.
    
    Args:
        query (str): The name to search for
        name_list (list): List of names to search through
        method (str): The matching method to use. Options: 
                      "fuzzywuzzy" - Uses fuzzywuzzy's process.extract
                      "difflib" - Uses difflib's get_close_matches
                      "regex" - Uses regex partial matching
        threshold (float): Minimum similarity score (0-100) for matches to be returned
        limit (int): Maximum number of results to return
    
    Returns:
        list: List of tuples containing (name, similarity_score)
    """
    query = query.lower().strip()
    results = []
    
    if method == "fuzzywuzzy":
        # Use fuzzywuzzy to find matches
        matches = process.extract(query, name_list, 
                                  scorer=fuzz.token_sort_ratio, 
                                  limit=limit)
        for name, score in matches:
            if score >= threshold:
                results.append((name, score))
    
    elif method == "difflib":
        # Use difflib to find matches
        matches = difflib.get_close_matches(query, name_list, n=limit, cutoff=threshold/100)
        
        # Calculate similarity scores
        for name in matches:
            similarity = difflib.SequenceMatcher(None, query, name).ratio() * 100
            results.append((name, similarity))
            
        # Sort by similarity score
        results.sort(key=lambda x: x[1], reverse=True)
    
    elif method == "regex":
        # Build a simple regex pattern for partial matching
        pattern = f".*{re.escape(query)}.*"
        regex = re.compile(pattern)
        
        # Find matches
        matches = []
        for name in name_list:
            if regex.match(name):
                # Calculate similarity using difflib
                similarity = difflib.SequenceMatcher(None, query, name).ratio() * 100
                if similarity >= threshold:
                    matches.append((name, similarity))
        
        # Sort by similarity and take top matches
        matches.sort(key=lambda x: x[1], reverse=True)
        for name, score in matches[:limit]:
            results.append((name, score))
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'fuzzywuzzy', 'difflib', or 'regex'")

    # Sort by similarity score
    results.sort(key=lambda x: x[1], reverse=True)
    # If no results, return empty list
    if not results:
        return [('', 0.0)]
    return results

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