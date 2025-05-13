"""
Script to evaluate the relationship between a specific drug and Burkitt's lymphoma using TxGNN
"""

from txgnn import TxData, TxGNN, TxEval
import pandas as pd

def find_drug_disease_eval_idx(drug_name):
    """Find the disease evaluation index for a given drug name in TxGNN.
    
    Args:
        drug_name (str): Name of the drug to look up
        
    Returns:
        float: The disease evaluation index for the drug if found, None otherwise
        str: The drug ID if found, None otherwise
    """
    try:
        # Initialize TxData which will load the knowledge graph
        data = TxData('./data')
        
        # Prepare the data split (required before accessing mappings)
        data.prepare_split(split='disease_eval', disease_eval_idx=7243.0)
        
        # Get the ID mappings
        mappings = data.retrieve_id_mapping()
        id2name_drug = mappings['id2name_drug']
        idx2id_drug = mappings['idx2id_drug']
        
        # Find the drug ID by matching the name
        drug_id = None
        for id_, name in id2name_drug.items():
            if name.lower() == drug_name.lower():
                drug_id = id_
                break
                
        if drug_id is None:
            print(f"Drug '{drug_name}' not found in the knowledge graph")
            return None, None
            
        # Find the drug's evaluation index
        for idx, id_ in idx2id_drug.items():
            if id_ == drug_id:
                return float(idx), drug_id
                
        print(f"Could not find evaluation index for drug '{drug_name}' (ID: {drug_id})")
        return None, drug_id
        
    except Exception as e:
        print(f"Error finding drug evaluation index: {e}")
        return None, None

def load_knowledge_graph():
    """Load and process the knowledge graph data"""
    try:
        # Initialize TxData which will download required files
        data = TxData(data_folder_path='./data')
        return data
    except Exception as e:
        print(f"Error loading knowledge graph: {e}")
        return None

def evaluate_drug_disease_relationship(drug_name=None):
    """
    Evaluate potential treatments for Burkitt's lymphoma.
    If drug_name is provided, focus on that specific drug.
    """
    # Initialize the data
    print("Initializing TxData...")
    data = load_knowledge_graph()
    if data is None:
        return None
    
    # Prepare split for Burkitt's lymphoma (ID: 7243.0)
    print("Preparing data split for Burkitt's lymphoma evaluation...")
    data.prepare_split(split='disease_eval', disease_eval_idx=7243.0)
    
    # Initialize and load the model
    print("Initializing TxGNN model...")
    try:
        model = TxGNN(data=data, device='cpu')  # Change to 'cuda:0' if using GPU
        model.load_pretrained('./model_ckpt')
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Create evaluator
    print("Setting up evaluator...")
    evaluator = TxEval(model=model)
    
    # Evaluate drug-disease relationships
    print("Running evaluation...")
    result = evaluator.eval_disease_centric(
        disease_idxs=[7243.0],  # Burkitt's lymphoma ID
        relation='indication',
        save_result=False,
        verbose=True  # Show detailed results
    )
    
    return result

if __name__ == "__main__":
    # Example usage
    results = evaluate_drug_disease_relationship()
    if results:
        print("\nEvaluation Results:")
        print(results)
    else:
        print("\nEvaluation failed. Please check the error messages above.")


if __name__ == "__main__":
    # Example usage with acetazolamide
    drug_name = "acetazolamide"
    results = evaluate_drug_disease_relationship(drug_name)
    print("\nResults:")
    print(results)
