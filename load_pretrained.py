from txgnn import TxData, TxGNN, TxEval
import pandas as pd
import dgl
import os
import torch
import sys

# Check if CUDA is available
device = 'cpu'
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    device = 'cuda:0'
else:
    print("CUDA is not available. Using CPU.")

# Check if checkpoint exists
checkpoint_path = './checkpoints_all_seeds/TxGNN_1_random'
if not os.path.exists(checkpoint_path):
    print(f"Error: Checkpoint directory {checkpoint_path} does not exist.")
    sys.exit(1)

if not os.path.exists(os.path.join(checkpoint_path, 'model.pt')):
    print(f"Error: Model file {os.path.join(checkpoint_path, 'model.pt')} does not exist.")
    sys.exit(1)

# Create Data Split
try:
    tx_data = TxData('./data')
    tx_data.prepare_split(split='random', seed=1)
    
    # Check if validation data exists and is not empty
    if tx_data.df_valid is None or len(tx_data.df_valid) == 0:
        print("Error: Validation data is empty. This will cause an error in heterograph creation.")
        print("Please ensure the validation data is properly loaded.")
        sys.exit(1)
        
    tx_gnn = TxGNN(data=tx_data, weight_bias_track=False, proj_name='TxGNN', exp_name='TxGNN', device=device)

    # Check if the data is loaded correctly
    print("1. Checking tx_data attributes:")
    print("G:", tx_data.G)
    print("\ndf_train shape:", tx_data.df_train.shape)
    print("\nFirst few rows of df_train:")
    print(tx_data.df_train.head())

    # Then check the graph structure
    print("\n2. Graph information:")
    print("Node types:", tx_data.G.ntypes)
    print("Edge types:", tx_data.G.etypes)
    print("Number of nodes:", {ntype: tx_data.G.number_of_nodes(ntype) for ntype in tx_data.G.ntypes})
    print("Number of edges:", {etype: tx_data.G.number_of_edges(etype) for etype in tx_data.G.etypes})
    
    print("\n3. Validation data shape:", tx_data.df_valid.shape)
    print("First few rows of df_valid:")
    print(tx_data.df_valid.head())

    # Load Pretrained Model
    tx_gnn.load_pretrained(checkpoint_path)

    # Create evaluator and get predictions
    tx_eval = TxEval(model=tx_gnn)
    burkitts_idx = 7243.0
    result = tx_eval.eval_disease_centric(
        disease_idxs=[burkitts_idx],  # Burkitt's lymphoma
        relation='indication',
        save_result=False,
        verbose=True
    )
    
except dgl._ffi.base.DGLError as e:
    print(f"DGL Error: {e}")
    print("This could be due to empty heterograph or invalid graph structure.")
    print("Attempting to debug the issue...")
    
    # Print validation data info for debugging
    if hasattr(tx_data, 'df_valid'):
        print("\nValidation data info:")
        print("Shape:", tx_data.df_valid.shape)
        print("Columns:", tx_data.df_valid.columns.tolist())
        print("Value counts for relation:", tx_data.df_valid['relation'].value_counts())
        
    sys.exit(1)
    
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)

