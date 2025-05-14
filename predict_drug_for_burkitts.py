#!/usr/bin/env python3
"""
predict_drug_for_burkitts.py

This script provides functionality to predict the likelihood that a specific drug
can be used to treat Burkitt's lymphoma, using a pretrained TxGNN model.
"""

import os
import sys
import torch
import pandas as pd
from txgnn import TxData, TxGNN, TxEval

class BurkittsDrugPredictor:
    """Class to predict drug efficacy for Burkitt's lymphoma treatment"""
    
    def __init__(self, data_path="./data", checkpoint_path="./checkpoints_all_seeds/TxGNN_1_random"):
        """
        Initialize the predictor with the specified data and checkpoint paths.
        
        Args:
            data_path: Path to the data directory
            checkpoint_path: Path to the pretrained model checkpoint
        """
        # Set device
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize data
        print("Loading data...")
        self.tx_data = TxData(data_path)
        self.tx_data.prepare_split(split='random', seed=42)
        
        # Load model
        print("Loading pretrained model...")
        self.tx_gnn = TxGNN(data=self.tx_data, device=self.device)
        self.tx_gnn.load_pretrained(checkpoint_path)
        
        # Initialize evaluator
        self.tx_eval = TxEval(model=self.tx_gnn)
        
        # Load entity mappings
        print("Loading entity mappings...")
        self.mappings = self.tx_data.retrieve_id_mapping()
        
        # Burkitt's lymphoma disease index
        self.burkitts_idx = 7243.0
        
        # Verify Burkitt's lymphoma is in the dataset
        try:
            disease_id = self.mappings['idx2id_disease'][self.burkitts_idx]
            disease_name = self.mappings['id2name_disease'][disease_id]
            print(f"Burkitt's lymphoma identified as: {disease_name} (ID: {disease_id}, Index: {self.burkitts_idx})")
        except KeyError:
            print("Warning: Burkitt's lymphoma index not found in dataset")
            
        # Create reverse mappings
        self._create_reverse_mappings()
    
    def _create_reverse_mappings(self):
        """Create reverse mappings from names to indices"""
        # Drug name to index mapping
        self.drug_name_to_idx = {}
        for idx, id_val in self.mappings['idx2id_drug'].items():
            try:
                name = self.mappings['id2name_drug'][id_val]
                self.drug_name_to_idx[name.lower()] = idx
            except KeyError:
                continue
        
        # Disease name to index mapping
        self.disease_name_to_idx = {}
        for idx, id_val in self.mappings['idx2id_disease'].items():
            try:
                name = self.mappings['id2name_disease'][id_val]
                self.disease_name_to_idx[name.lower()] = idx
            except KeyError:
                continue
    
    def predict_for_burkitts(self, drug_name=None, relation='indication', top_n=10):
        """
        Predict likelihood of a drug treating Burkitt's lymphoma.
        
        Args:
            drug_name: Name of the drug to evaluate. If None, returns predictions for all drugs.
            relation: Relationship type ('indication', 'contraindication', or 'off-label')
            top_n: Number of top drugs to return if drug_name is None
            
        Returns:
            If drug_name is provided: Score for the specific drug
            If drug_name is None: DataFrame with top_n drugs and their scores
        """
        # Get all predictions for Burkitt's lymphoma
        result = self.tx_eval.eval_disease_centric(
            disease_idxs=[self.burkitts_idx],
            relation=relation,
            verbose=False,
            return_raw=True
        )
        
        # If raw data isn't returned as expected, try again with verbose output
        if not isinstance(result, dict) or 'result_df' not in result:
            print("Initial prediction failed, retrying with verbose output...")
            result = self.tx_eval.eval_disease_centric(
                disease_idxs=[self.burkitts_idx],
                relation=relation,
                verbose=True,
                return_raw=True
            )
        
        # Create a DataFrame from results
        if isinstance(result, dict) and 'result_df' in result:
            predictions_df = result['result_df']
            
            # Add drug names to the predictions
            predictions_df['drug_name'] = predictions_df['drug_idx'].apply(
                lambda x: self.mappings['id2name_drug'].get(
                    self.mappings['idx2id_drug'].get(x, ''), 'Unknown'
                )
            )
            
            # If specific drug requested
            if drug_name:
                drug_name_lower = drug_name.lower()
                if drug_name_lower in self.drug_name_to_idx:
                    drug_idx = self.drug_name_to_idx[drug_name_lower]
                    # Find this drug in the predictions
                    drug_result = predictions_df[predictions_df['drug_idx'] == drug_idx]
                    if not drug_result.empty:
                        score = drug_result['score'].values[0]
                        rank = drug_result['rank'].values[0]
                        return {
                            'drug_name': drug_name,
                            'score': score,
                            'rank': rank,
                            'total_drugs': len(predictions_df)
                        }
                    else:
                        return {
                            'drug_name': drug_name,
                            'error': 'Drug found in database but not in prediction results'
                        }
                else:
                    return {
                        'drug_name': drug_name,
                        'error': 'Drug not found in database'
                    }
            else:
                # Return top N predictions
                return predictions_df.sort_values('score', ascending=False).head(top_n)[
                    ['drug_name', 'score', 'rank']
                ].reset_index(drop=True)
        else:
            if drug_name:
                return {
                    'drug_name': drug_name,
                    'error': 'Prediction failed'
                }
            else:
                return pd.DataFrame(columns=['drug_name', 'score', 'rank'])
    
    def predict_custom_drug_disease(self, drug_name, disease_name, relation='indication'):
        """
        Predict likelihood for a custom drug-disease pair.
        
        Args:
            drug_name: Name of the drug
            disease_name: Name of the disease
            relation: Relationship type ('indication', 'contraindication', or 'off-label')
            
        Returns:
            Prediction score or error message
        """
        drug_name_lower = drug_name.lower()
        disease_name_lower = disease_name.lower()
        
        # Check if drug and disease exist in database
        if drug_name_lower not in self.drug_name_to_idx:
            return {'error': f"Drug '{drug_name}' not found in database"}
        
        if disease_name_lower not in self.disease_name_to_idx:
            return {'error': f"Disease '{disease_name}' not found in database"}
        
        # Get indices
        drug_idx = self.drug_name_to_idx[drug_name_lower]
        disease_idx = self.disease_name_to_idx[disease_name_lower]
        
        # Create DataFrame with the pair to predict
        pairs_df = pd.DataFrame({
            'x_idx': [drug_idx],
            'relation': [relation],
            'y_idx': [disease_idx]
        })
        
        # Get prediction
        try:
            predictions = self.tx_gnn.predict(pairs_df)
            # The prediction structure depends on the specific model implementation
            # We need to extract the score from the prediction object
            if predictions and len(predictions) > 0:
                # Try to extract score (implementation may vary)
                for etype, pred in predictions.items():
                    if len(pred[0]) > 0:
                        score = float(pred[0].cpu().detach().numpy()[0])
                        return {
                            'drug_name': drug_name,
                            'disease_name': disease_name,
                            'relation': relation,
                            'score': score
                        }
            
            return {
                'drug_name': drug_name,
                'disease_name': disease_name,
                'relation': relation,
                'error': 'Could not extract prediction score'
            }
        except Exception as e:
            return {
                'drug_name': drug_name,
                'disease_name': disease_name,
                'relation': relation,
                'error': f'Prediction failed: {str(e)}'
            }
    
    def list_available_drugs(self, limit=20):
        """Return a list of available drugs in the database"""
        drug_names = list(self.drug_name_to_idx.keys())
        if limit and limit < len(drug_names):
            return sorted(drug_names)[:limit]
        return sorted(drug_names)
    
    def get_burkitts_info(self):
        """Return information about Burkitt's lymphoma from the database"""
        try:
            disease_id = self.mappings['idx2id_disease'][self.burkitts_idx]
            disease_name = self.mappings['id2name_disease'][disease_id]
            return {
                'name': disease_name,
                'id': disease_id,
                'index': self.burkitts_idx
            }
        except KeyError:
            return {'error': 'Burkitt\'s lymphoma not found in database'}


# Example usage
if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        drug_name = sys.argv[1]
        predictor = BurkittsDrugPredictor()
        result = predictor.predict_for_burkitts(drug_name)
        print(f"\nPrediction for {drug_name} treating Burkitt's lymphoma:")
        for k, v in result.items():
            print(f"{k}: {v}")
    else:
        # No drug specified, show top predictions
        predictor = BurkittsDrugPredictor()
        print("\nTop 10 predicted drugs for treating Burkitt's lymphoma:")
        top_drugs = predictor.predict_for_burkitts()
        print(top_drugs)
        
        # Show a few available drugs
        print("\nSample of available drugs:")
        sample_drugs = predictor.list_available_drugs(10)
        for i, drug in enumerate(sample_drugs, 1):
            print(f"{i}. {drug}")