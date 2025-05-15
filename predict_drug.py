#!/usr/bin/env python3
"""
predict_drug.py

This script provides functionality to predict the likelihood that a specific drug
can be used to treat a given disease, using a pretrained TxGNN model.
"""

import os
import sys
import torch
import pandas as pd
from txgnn import TxData, TxGNN, TxEval
from disease_lookup import fuzzy_search_list
from typing import Optional, Tuple, List
from loguru import logger

class DrugPredictor:
    """Class to predict drug efficacy for disease treatment"""
    
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
        
        # Ensure model is initialized before creating evaluator
        if not hasattr(self.tx_gnn, 'best_model'):
            self.tx_gnn.best_model = self.tx_gnn.model
        
        # Initialize evaluator
        self.tx_eval = TxEval(model=self.tx_gnn)
        
        # Load entity mappings
        print("Loading entity mappings...")
        self.mappings = self.tx_data.retrieve_id_mapping()
        
        # Create reverse mappings
        self._create_reverse_mappings()

        # load WHO drug names
        self.who_filtered = self._load_who_drug_names()
    
    def _load_who_drug_names(self):
        who_filtered = set()
        with open('who_essentials/who_filtered_db_ids.txt', 'r') as f:
            for line in f:
                who_filtered.add(line.strip())
        return who_filtered

    
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

    def predict_for_disease(self, disease_name, drug_name=None, relation='indication', top_n=10):
        """
        Predict likelihood of drugs treating a specific disease.
        
        Args:s
            disease_name: Name of the disease to evaluate
            drug_name: Name of the drug to evaluate. If None, returns predictions for all drugs.
            relation: Relationship type ('indication', 'contraindication', or 'off-label')
            top_n: Number of top drugs to return if drug_name is None
            
        Returns:
            If drug_name is provided: Score for the specific drug
            If drug_name is None: DataFrame with top_n drugs and their scores
        """
        # Check if disease exists
        disease_name_lower = disease_name.lower()
        if not self.verify_disease_name(disease_name_lower):
            return pd.DataFrame(columns=['drug_name', 'score', 'rank'])
        
        # Get disease index
        disease_idx = self.disease_name_to_idx[disease_name_lower]
        
        # Get all predictions for the disease
        logger.info(f"Predicting for disease '{disease_name}'...")
        result = self.tx_eval.eval_disease_centric(
            disease_idxs=[disease_idx],
            relation=relation,
            verbose=False,
            return_raw=True
        )
        
    
    def list_available_drugs(self, limit: Optional[int] = None):
        """Return a list of available drugs in the database"""
        drug_names = list(self.drug_name_to_idx.keys())
        if limit and limit < len(drug_names):
            return sorted(drug_names)[:limit]
        return sorted(drug_names)
    
    def list_available_diseases(self, limit: Optional[int] = None):
        """Return a list of available diseases in the database"""
        disease_names = list(self.disease_name_to_idx.keys())
        if limit and limit < len(disease_names):
            return sorted(disease_names)[:limit]
        return sorted(disease_names)
    
    def get_disease_info(self, disease_name):
        """Return information about a specific disease from the database"""
        disease_name_lower = disease_name.lower()
        if disease_name_lower not in self.disease_name_to_idx:
            return {'error': f"Disease '{disease_name}' not found in database"}
        
        disease_idx = self.disease_name_to_idx[disease_name_lower]
        try:
            disease_id = self.mappings['idx2id_disease'][disease_idx]
            disease_name_exact = self.mappings['id2name_disease'][disease_id]
            return {
                'name': disease_name_exact,
                'id': disease_id,
                'index': disease_idx
            }
        except KeyError:
            return {'error': f"Disease '{disease_name}' found in name mapping but not in ID mapping"}
    
    def fuzzy_search_disease(self, disease_name, limit: Optional[int] = 10) -> List[Tuple[str, int, float]]:
        """
        Fuzzy search for disease names
        
        Args:
            disease_name: Name of the disease to search for
            limit: Maximum number of results to return
        
        Returns:
            List of tuples containing the disease name, index, and similarity score
        """
        results = fuzzy_search_list(disease_name, self.disease_name_to_idx.keys(), limit=limit)
        return [(name, self.disease_name_to_idx[name.lower()], score) for name, score in results]
    
    def verify_disease_name(self, disease_name: str) -> Optional[Tuple[str, int]]:
        """
        Check if a disease name exists in the database and return the index if it does
        
        Args:
            disease_name: Name of the disease to check
        
        Returns:
            Tuple containing the disease name, index if the disease exists, None otherwise
        """
        if disease_name in self.disease_name_to_idx:
            return disease_name, self.disease_name_to_idx[disease_name]
        else:
            logger.error(f"Disease '{disease_name}' not found")
            similar_disease_names = [x[0] for x in self.fuzzy_search_disease(disease_name, limit=3)]
            if len(similar_disease_names) > 0:
                logger.error(f"Did you mean?: {similar_disease_names}")
            return None

# Example usage
if __name__ == "__main__":
    predictor = DrugPredictor()
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python predict_drug.py <disease_name> [drug_name]")
        print("\nAvailable diseases (sample):")
        sample_diseases = predictor.list_available_diseases(10)
        for i, disease in enumerate(sample_diseases, 1):
            print(f"{i}. {disease}")
        sys.exit(1)
    
    disease_name = sys.argv[1]
    
    if len(sys.argv) > 2:
        # Specific drug for disease requested
        drug_name = sys.argv[2]
        result = predictor.predict_for_disease(disease_name, drug_name)
        print(f"\nPrediction for {drug_name} treating {disease_name}:")
        for k, v in result.items():
            print(f"{k}: {v}")
    else:
        # No drug specified, show top predictions for disease
        print(f"\nTop 10 predicted drugs for treating {disease_name}:")
        top_drugs = predictor.predict_for_disease(disease_name)
        print(top_drugs)
        
        # Show a few available drugs
        print("\nSample of available drugs:")
        sample_drugs = predictor.list_available_drugs(10)
        for i, drug in enumerate(sample_drugs, 1):
            print(f"{i}. {drug}")