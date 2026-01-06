#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Splitting Utilities

This module provides dataset splitting strategies for molecular property prediction,
including scaffold-based splitting which ensures molecules with similar scaffolds
are grouped together.

Author: Iman Peivaste
"""

from typing import List, Tuple
import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold


def generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """
    Generate Bemis-Murcko scaffold from SMILES string.
    
    The scaffold represents the core structure of a molecule, removing
    side chains and functional groups.
    
    Args:
        smiles: SMILES string representation of molecule
        include_chirality: Whether to include stereochemical information
        
    Returns:
        Scaffold SMILES string
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles,
        includeChirality=include_chirality
    )
    return scaffold


class Splitter:
    """
    Abstract base class for dataset splitters.
    """
    def __init__(self):
        pass


class ScaffoldSplitter(Splitter):
    """
    Split dataset by Bemis-Murcko scaffolds.
    
    This splitter groups molecules by their scaffold structure, ensuring that
    molecules with similar core structures are kept in the same split. This
    is important for molecular property prediction as it tests generalization
    to novel scaffolds.
    
    Adapted from DeepChem scaffold splitter implementation.
    """
    
    def __init__(self):
        super(ScaffoldSplitter, self).__init__()
    
    def split(
        self,
        dataset: List,
        frac_train: float = None,
        frac_valid: float = None,
        frac_test: float = None
    ) -> Tuple[List, List, List]:
        """
        Split dataset into train, validation, and test sets based on scaffolds.
        
        Args:
            dataset: List of data samples, where each sample is a list/tuple
                     with SMILES string as the first element (dataset[i][0])
            frac_train: Fraction of data for training set
            frac_valid: Fraction of data for validation set
            frac_test: Fraction of data for test set
            
        Returns:
            Tuple of (train_dataset, valid_dataset, test_dataset)
        """
        np.testing.assert_almost_equal(
            frac_train + frac_valid + frac_test, 1.0
        )
        
        num_samples = len(dataset)
        
        # Group samples by scaffold
        scaffold_to_indices = {}
        for idx in range(num_samples):
            smiles = dataset[idx][0]
            scaffold = generate_scaffold(smiles, include_chirality=True)
            if scaffold not in scaffold_to_indices:
                scaffold_to_indices[scaffold] = [idx]
            else:
                scaffold_to_indices[scaffold].append(idx)
        
        # Sort scaffolds by size (largest first), then by first index
        scaffold_to_indices = {
            key: sorted(value) for key, value in scaffold_to_indices.items()
        }
        scaffold_sets = [
            scaffold_set
            for (scaffold, scaffold_set) in sorted(
                scaffold_to_indices.items(),
                key=lambda x: (len(x[1]), x[1][0]),
                reverse=True
            )
        ]

        # Assign scaffolds to splits based on target fractions
        train_cutoff = frac_train * num_samples
        valid_cutoff = (frac_train + frac_valid) * num_samples
        train_indices = []
        valid_indices = []
        test_indices = []
        
        for scaffold_set in scaffold_sets:
            if len(train_indices) + len(scaffold_set) > train_cutoff:
                if (len(train_indices) + len(valid_indices) +
                        len(scaffold_set) > valid_cutoff):
                    test_indices.extend(scaffold_set)
                else:
                    valid_indices.extend(scaffold_set)
            else:
                train_indices.extend(scaffold_set)

        # Verify no overlap between splits
        assert len(set(train_indices).intersection(set(valid_indices))) == 0
        assert len(set(test_indices).intersection(set(valid_indices))) == 0
        assert len(set(train_indices).intersection(set(test_indices))) == 0

        # Create split datasets
        train_dataset = [dataset[i] for i in train_indices]
        valid_dataset = [dataset[i] for i in valid_indices]
        test_dataset = [dataset[i] for i in test_indices]

        return train_dataset, valid_dataset, test_dataset

