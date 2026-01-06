#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Molecular Graph Construction Utilities

This module provides functions to convert SMILES strings into graph representations
suitable for graph neural networks, including 3D conformer generation and feature encoding.

Author: Iman Peivaste
"""

from typing import Optional, Tuple, List
import numpy as np
import torch
import dgl
from rdkit import Chem
from rdkit.Chem import AllChem
from jarvis.core.specie import chem_data, get_node_attributes


def calculate_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two 3D points.
    
    Args:
        point_a: First point coordinates [x, y, z]
        point_b: Second point coordinates [x, y, z]
        
    Returns:
        Euclidean distance
    """
    vector = point_b - point_a
    return np.linalg.norm(vector)


def get_bond_length_approximation(bond_type: Chem.BondType) -> float:
    """
    Get approximate bond length based on bond type.
    
    Args:
        bond_type: RDKit bond type
        
    Returns:
        Approximate bond length in Angstroms
    """
    bond_length_dict = {
        Chem.BondType.SINGLE: 1.0,
        Chem.BondType.DOUBLE: 1.4,
        Chem.BondType.TRIPLE: 1.8,
        Chem.BondType.AROMATIC: 1.5
    }
    return bond_length_dict.get(bond_type, 1.0)


def encode_bond_features(bond: Chem.Bond) -> List[float]:
    """
    Encode bond features into a 21-dimensional vector.
    
    Features include: bond direction, bond type, bond length, ring membership.
    
    Args:
        bond: RDKit bond object
        
    Returns:
        21-dimensional feature vector
    """
    # Bond direction (7 dimensions)
    bond_dir = [0.0] * 7
    bond_dir[bond.GetBondDir()] = 1.0
    
    # Bond type (4 dimensions)
    bond_type = [0.0] * 4
    bond_type[int(bond.GetBondTypeAsDouble()) - 1] = 1.0
    
    # Bond length and squared length
    bond_length = get_bond_length_approximation(bond.GetBondType())
    
    # Ring membership (2 dimensions)
    in_ring = [0.0, 0.0]
    in_ring[int(bond.IsInRing())] = 1.0
    
    # Placeholder for additional features (6 dimensions)
    additional_features = [0.0] * 6

    edge_encode = (
        bond_dir + bond_type + [bond_length, bond_length**2] +
        in_ring + additional_features
    )
    return edge_encode


def compute_non_bonded_interactions(
    charge_list: List[float],
    atom_i: int,
    atom_j: int,
    distance: float
) -> List[float]:
    """
    Compute non-bonded interaction features between two atoms.
    
    Args:
        charge_list: List of atomic charges
        atom_i: Index of first atom
        atom_j: Index of second atom
        distance: Distance between atoms
        
    Returns:
        6-dimensional feature vector [q_i, q_j, q_i*q_j, 1/r, 1/r^6, 1/r^12]
    """
    charge_list = [float(charge) for charge in charge_list]
    q_i = [charge_list[atom_i]]
    q_j = [charge_list[atom_j]]
    q_ij = [charge_list[atom_i] * charge_list[atom_j]]
    
    # Distance-based features
    inv_dist = [1.0 / distance] if distance > 0 else [0.0]
    inv_dist_6 = [1.0 / (distance**6)] if distance > 0 else [0.0]
    inv_dist_12 = [1.0 / (distance**12)] if distance > 0 else [0.0]

    return q_i + q_j + q_ij + inv_dist + inv_dist_6 + inv_dist_12


def check_mmff_force_field(mol: Chem.Mol, seed: int = 42) -> bool:
    """
    Check if MMFF force field can be applied to molecule.
    
    Args:
        mol: RDKit molecule object with conformer
        seed: Random seed for deterministic 3D coordinate generation
        
    Returns:
        True if MMFF can be applied, False otherwise
    """
    try:
        # FIX: Added randomSeed to ensure deterministic 3D coordinates
        AllChem.EmbedMolecule(mol, randomSeed=seed)
        
        # Optimize geometry
        AllChem.MMFFGetMoleculeForceField(
            mol, AllChem.MMFFGetMoleculeProperties(mol)
        )
        return True
    except (ValueError, RuntimeError):
        return False


def check_uff_force_field(mol: Chem.Mol) -> bool:
    """
    Check if UFF force field can be applied to molecule.
    
    Args:
        mol: RDKit molecule object with conformer
        
    Returns:
        True if UFF can be applied, False otherwise
    """
    try:
        AllChem.EmbedMolecule(mol)
        AllChem.UFFGetMoleculeForceField(mol)
        return True
    except (ValueError, RuntimeError):
        return False


def check_bond_exists(
    src_list: List[int],
    dst_list: List[int],
    atom_i: int,
    atom_j: int
) -> bool:
    """
    Check if a bond already exists between two atoms.
    
    Args:
        src_list: List of source atom indices
        dst_list: List of destination atom indices
        atom_i: First atom index
        atom_j: Second atom index
        
    Returns:
        True if bond exists, False otherwise
    """
    for idx in range(len(src_list)):
        if src_list[idx] == atom_i and dst_list[idx] == atom_j:
            return True
    return False


def clean_tensor_features(feature_list: List[float]) -> List[float]:
    """
    Clean feature list by handling NaN and Inf values.
    
    Args:
        feature_list: List of feature values
        
    Returns:
        Cleaned feature list with NaN and Inf replaced
    """
    import math
    
    # Check for problematic values
    has_nan = any(
        isinstance(x, float) and math.isnan(x) for x in feature_list
    )
    has_inf = any(
        isinstance(x, float) and (x == float('inf') or x == float('-inf'))
        for x in feature_list
    )

    if has_nan or has_inf:
        # Replace NaN with 0
        cleaned = [
            0.0 if isinstance(x, float) and math.isnan(x) else x
            for x in feature_list
        ]
        # Replace Inf with 1, -Inf with -1
        cleaned = [
            1.0 if x == float('inf') else (-1.0 if x == float('-inf') else x)
            for x in cleaned
        ]
        return cleaned
    else:
        return feature_list


def atom_to_graph(
    smiles: str,
    encoder_atom: str,
    encoder_bond: str
) -> Optional[dgl.DGLGraph]:
    """
    Convert SMILES string to DGL graph representation.
    
    Args:
        smiles: SMILES string representation of molecule
        encoder_atom: Atom feature encoding method (e.g., 'cgcnn')
        encoder_bond: Bond feature encoding method (e.g., 'dim_14')
        
    Returns:
        DGL graph object with node and edge features, or None if conversion fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    mol = Chem.AddHs(mol)
    
    # Check molecule size (limit to 700 atoms)
    smiles_with_h = Chem.MolToSmiles(mol)
    atom_count = len([
        c for c in smiles_with_h if c not in ['[', ']', '(', ')']
    ])
    
    if atom_count > 700:
        return None
    
    # Try to generate conformer and apply force field
    # Ensure the function call uses the fixed function with seed for determinism
    if not check_mmff_force_field(mol, seed=42):
        return None
    
    num_conformers = mol.GetNumConformers()
    if num_conformers == 0:
        return None
    
    # Compute Gasteiger charges
    AllChem.ComputeGasteigerCharges(mol)
    
    # Extract atom features and coordinates
    atom_features = []
    coordinates = []
    atom_charges = []
    
    for atom_idx, atom in enumerate(mol.GetAtoms()):
        # Get atom features using specified encoder
        atom_feat = list(get_node_attributes(
            atom.GetSymbol(),
            atom_features=encoder_atom
        ))
        atom_features.append(atom_feat)
        
        # Get 3D coordinates
        pos = mol.GetConformer().GetAtomPosition(atom_idx)
        coordinates.append([pos.x, pos.y, pos.z])
        
        # Get Gasteiger charge
        charge = atom.GetProp("_GasteigerCharge")
        atom_charges.append(charge)
    
    # Extract bond features and build edge lists
    edge_features = []
    src_list = []
    dst_list = []
    edge_ids = []
    
    for bond in mol.GetBonds():
        src = bond.GetBeginAtomIdx()
        dst = bond.GetEndAtomIdx()
        
        # Add bidirectional edges
        src_list.extend([src, dst])
        dst_list.extend([dst, src])
        
        # Compute bond features
        bond_feat = encode_bond_features(bond)
        edge_features.append(bond_feat)
        edge_features.append(bond_feat)  # Same for reverse edge
        
        edge_ids.append([1])  # Covalent bond
        edge_ids.append([1])
    
    # Add non-bonded interactions within cutoff distance (5 Angstroms)
    cutoff_distance = 5.0
    for i in range(len(coordinates)):
        coord_i = np.array(coordinates[i])
        for j in range(i + 1, len(coordinates)):
            coord_j = np.array(coordinates[j])
            distance = calculate_distance(coord_i, coord_j)
            
            if 0 < distance <= cutoff_distance:
                # Check if covalent bond already exists
                if not check_bond_exists(src_list, dst_list, i, j):
                    src_list.extend([i, j])
                    dst_list.extend([j, i])
                    
                    # Non-bonded interaction features
                    non_bonded_feat = [0.0] * 15
                    non_bonded_feat.extend(
                        compute_non_bonded_interactions(
                            atom_charges, i, j, distance
                        )
                    )
                    
                    # Clean features
                    cleaned_feat = clean_tensor_features(non_bonded_feat)
                    edge_features.append(cleaned_feat)
                    edge_features.append(cleaned_feat)
                    
                    edge_ids.append([0])  # Non-bonded interaction
                    edge_ids.append([0])
    
    # Convert to tensors
    coord_tensor = torch.tensor(coordinates, dtype=torch.float32)
    edge_feat_tensor = torch.tensor(edge_features, dtype=torch.float32)
    edge_id_tensor = torch.tensor(edge_ids, dtype=torch.float32)
    node_feat_tensor = torch.tensor(atom_features, dtype=torch.float32)
    
    # Create DGL graph
    num_atoms = mol.GetNumAtoms()
    graph = dgl.DGLGraph()
    graph.add_nodes(num_atoms)
    graph.add_edges(src_list, dst_list)
    
    # Add node and edge data
    graph.ndata['feat'] = node_feat_tensor
    graph.ndata['coor'] = coord_tensor
    graph.edata['feat'] = edge_feat_tensor
    graph.edata['id'] = edge_id_tensor
    
    return graph


def path_complex_mol(
    smiles: str,
    encoder_atom: str,
    encoder_bond: str
) -> Optional[dgl.DGLGraph]:
    """
    Main function to convert SMILES to graph representation.
    
    This is a wrapper function that handles the conversion process.
    
    Args:
        smiles: SMILES string representation of molecule
        encoder_atom: Atom feature encoding method
        encoder_bond: Bond feature encoding method
        
    Returns:
        DGL graph object or None if conversion fails
    """
    graph = atom_to_graph(smiles, encoder_atom, encoder_bond)
    return graph if graph is not None else None

