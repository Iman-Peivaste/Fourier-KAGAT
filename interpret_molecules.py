# Author: Iman Peivaste
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fourier-KAGAT Model Interpretation Script

This script visualizes attention weights from trained Fourier-KAGAT models to understand
which atoms and bonds the model focuses on when making predictions.

Author: Iman Peivaste
"""

import os
import argparse
from typing import Tuple, Optional, Dict, List
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import io
import yaml
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from model.fourier_kagat import FourierKAGAT
from utils.graph_path import path_complex_mol


def visualize_molecule_attention(
    smiles: str,
    molecule_id: str,
    model: nn.Module,
    device: torch.device,
    encode_dim: List[int],
    encoder_atom: str,
    encoder_bond: str,
    output_file: Optional[str] = None
) -> Optional[Dict]:
    """
    Extract and visualize attention weights for a molecule.
    
    Args:
        smiles: SMILES string of molecule
        molecule_id: Identifier for the molecule
        model: Trained Fourier-KAGAT model
        device: Computing device
        encode_dim: [node_dim, edge_dim]
        encoder_atom: Atom encoder type
        encoder_bond: Bond encoder type
        output_file: Optional output file path
        
    Returns:
        Dictionary containing attention statistics and visualization data
    """
    print(f"Processing {molecule_id}: {smiles}")
    
    try:
        graph = path_complex_mol(smiles, encoder_atom, encoder_bond)
        if graph is None:
            print(f"Failed to convert SMILES to graph: {smiles}")
            return None
        
        graph = graph.to(device)
        node_feats = graph.ndata['feat'].to(device)
        edge_feats = graph.edata['feat'].to(device)
        
        model.eval()
        with torch.no_grad():
            prediction, attentions = model(
                graph, node_feats, edge_feats, get_attention=True
            )
        
        pred_value = prediction.item()
        print(f"Prediction: {pred_value:.4f}")
        
        # Process attention from last layer
        final_attention = attentions[-1]
        
        # Average across attention heads
        if final_attention.dim() == 3:
            edge_weights = final_attention.mean(dim=1).squeeze().cpu().numpy()
        else:
            edge_weights = final_attention.mean(dim=1).cpu().numpy()
        
        # Normalize edge weights
        if edge_weights.max() > edge_weights.min():
            edge_weights_norm = (
                (edge_weights - edge_weights.min()) /
                (edge_weights.max() - edge_weights.min())
            )
        else:
            edge_weights_norm = np.ones_like(edge_weights) * 0.5
        
        # Map edge attention to atoms
        src, dst = graph.edges()
        src = src.cpu().numpy()
        dst = dst.cpu().numpy()
        
        mol_with_h = Chem.AddHs(Chem.MolFromSmiles(smiles))
        num_atoms = mol_with_h.GetNumAtoms()
        atom_weights = np.zeros(num_atoms)
        
        # Aggregate edge weights to atoms (max aggregation)
        for i, (s, d) in enumerate(zip(src, dst)):
            weight = edge_weights_norm[i]
            atom_weights[d] = max(atom_weights[d], weight)
        
        # Normalize atom weights
        if atom_weights.max() > atom_weights.min():
            atom_weights_norm = (
                (atom_weights - atom_weights.min()) /
                (atom_weights.max() - atom_weights.min())
            )
        else:
            atom_weights_norm = np.ones_like(atom_weights) * 0.5
        
        # Filter non-hydrogen atoms for visualization
        non_h_indices = []
        non_h_weights = []
        
        for atom in mol_with_h.GetAtoms():
            if atom.GetSymbol() != 'H':
                idx = atom.GetIdx()
                non_h_indices.append(idx)
                non_h_weights.append(atom_weights_norm[idx])
        
        # Create visualization
        highlight_atoms = []
        highlight_colors = {}
        
        threshold = np.percentile(non_h_weights, 70)
        
        for idx, weight in zip(non_h_indices, non_h_weights):
            if weight > threshold:
                highlight_atoms.append(idx)
                color = cm.jet(weight)
                highlight_colors[idx] = (color[0], color[1], color[2])
        
        # Draw molecule
        drawer = rdMolDraw2D.MolDraw2DCairo(600, 600)
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer,
            mol_with_h,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=highlight_colors
        )
        
        drawer.FinishDrawing()
        png_data = drawer.GetDrawingText()
        
        # Save image
        if output_file is None:
            output_file = f"attention_{molecule_id}.png"
        
        with open(output_file, 'wb') as f:
            f.write(png_data)
        
        print(f"Visualization saved to {output_file}")
        print(f"Max attention: {atom_weights_norm.max():.4f}")
        print(f"Number of highlighted atoms: {len(highlight_atoms)}")
        
        return {
            'molecule_id': molecule_id,
            'smiles': smiles,
            'prediction': pred_value,
            'atom_weights': atom_weights_norm,
            'edge_weights': edge_weights_norm,
            'max_attention': atom_weights_norm.max(),
            'output_file': output_file
        }
        
    except Exception as e:
        print(f"Error processing {molecule_id}: {e}")
        return None


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interpret Fourier-KAGAT model predictions using attention weights"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/gat_path.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model_classification.pth",
        help="Path to trained model file"
    )
    parser.add_argument(
        "--smiles",
        type=str,
        nargs='+',
        help="SMILES strings to interpret"
    )
    parser.add_argument(
        "--ids",
        type=str,
        nargs='+',
        help="Molecule IDs (same order as SMILES)"
    )
    
    args = parser.parse_args()
    
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if not hasattr(args, key):
                setattr(args, key, value)
    
    return args


def main() -> None:
    """Main interpretation function."""
    args = parse_arguments()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model parameters
    encode_dim = [92, 21]  # CGCNN: 92, dim_14: 21
    target_dim = 1
    grid_size = args.grid
    num_heads = args.head
    num_layers = args.num_layers
    pooling = args.pooling
    encoder_atom = args.encoder_atom
    encoder_bond = args.encoder_bond
    
    # Create model
    model = FourierKAGAT(
        in_node_dim=encode_dim[0],
        in_edge_dim=encode_dim[1],
        hidden_dim=64,
        out_1=32,
        out_2=target_dim,
        grid_size=grid_size,
        head=num_heads,
        layer_num=num_layers,
        pooling=pooling
    )
    
    # Load trained weights
    if os.path.exists(args.model):
        model.load_state_dict(
            torch.load(args.model, map_location=device, weights_only=False)
        )
        print(f"Loaded model from {args.model}")
    else:
        print(f"Warning: Model file not found: {args.model}")
        print("Using random weights (interpretation will be meaningless)")
    
    model = model.to(device)
    model.eval()
    
    # Process molecules
    if args.smiles:
        smiles_list = args.smiles
        if args.ids:
            ids_list = args.ids
        else:
            ids_list = [f"mol_{i+1}" for i in range(len(smiles_list))]
    else:
        # Default examples
        smiles_list = [
            "O=Cc1cc(C=O)c(O)c(C=O)c1",  # Active molecule
            "O=C1c2ccccc2C(=O)C1(O)O"    # Inactive molecule
        ]
        ids_list = ["ID486_Active", "ID331_Inactive"]
    
    results = []
    for smiles, mol_id in zip(smiles_list, ids_list):
        result = visualize_molecule_attention(
            smiles=smiles,
            molecule_id=mol_id,
            model=model,
            device=device,
            encode_dim=encode_dim,
            encoder_atom=encoder_atom,
            encoder_bond=encoder_bond
        )
        if result:
            results.append(result)
    
    print(f"\nProcessed {len(results)} molecules successfully")


if __name__ == '__main__':
    main()

