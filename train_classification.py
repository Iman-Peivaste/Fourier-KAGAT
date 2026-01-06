# Author: Iman Peivaste
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fourier-KAGAT Training Script for Binary Classification

This script trains the Fourier-KAGAT (Fourier-based Kolmogorov-Arnold Graph Attention Network)
model for binary classification tasks on molecular property prediction.

Author: Iman Peivaste
"""

import os
import argparse
import random
from typing import Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import yaml
import dgl
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, f1_score
)
from sklearn.model_selection import StratifiedKFold

from model.fourier_kagat import FourierKAGAT
from utils.splitters import ScaffoldSplitter
from utils.graph_path import path_complex_mol


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class MolecularDataset(Dataset):
    """
    Dataset class for molecular graphs and labels.
    
    Args:
        label_list: List of label tensors
        graph_list: List of DGL graph objects
    """
    
    def __init__(
        self,
        label_list: List[torch.Tensor],
        graph_list: List[dgl.DGLGraph]
    ) -> None:
        self.labels = label_list
        self.graphs = graph_list
        self.device = torch.device('cpu')

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dgl.DGLGraph]:
        label = self.labels[index].to(self.device)
        graph = self.graphs[index].to(self.device)
        return label, graph


def collate_batch(
    batch: List[Tuple[torch.Tensor, dgl.DGLGraph]]
) -> Tuple[torch.Tensor, dgl.DGLGraph]:
    """
    Collate function for batching molecular graphs.
    
    Args:
        batch: List of (label, graph) tuples
        
    Returns:
        Batched labels and graphs
    """
    labels, graphs = zip(*batch)
    labels = torch.stack(labels)
    batched_graph = dgl.batch(graphs)
    return labels, batched_graph


def has_zero_in_degree_nodes(graph: dgl.DGLGraph) -> bool:
    """
    Check if graph has nodes with zero in-degree.
    
    Args:
        graph: DGL graph object
        
    Returns:
        True if any node has zero in-degree
    """
    return (graph.in_degrees() == 0).any().item()


def try_alternative_smiles_conversion(
    smiles: str,
    encoder_atom: str,
    encoder_bond: str
) -> Tuple[Optional[dgl.DGLGraph], Optional[str]]:
    """
    Try alternative methods to convert SMILES to graph when primary method fails.
    
    Args:
        smiles: SMILES string
        encoder_atom: Atom encoder type
        encoder_bond: Bond encoder type
        
    Returns:
        Tuple of (graph, recovery_method) or (None, None) if all methods fail
    """
    from rdkit import Chem
    
    # Method 1: Try sanitizing the molecule
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is not None:
        try:
            Chem.SanitizeMol(mol)
            smiles_sanitized = Chem.MolToSmiles(mol)
            graph = path_complex_mol(smiles_sanitized, encoder_atom, encoder_bond)
            if graph is not None:
                return graph, "sanitized"
        except Exception:
            pass
    
    # Method 2: Try removing and re-adding hydrogens
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        try:
            mol_no_h = Chem.RemoveHs(mol)
            mol_with_h = Chem.AddHs(mol_no_h)
            smiles_reh = Chem.MolToSmiles(mol_with_h)
            graph = path_complex_mol(smiles_reh, encoder_atom, encoder_bond)
            if graph is not None:
                return graph, "re-hydrogenated"
        except Exception:
            pass
    
    # Method 3: Try canonical SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        try:
            smiles_canonical = Chem.MolToSmiles(mol, canonical=True)
            if smiles_canonical != smiles:
                graph = path_complex_mol(smiles_canonical, encoder_atom, encoder_bond)
                if graph is not None:
                    return graph, "canonical"
        except Exception:
            pass
    
    return None, None


def create_dataset(
    data_file: str,
    encoder_atom: str,
    encoder_bond: str,
    batch_size: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    data_dir: str = "data",
    processed_dir: str = "data/processed"
) -> None:
    """
    Create and save processed dataset from raw data files.
    
    Args:
        data_file: Name of dataset (without extension)
        encoder_atom: Atom feature encoder type
        encoder_bond: Bond feature encoder type
        batch_size: Batch size for training
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        data_dir: Directory containing raw data files
        processed_dir: Directory to save processed data
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    # For HER dataset, use 10-fold CV format
    if data_file == 'her':
        processed_file = os.path.join(processed_dir, f"{data_file}_10fold.pth")
    else:
        processed_file = os.path.join(processed_dir, f"{data_file}.pth")
    
    if os.path.exists(processed_file):
        return
    
    # Load data based on dataset type
    if data_file == 'her':
        df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        
        smiles_train = df_train['SMILES']
        her_values_train = df_train['HER (µmol/h)'].fillna(0)
        labels_train = (her_values_train > 1.07).astype(int)
        
        smiles_test = df_test['SMILES']
        her_values_test = df_test['HER (µmol/h)'].fillna(0)
        labels_test = (her_values_test > 1.07).astype(int)
        
        smiles_list = pd.concat([smiles_train, smiles_test], ignore_index=True)
        labels = pd.concat([labels_train, labels_test], ignore_index=True)
        train_size = len(smiles_train)
    else:
        df = pd.read_csv(os.path.join(data_dir, f"{data_file}.csv"))
        smiles_list = df['smiles']
        labels = df['label']
        train_size = None
    
    # Convert SMILES to graphs
    data_list = []
    original_indices = []
    failed_smiles = []
    
    for idx in range(len(smiles_list)):
        if idx % 1000 == 0 and idx > 0:
            print(f"Processed {idx} molecules...")
        
        smiles = smiles_list.iloc[idx]
        
        # Try primary conversion
        graph = path_complex_mol(smiles, encoder_atom, encoder_bond)
        
        # Try alternative methods if primary fails
        if graph is None:
            graph, recovery_method = try_alternative_smiles_conversion(
                smiles, encoder_atom, encoder_bond
            )
        
        # Check graph validity
        if graph is None or has_zero_in_degree_nodes(graph):
            failed_smiles.append((idx, smiles))
            continue
        
        label = torch.tensor(labels.iloc[idx], dtype=torch.float32)
        data_list.append([smiles, label, graph])
        
        if data_file == 'her':
            original_indices.append(idx)
    
    print(f"Successfully converted {len(data_list)}/{len(smiles_list)} molecules")
    
    # Split dataset
    if data_file == 'her' and train_size is not None:
        # 1. Recover the fixed split from Li et al. (First 572 = Train, Last 96 = Test)
        train_data_list = []
        test_data_list = []
        
        # Using original indices logic
        for idx, orig_idx in enumerate(original_indices):
            if orig_idx < train_size:
                train_data_list.append(data_list[idx])
            else:
                test_data_list.append(data_list[idx])
        
        print(f"Fixed Split Recovered: Train N={len(train_data_list)} | Test N={len(test_data_list)}")

        # 2. Setup Stratified K-Fold on the TRAINING set ONLY
        # Extract labels for stratification
        train_labels_arr = np.array([item[1].item() for item in train_data_list])
        train_indices_arr = np.arange(len(train_data_list))
        
        k_folds = 10
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        folds_data = []
        print(f'\nGenerating {k_folds}-Fold CV on Training Set...')
        
        for fold_idx, (t_idx, v_idx) in enumerate(skf.split(train_indices_arr, train_labels_arr)):
            # t_idx: Indices for training this fold (90% of 572)
            # v_idx: Indices for validation this fold (10% of 572)
            
            fold_train = [train_data_list[i] for i in t_idx]
            fold_valid = [train_data_list[i] for i in v_idx]
            
            folds_data.append({
                'fold_id': fold_idx,
                'train_label': [x[1] for x in fold_train],
                'train_graph_list': [x[2] for x in fold_train],
                'valid_label': [x[1] for x in fold_valid],
                'valid_graph_list': [x[2] for x in fold_valid],
                # CRITICAL: The Test set is the FIXED 96 samples for every fold
                'test_label': [x[1] for x in test_data_list],
                'test_graph_list': [x[2] for x in test_data_list]
            })
            print(f"  Fold {fold_idx+1}: Train {len(fold_train)} | Val {len(fold_valid)} | Fixed Test {len(test_data_list)}")

        # Save in the new format
        torch.save({
            'folds_data': folds_data,
            'k_folds': k_folds,
            'batch_size': batch_size,
            'shuffle': True,
        }, processed_file)
        
        print(f"10-Fold CV dataset saved to {processed_file}")
        return
    
    # For non-HER datasets, use standard splitting
    splitter = ScaffoldSplitter()
    train_data, valid_data, test_data = splitter.split(
        data_list, frac_train=train_ratio, frac_valid=val_ratio, frac_test=test_ratio
    )
    
    # Extract labels and graphs
    train_labels = [d[1] for d in train_data]
    train_graphs = [d[2] for d in train_data]
    
    valid_labels = [d[1] for d in valid_data] if valid_data else []
    valid_graphs = [d[2] for d in valid_data] if valid_data else []
    
    test_labels = [d[1] for d in test_data]
    test_graphs = [d[2] for d in test_data]
    
    # Save processed data
    torch.save({
        'train_label': train_labels,
        'train_graph_list': train_graphs,
        'valid_label': valid_labels,
        'valid_graph_list': valid_graphs,
        'test_label': test_labels,
        'test_graph_list': test_graphs,
        'batch_size': batch_size,
        'shuffle': True,
    }, processed_file)
    
    print(f"Dataset saved to {processed_file}")


def train_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epoch: int
) -> Tuple[float, float]:
    """
    Train model for one epoch.
    
    Args:
        model: Fourier-KAGAT model
        device: Computing device
        train_loader: Training data loader
        valid_loader: Validation data loader
        optimizer: Optimizer
        loss_fn: Loss function
        epoch: Current epoch number
        
    Returns:
        Tuple of (train_loss, valid_loss)
    """
    model.train()
    total_train_loss = 0.0
    
    for batch_idx, (labels, graphs) in enumerate(train_loader):
        optimizer.zero_grad()
        
        labels = labels.to(device)
        graphs = graphs.to(device)
        node_features = graphs.ndata['feat']
        edge_features = graphs.edata['feat']
        
        output = model(graphs, node_features, edge_features)
        
        # Handle output shape
        if output.dim() > 1 and output.size(1) == 1:
            output = output.squeeze(1)
        if labels.dim() > 1 and labels.size(1) == 1:
            labels = labels.squeeze(1)
        
        # Mask for valid labels (handle -1 as missing)
        mask = (labels != -1).to(dtype=output.dtype)
        labels_clean = torch.where(labels == -1, torch.zeros_like(labels), labels)
        
        loss_elem = loss_fn(output, labels_clean)
        loss = (loss_elem * mask).sum() / mask.sum().clamp_min(1.0)
        
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
    
    # Validation
    model.eval()
    total_valid_loss = 0.0
    num_valid_batches = 0
    
    with torch.no_grad():
        for labels, graphs in valid_loader:
            num_valid_batches += 1
            labels = labels.to(device)
            graphs = graphs.to(device)
            node_features = graphs.ndata['feat']
            edge_features = graphs.edata['feat']
            
            output = model(graphs, node_features, edge_features)
            
            if output.dim() > 1 and output.size(1) == 1:
                output = output.squeeze(1)
            if labels.dim() > 1 and labels.size(1) == 1:
                labels = labels.squeeze(1)
            
            mask = (labels != -1).to(dtype=output.dtype)
            labels_clean = torch.where(labels == -1, torch.zeros_like(labels), labels)
            
            loss_elem = loss_fn(output, labels_clean)
            valid_loss = (loss_elem * mask).sum() / mask.sum().clamp_min(1.0)
            total_valid_loss += valid_loss.item()
    
    avg_valid_loss = total_valid_loss / num_valid_batches if num_valid_batches > 0 else 0.0
    
    print(f"Epoch {epoch} | Train Loss: {total_train_loss:.4f} | "
          f"Valid Loss: {avg_valid_loss:.4f}")
    
    return total_train_loss, avg_valid_loss


def evaluate(
    model: nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    return_predictions: bool = False
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Evaluate model on dataset.
    
    Args:
        model: Trained model
        device: Computing device
        data_loader: Data loader
        return_predictions: Whether to return predictions and labels
        
    Returns:
        AUC score, and optionally predictions and labels
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for labels, graphs in data_loader:
            graphs = graphs.to(device)
            node_features = graphs.ndata['feat'].to(device)
            edge_features = graphs.edata['feat'].to(device)
            
            output = model(graphs, node_features, edge_features).cpu()
            
            if output.dim() > 1 and output.size(1) == 1:
                output = output.squeeze(1)
            
            if labels.dim() > 1:
                labels = labels.squeeze(1) if labels.size(1) == 1 else labels.squeeze(0)
            
            # Handle missing labels
            labels_clean = torch.where(labels == -1, torch.zeros_like(labels), labels)
            
            all_predictions.append(output.numpy())
            all_labels.append(labels_clean.numpy())
    
    predictions = np.concatenate(all_predictions).flatten()
    labels = np.concatenate(all_labels).flatten()
    
    auc = roc_auc_score(labels, predictions)
    
    if return_predictions:
        return auc, predictions, labels
    else:
        return auc, None, None


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments and load configuration.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Train Fourier-KAGAT model for binary classification"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/gat_path.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Load configuration from YAML
    if os.path.exists(args.config):
        with open(args.config, "r") as config_file:
            config = yaml.safe_load(config_file)
        for key, value in config.items():
            setattr(args, key, value)
    
    return args


def main() -> None:
    """Main training function."""
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parse arguments
    args = parse_arguments()
    
    # Extract parameters
    dataset_name = args.select_dataset
    batch_size = args.batch_size
    train_ratio = args.train_ratio
    val_ratio = args.vali_ratio
    test_ratio = args.test_ratio
    
    # Dataset-specific target dimensions
    target_dims = {
        'tox21': 12, 'muv': 17, 'sider': 27, 'clintox': 2,
        'bace': 1, 'bbbp': 1, 'hiv': 1, 'her': 1
    }
    target_dim = target_dims.get(dataset_name, 1)
    
    encoder_atom = args.encoder_atom
    encoder_bond = args.encoder_bond
    encode_dim = [92, 21]  # CGCNN: 92, dim_14: 21
    
    # Create dataset
    create_dataset(
        dataset_name, encoder_atom, encoder_bond,
        batch_size, train_ratio, val_ratio, test_ratio
    )
    
    # Load processed data - handle 10-fold CV for HER dataset
    if dataset_name == 'her':
        processed_file = f"data/processed/{dataset_name}_10fold.pth"
        state = torch.load(processed_file, map_location=device)
        folds_data = state['folds_data']
        k_folds = state['k_folds']
        print(f"Loaded {k_folds}-fold Cross-Validation Data with Fixed Test Set.")
    else:
        # Fallback for other datasets - treat as 1 fold
        processed_file = f"data/processed/{dataset_name}.pth"
        state = torch.load(processed_file, map_location=device)
        folds_data = [{
            'fold_id': 0,
            'train_label': state['train_label'],
            'train_graph_list': state['train_graph_list'],
            'valid_label': state['valid_label'],
            'valid_graph_list': state['valid_graph_list'],
            'test_label': state['test_label'],
            'test_graph_list': state['test_graph_list']
        }]
        k_folds = 1
    
    # Training parameters
    num_heads = args.head
    num_layers = args.num_layers
    learning_rate = args.LR
    num_epochs = args.NUM_EPOCHS
    grid_size = args.grid
    pooling = args.pooling
    loss_type = args.loss_sclect
    
    # Set random seed
    set_seed(42)
    
    fold_test_results = []  # Store AUC for each fold on fixed test set
    
    # --- START FOLD LOOP ---
    for fold_idx in range(k_folds):
        current_fold = folds_data[fold_idx]
        print(f"\n{'='*60}")
        print(f"Running Fold {fold_idx+1}/{k_folds}")
        print(f"{'='*60}")
        
        # Load Datasets for this Fold
        loaded_train_dataset = MolecularDataset(
            current_fold['train_label'], current_fold['train_graph_list']
        )
        loaded_valid_dataset = MolecularDataset(
            current_fold['valid_label'], current_fold['valid_graph_list']
        )
        loaded_test_dataset = MolecularDataset(
            current_fold['test_label'], current_fold['test_graph_list']
        )
        
        print(f"Train samples: {len(loaded_train_dataset)}")
        print(f"Valid samples: {len(loaded_valid_dataset)}")
        print(f"Test samples: {len(loaded_test_dataset)}")
        
        # Create Loaders
        loaded_train_loader = DataLoader(
            loaded_train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=False, drop_last=True, collate_fn=collate_batch
        )
        loaded_valid_loader = DataLoader(
            loaded_valid_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=False, drop_last=False, collate_fn=collate_batch
        )
        loaded_test_loader = DataLoader(
            loaded_test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=False, drop_last=False, collate_fn=collate_batch
        )
        
        # Re-Initialize Model (Fresh weights for every fold!)
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
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        model = model.to(device)
        
        # Setup loss function
        if loss_type == 'l1':
            loss_fn = nn.L1Loss(reduction='sum')
        elif loss_type == 'l2':
            loss_fn = nn.MSELoss(reduction='none')
        elif loss_type == 'sml1':
            loss_fn = nn.SmoothL1Loss(reduction='sum')
        elif loss_type == 'bce':
            loss_fn = nn.BCELoss(reduction='mean')
        else:
            loss_fn = nn.BCELoss(reduction='mean')
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        
        # Train this Fold
        best_val_loss = float('inf')
        best_fold_model = None
        
        for epoch in range(1, num_epochs + 1):
            train_loss, valid_loss = train_epoch(
                model, device, loaded_train_loader, loaded_valid_loader,
                optimizer, loss_fn, epoch
            )
            
            # Save best model based on VALIDATION loss (standard CV practice)
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_fold_model = model.state_dict().copy()
                print(f"New best validation loss: {best_val_loss:.5f}")
            
            scheduler.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Best Val Loss = {best_val_loss:.5f}")
        
        # Test this Fold's Best Model on Fixed Test Set
        if best_fold_model is not None:
            model.load_state_dict(best_fold_model)
        
        fold_auc, _, _ = evaluate(model, device, loaded_test_loader)
        fold_test_results.append(fold_auc)
        print(f"Fold {fold_idx+1} Result -> Test AUC: {fold_auc:.5f}")
    
    # --- END FOLD LOOP ---
    
    # Print final statistics
    print("\n" + "="*60)
    print(f"FINAL RESULTS ({k_folds}-Fold CV Training -> Fixed Test Set Evaluation)")
    print("="*60)
    
    mean_auc = np.mean(fold_test_results)
    std_auc = np.std(fold_test_results)
    
    print(f"Mean AUC: {mean_auc:.5f}")
    print(f"Std  AUC: {std_auc:.5f}")
    print(f"Individual Fold Results: {[f'{auc:.5f}' for auc in fold_test_results]}")
    print("="*60)
    
    # Save the last fold's model (or you could save all 10 models)
    if k_folds > 0:
        torch.save(model.state_dict(), 'model_classification.pth')
        print(f"\nModel from fold {k_folds} saved to model_classification.pth")


if __name__ == '__main__':
    main()

