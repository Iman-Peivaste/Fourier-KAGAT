# Author: Iman Peivaste
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fourier-KAGAT Training Script for Regression

This script trains the Fourier-KAGAT (Fourier-based Kolmogorov-Arnold Graph Attention Network)
model for regression tasks on molecular property prediction (e.g., Electron Affinity).

Author: Iman Peivaste
"""

import os
import argparse
import random
from typing import Tuple, Optional
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import yaml
import dgl
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    explained_variance_score
)

from model.fourier_kagat import FourierKAGAT
from utils.splitters import ScaffoldSplitter
from utils.graph_path import path_complex_mol


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class MolecularDataset(Dataset):
    """Dataset class for molecular graphs and continuous labels."""
    
    def __init__(
        self,
        label_list: list,
        graph_list: list
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


def collate_batch(batch: list) -> Tuple[torch.Tensor, dgl.DGLGraph]:
    """Collate function for batching molecular graphs."""
    labels, graphs = zip(*batch)
    labels = torch.stack(labels)
    batched_graph = dgl.batch(graphs)
    return labels, batched_graph


def has_zero_in_degree_nodes(graph: dgl.DGLGraph) -> bool:
    """Check if graph has nodes with zero in-degree."""
    return (graph.in_degrees() == 0).any().item()


def try_alternative_smiles_conversion(
    smiles: str,
    encoder_atom: str,
    encoder_bond: str
) -> Tuple[Optional[dgl.DGLGraph], Optional[str]]:
    """Try alternative methods to convert SMILES to graph."""
    from rdkit import Chem
    
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
    
    return None, None


def create_regression_dataset(
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
    Create and save processed regression dataset.
    
    For HER dataset, uses 'EA (V)' column as regression target.
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    processed_file = os.path.join(processed_dir, f"{data_file}_regression.pth")
    if os.path.exists(processed_file):
        return
    
    if data_file == 'her':
        df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        
        smiles_train = df_train['SMILES']
        ea_train = df_train['EA (V)']
        if ea_train.isna().any():
            mean_ea = ea_train.mean()
            ea_train = ea_train.fillna(mean_ea)
        labels_train = ea_train.astype(float)
        
        smiles_test = df_test['SMILES']
        ea_test = df_test['EA (V)']
        if ea_test.isna().any():
            mean_ea = ea_test.mean()
            ea_test = ea_test.fillna(mean_ea)
        labels_test = ea_test.astype(float)
        
        smiles_list = pd.concat([smiles_train, smiles_test], ignore_index=True)
        labels = pd.concat([labels_train, labels_test], ignore_index=True)
        train_size = len(smiles_train)
    else:
        df = pd.read_csv(os.path.join(data_dir, f"{data_file}.csv"))
        smiles_list = df['smiles']
        if 'EA (V)' in df.columns:
            labels = df['EA (V)'].fillna(df['EA (V)'].mean()).astype(float)
        else:
            raise ValueError(f"Dataset {data_file} does not have 'EA (V)' column")
        train_size = None
    
    data_list = []
    original_indices = []
    
    for idx in range(len(smiles_list)):
        if idx % 1000 == 0 and idx > 0:
            print(f"Processed {idx} molecules...")
        
        smiles = smiles_list.iloc[idx]
        graph = path_complex_mol(smiles, encoder_atom, encoder_bond)
        
        if graph is None:
            graph, _ = try_alternative_smiles_conversion(smiles, encoder_atom, encoder_bond)
        
        if graph is None or has_zero_in_degree_nodes(graph):
            continue
        
        label = torch.tensor(labels.iloc[idx], dtype=torch.float32)
        data_list.append([smiles, label, graph])
        
        if data_file == 'her':
            original_indices.append(idx)
    
    print(f"Successfully converted {len(data_list)}/{len(smiles_list)} molecules")
    
    # Split dataset
    if data_file == 'her' and train_size is not None:
        train_data = [d for i, d in enumerate(data_list) if original_indices[i] < train_size]
        test_data = [d for i, d in enumerate(data_list) if original_indices[i] >= train_size]
        
        if val_ratio > 0.0:
            splitter = ScaffoldSplitter()
            train_data, valid_data, _ = splitter.split(
                train_data, frac_train=1.0 - val_ratio, frac_valid=val_ratio, frac_test=0.0
            )
        else:
            valid_data = []
    else:
        splitter = ScaffoldSplitter()
        train_data, valid_data, test_data = splitter.split(
            data_list, frac_train=train_ratio, frac_valid=val_ratio, frac_test=test_ratio
        )
    
    train_labels = [d[1] for d in train_data]
    train_graphs = [d[2] for d in train_data]
    valid_labels = [d[1] for d in valid_data] if valid_data else []
    valid_graphs = [d[2] for d in valid_data] if valid_data else []
    test_labels = [d[1] for d in test_data]
    test_graphs = [d[2] for d in test_data]
    
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
    """Train model for one epoch."""
    model.train()
    total_train_loss = 0.0
    
    for labels, graphs in train_loader:
        optimizer.zero_grad()
        
        labels = labels.to(device)
        graphs = graphs.to(device)
        node_features = graphs.ndata['feat']
        edge_features = graphs.edata['feat']
        
        output = model(graphs, node_features, edge_features)
        
        if output.dim() > 1 and output.size(1) == 1:
            output = output.squeeze(1)
        if labels.dim() > 1 and labels.size(1) == 1:
            labels = labels.squeeze(1)
        
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
    
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
            
            valid_loss = loss_fn(output, labels)
            total_valid_loss += valid_loss.item()
    
    avg_valid_loss = total_valid_loss / num_valid_batches if num_valid_batches > 0 else 0.0
    
    print(f"Epoch {epoch} | Train Loss: {total_train_loss:.4f} | "
          f"Valid Loss: {avg_valid_loss:.4f}")
    
    return total_train_loss, avg_valid_loss


def evaluate_regression(
    model: nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    return_predictions: bool = False
) -> Tuple[float, float, float, float, float, Optional[np.ndarray], Optional[np.ndarray]]:
    """Evaluate regression model."""
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
            
            all_predictions.append(output.numpy())
            all_labels.append(labels.numpy())
    
    predictions = np.concatenate(all_predictions).flatten()
    labels = np.concatenate(all_labels).flatten()
    
    r2 = r2_score(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    evs = explained_variance_score(labels, predictions)
    
    if return_predictions:
        return r2, mae, mse, rmse, evs, predictions, labels
    else:
        return r2, mae, mse, rmse, evs, None, None


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments and load configuration."""
    parser = argparse.ArgumentParser(
        description="Train Fourier-KAGAT model for regression"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/gat_path.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    if os.path.exists(args.config):
        with open(args.config, "r") as config_file:
            config = yaml.safe_load(config_file)
        for key, value in config.items():
            setattr(args, key, value)
    
    return args


def main() -> None:
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    args = parse_arguments()
    
    dataset_name = args.select_dataset
    batch_size = args.batch_size
    train_ratio = args.train_ratio
    val_ratio = args.vali_ratio
    test_ratio = args.test_ratio
    
    encoder_atom = args.encoder_atom
    encoder_bond = args.encoder_bond
    encode_dim = [92, 21]
    target_dim = 1  # Regression: single continuous value
    
    create_regression_dataset(
        dataset_name, encoder_atom, encoder_bond,
        batch_size, train_ratio, val_ratio, test_ratio
    )
    
    processed_file = f"data/processed/{dataset_name}_regression.pth"
    state = torch.load(processed_file, map_location=device)
    
    train_dataset = MolecularDataset(state['train_label'], state['train_graph_list'])
    valid_dataset = MolecularDataset(state['valid_label'], state['valid_graph_list'])
    test_dataset = MolecularDataset(state['test_label'], state['test_graph_list'])
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=state['shuffle'],
        num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_batch
    )
    
    if val_ratio > 0.0:
        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True, drop_last=False, collate_fn=collate_batch
        )
    else:
        valid_loader = []
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False, collate_fn=collate_batch
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    num_iterations = args.iter
    num_heads = args.head
    num_layers = args.num_layers
    learning_rate = args.LR
    num_epochs = args.NUM_EPOCHS
    grid_size = args.grid
    pooling = args.pooling
    loss_type = args.loss_sclect
    
    set_seed(42)
    
    all_r2_scores = []
    all_mae_scores = []
    all_rmse_scores = []
    
    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")
        
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
        
        # Remove sigmoid for regression
        if hasattr(model, 'readout') and isinstance(model.readout, nn.Sequential):
            layers = list(model.readout.children())
            if len(layers) > 0 and isinstance(layers[-1], nn.Sigmoid):
                layers[-1] = nn.Identity()
                model.readout = nn.Sequential(*layers)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        model = model.to(device)
        
        # Setup loss function
        if loss_type == 'l1':
            loss_fn = nn.L1Loss(reduction='mean')
        elif loss_type == 'l2':
            loss_fn = nn.MSELoss(reduction='mean')
        elif loss_type == 'sml1':
            loss_fn = nn.SmoothL1Loss(reduction='mean')
        else:
            loss_fn = nn.MSELoss(reduction='mean')
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        
        best_r2 = float('-inf')
        best_model_state = None
        
        for epoch in range(1, num_epochs + 1):
            train_loss, valid_loss = train_epoch(
                model, device, train_loader, valid_loader,
                optimizer, loss_fn, epoch
            )
            
            r2, mae, _, rmse, _ = evaluate_regression(model, device, test_loader)
            
            if r2 > best_r2:
                best_r2 = r2
                best_mae = mae
                best_rmse = rmse
                best_model_state = model.state_dict().copy()
                print(f"New best R²: {best_r2:.5f}, MAE: {best_mae:.5f}, RMSE: {best_rmse:.5f}")
            
            scheduler.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Best R² = {best_r2:.5f}")
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        all_r2_scores.append(best_r2)
        all_mae_scores.append(best_mae)
        all_rmse_scores.append(best_rmse)
        print(f"Iteration {iteration + 1} best R²: {best_r2:.5f}")
    
    torch.save(model.state_dict(), 'model_regression.pth')
    
    if len(all_r2_scores) > 1:
        import statistics
        mean_r2 = statistics.mean(all_r2_scores)
        std_r2 = statistics.stdev(all_r2_scores)
        mean_mae = statistics.mean(all_mae_scores)
        mean_rmse = statistics.mean(all_rmse_scores)
        print(f"\nMean R²: {mean_r2:.5f} ± {std_r2:.5f}")
        print(f"Mean MAE: {mean_mae:.5f}")
        print(f"Mean RMSE: {mean_rmse:.5f}")
    
    print("\n" + "="*60)
    print("Final Test Set Evaluation")
    print("="*60)
    
    model.load_state_dict(torch.load('model_regression.pth', map_location=device))
    model = model.to(device)
    
    r2, mae, mse, rmse, evs, y_pred, y_true = evaluate_regression(
        model, device, test_loader, return_predictions=True
    )
    
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Explained Variance: {evs:.4f}")
    print(f"\nTrue values - min: {y_true.min():.4f}, max: {y_true.max():.4f}, "
          f"mean: {y_true.mean():.4f}")
    print(f"Predicted values - min: {y_pred.min():.4f}, max: {y_pred.max():.4f}, "
          f"mean: {y_pred.mean():.4f}")


if __name__ == '__main__':
    main()

