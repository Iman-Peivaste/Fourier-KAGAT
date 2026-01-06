#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fourier-KAGAT Model

This module implements the Fourier-KAGAT architecture, which combines Graph Attention Networks
with Fourier-based Kolmogorov-Arnold Network layers for molecular property prediction.

Author: Iman Peivaste
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np

from dgl.nn import SumPooling, AvgPooling, MaxPooling
from dgl.nn.functional import edge_softmax


class KANLinear(nn.Module):
    """
    Kolmogorov-Arnold Network linear layer using Fourier basis functions.
    
    This layer implements a learnable function using Fourier series expansion,
    allowing for more flexible non-linear transformations compared to standard
    linear layers.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        grid_size: Number of Fourier basis functions
        add_bias: Whether to include bias term
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_size: int,
        add_bias: bool = False
    ) -> None:
        super(KANLinear, self).__init__()
        self.grid_size = grid_size
        self.add_bias = add_bias
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize Fourier coefficients with proper scaling
        scale_factor = np.sqrt(input_dim) * np.sqrt(self.grid_size)
        self.fourier_coeffs = nn.Parameter(
            torch.randn(2, output_dim, input_dim, grid_size) / scale_factor
        )
        
        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through KAN linear layer.
        
        Args:
            x: Input tensor of shape (..., input_dim)
            
        Returns:
            Output tensor of shape (..., output_dim)
        """
        original_shape = x.shape
        output_shape = original_shape[:-1] + (self.output_dim,)
        x = x.view(-1, self.input_dim)
        
        # Generate Fourier basis: cos(k*x) and sin(k*x) for k=1 to grid_size
        k_values = torch.arange(
            1, self.grid_size + 1, device=x.device
        ).reshape(1, 1, 1, self.grid_size)
        x_reshaped = x.view(x.shape[0], 1, x.shape[1], 1)
        
        # Compute cosine and sine components
        cos_components = torch.cos(k_values * x_reshaped)
        sin_components = torch.sin(k_values * x_reshaped)
        
        cos_components = cos_components.reshape(1, x.shape[0], x.shape[1], self.grid_size)
        sin_components = sin_components.reshape(1, x.shape[0], x.shape[1], self.grid_size)
        
        # Combine cosine and sine, then apply Fourier coefficients
        basis_functions = torch.concat([cos_components, sin_components], axis=0)
        y = torch.einsum("dbik,djik->bj", basis_functions, self.fourier_coeffs)
        
        if self.add_bias:
            y += self.bias
        
        return y.view(output_shape)


class GATKANLayer(nn.Module):
    """
    Graph Attention Network layer with KAN components.
    
    This layer performs graph convolution using attention mechanism, where
    both node and edge features are transformed through KAN layers.
    
    Args:
        in_node_feats: Input node feature dimension
        in_edge_feats: Input edge feature dimension
        out_node_feats: Output node feature dimension
        out_edge_feats: Output edge feature dimension
        num_heads: Number of attention heads
        grid_size: KAN grid size for Fourier basis
        bias: Whether to use bias in edge transformation
    """
    
    def __init__(
        self,
        in_node_feats: int,
        in_edge_feats: int,
        out_node_feats: int,
        out_edge_feats: int,
        num_heads: int,
        grid_size: int,
        bias: bool = True
    ) -> None:
        super(GATKANLayer, self).__init__()
        self.num_heads = num_heads
        self.out_node_feats = out_node_feats
        self.out_edge_feats = out_edge_feats
        
        # Linear transformations for node and edge features
        self.fc_node = nn.Linear(
            in_node_feats + in_edge_feats,
            out_node_feats * num_heads,
            bias=True
        )
        self.fc_ni = nn.Linear(
            in_node_feats,
            out_edge_feats * num_heads,
            bias=False
        )
        self.fc_fij = nn.Linear(
            in_edge_feats,
            out_edge_feats * num_heads,
            bias=False
        )
        self.fc_nj = nn.Linear(
            in_node_feats,
            out_edge_feats * num_heads,
            bias=False
        )
        
        # Attention parameter
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_edge_feats)))
        
        # KAN layers for output transformation
        self.output_node = KANLinear(
            out_node_feats, out_node_feats, grid_size, add_bias=True
        )
        self.output_edge = KANLinear(
            out_edge_feats, out_edge_feats, grid_size, add_bias=True
        )
        self.edge_kan = KANLinear(
            out_edge_feats * num_heads,
            out_edge_feats * num_heads,
            gridsize=1,
            add_bias=True
        )
        self.node_kan = KANLinear(
            in_node_feats + in_edge_feats,
            in_node_feats + in_edge_feats,
            gridsize=1,
            add_bias=True
        )

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_edge_feats,)))
        else:
            self.register_buffer('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize model parameters using Xavier normal initialization."""
        nn.init.xavier_normal_(self.fc_node.weight)
        nn.init.xavier_normal_(self.fc_ni.weight)
        nn.init.xavier_normal_(self.fc_fij.weight)
        nn.init.xavier_normal_(self.fc_nj.weight)
        nn.init.xavier_normal_(self.attn)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def message_func(self, edges: dgl.EdgeBatch) -> dict:
        """Message function for graph message passing."""
        return {'feat': edges.data['feat']}

    def reduce_func(self, nodes: dgl.NodeBatch) -> dict:
        """
        Reduce function that aggregates messages from neighbors.
        
        Computes average of incoming edge features.
        """
        num_edges = nodes.mailbox['feat'].size(1)
        aggregated_feats = torch.sum(nodes.mailbox['feat'], dim=1) / num_edges
        return {'agg_feats': aggregated_feats}

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        get_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through GAT-KAN layer.
        
        Args:
            graph: DGL graph object
            node_feats: Node features tensor
            edge_feats: Edge features tensor
            get_attention: Whether to return attention weights
            
        Returns:
            Updated node features, edge features, and optionally attention weights
        """
        with graph.local_scope():
            graph.ndata['feat'] = node_feats
            graph.edata['feat'] = edge_feats
            
            # Compute edge features from node and edge information
            f_ni = self.fc_ni(node_feats)  # Node i contribution
            f_nj = self.fc_nj(node_feats)  # Node j contribution
            f_fij = self.fc_fij(edge_feats)  # Edge contribution

            graph.srcdata.update({'f_ni': f_ni})
            graph.dstdata.update({'f_nj': f_nj})
            graph.apply_edges(fn.u_add_v('f_ni', 'f_nj', 'f_tmp'))
            
            # Combine node and edge features, then transform with KAN
            f_out = graph.edata.pop('f_tmp') + f_fij
            f_out = self.edge_kan(f_out)

            if self.bias is not None:
                f_out = f_out + self.bias
            f_out = F.leaky_relu(f_out)
            f_out = f_out.view(-1, self.num_heads, self.out_edge_feats)
            
            # Compute attention scores
            attention_scores = (f_out * self.attn).sum(dim=-1).unsqueeze(-1)

            # Aggregate edge features to nodes
            graph.send_and_recv(
                graph.edges(),
                self.message_func,
                reduce_func=self.reduce_func
            )
            merged_feats = torch.cat(
                (graph.ndata['feat'], graph.ndata['agg_feats']),
                dim=1
            )
            merged_feats = self.node_kan(merged_feats)

            # Apply attention weights
            graph.edata['a'] = edge_softmax(graph, attention_scores)
            graph.ndata['h_out'] = self.fc_node(merged_feats).view(
                -1, self.num_heads, self.out_node_feats
            )
            
            graph.update_all(fn.u_mul_e('h_out', 'a', 'm'), fn.sum('m', 'h_out'))

            h_out = F.leaky_relu(graph.ndata['h_out'])
            h_out = h_out.view(-1, self.num_heads, self.out_node_feats)

            # Aggregate across attention heads
            h_out = torch.sum(h_out, dim=1)
            f_out = torch.sum(f_out, dim=1)

            # Final KAN transformations
            out_node = self.output_node(h_out)
            out_edge = self.output_edge(f_out)
            
            if get_attention:
                return out_node, out_edge, graph.edata.pop('a')
            else:
                return out_node, out_edge


class FourierKAGAT(nn.Module):
    """
    Fourier-KAGAT: Fourier-based Kolmogorov-Arnold Graph Attention Network.
    
    Main model class that stacks multiple GAT-KAN layers with Fourier basis functions
    for molecular property prediction. Supports both classification and regression tasks.
    
    Args:
        in_node_dim: Input node feature dimension
        in_edge_dim: Input edge feature dimension
        hidden_dim: Hidden layer dimension
        out_1: First output layer dimension
        out_2: Final output dimension (number of tasks)
        grid_size: KAN grid size for Fourier basis
        head: Number of attention heads
        layer_num: Number of GAT-KAN layers
        pooling: Pooling method ('avg', 'max', or 'sum')
    """
    
    def __init__(
        self,
        in_node_dim: int,
        in_edge_dim: int,
        hidden_dim: int,
        out_1: int,
        out_2: int,
        grid_size: int,
        head: int,
        layer_num: int,
        pooling: str
    ) -> None:
        super(FourierKAGAT, self).__init__()
        self.in_node_dim = in_node_dim
        self.in_edge_dim = in_edge_dim
        self.hidden_dim = hidden_dim
        self.out_1 = out_1
        self.out_2 = out_2
        self.head = head
        self.layer_num = layer_num
        self.grid_size = grid_size
        self.pooling = pooling

        # Build GAT-KAN layers
        self.attentions = nn.ModuleList()
        
        # First layer: input dimensions
        self.attentions.append(
            GATKANLayer(
                in_node_feats=in_node_dim,
                in_edge_feats=in_edge_dim,
                out_node_feats=hidden_dim,
                out_edge_feats=hidden_dim,
                num_heads=self.head,
                grid_size=self.grid_size
            )
        )
        
        # Subsequent layers: hidden dimensions
        for _ in range(self.layer_num - 1):
            self.attentions.append(
                GATKANLayer(
                    in_node_feats=hidden_dim,
                    in_edge_feats=hidden_dim,
                    out_node_feats=hidden_dim,
                    out_edge_feats=hidden_dim,
                    num_heads=self.head,
                    grid_size=self.grid_size
                )
            )

        # Pooling layers
        self.leaky_relu = nn.LeakyReLU()
        self.sum_pool = SumPooling()
        self.avg_pool = AvgPooling()
        self.max_pool = MaxPooling()

        # Output layers with KAN transformations
        output_layers = [
            KANLinear(hidden_dim, out_1, grid_size, add_bias=False),
            self.leaky_relu,
            KANLinear(out_1, out_2, grid_size, add_bias=True),
            nn.Sigmoid()  # For classification; can be removed for regression
        ]
        self.readout = nn.Sequential(*output_layers)

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_feature: torch.Tensor,
        edge_feature: torch.Tensor,
        get_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass through Fourier-KAGAT model.
        
        Args:
            graph: DGL graph object (can be batched)
            node_feature: Node features tensor
            edge_feature: Edge features tensor
            get_attention: Whether to return attention weights for interpretation
            
        Returns:
            Model predictions, and optionally attention weights from all layers
        """
        all_attention_weights = []
        
        # Pass through all GAT-KAN layers
        for layer in self.attentions:
            if get_attention:
                node_feature, edge_feature, attention_weights = layer(
                    graph, node_feature, edge_feature, get_attention=True
                )
                all_attention_weights.append(attention_weights)
            else:
                node_feature, edge_feature = layer(
                    graph, node_feature, edge_feature, get_attention=False
                )
        
        # Apply activation
        node_feature = F.leaky_relu(node_feature)

        # Graph-level pooling
        if self.pooling == 'avg':
            graph_repr = self.avg_pool(graph, node_feature)
        elif self.pooling == 'max':
            graph_repr = self.max_pool(graph, node_feature)
        elif self.pooling == 'sum':
            graph_repr = self.sum_pool(graph, node_feature)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # Final prediction
        output = self.readout(graph_repr)
        
        if get_attention:
            return output, all_attention_weights
        else:
            return output

