# Fourier-KAGAT: Fourier-based Kolmogorov-Arnold Graph Attention Network for Molecular Property Prediction

This repository contains the implementation of Fourier-KAGAT, a graph neural network architecture that combines Graph Attention Networks with Fourier-based Kolmogorov-Arnold Network layers for molecular property prediction.

## Author

**Iman Peivaste**

## Overview

Fourier-KAGAT uses Fourier basis functions within Kolmogorov-Arnold Network layers integrated into a graph attention framework to learn molecular representations. The model can be used for both binary classification and regression tasks on molecular datasets.

## Features

- **Graph-based molecular representation**: Converts SMILES strings to graph structures with 3D conformers
- **KAN-enhanced attention**: Uses Kolmogorov-Arnold Networks for flexible feature transformations
- **Multi-head attention**: Captures diverse molecular patterns
- **Interpretability**: Attention visualization for understanding model decisions
- **Both classification and regression**: Supports binary classification and continuous value prediction

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd git_hub
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have RDKit and DGL properly installed. For RDKit, you may need:
```bash
conda install -c conda-forge rdkit
```

## Usage

### Training for Classification

Train a Fourier-KAGAT model for binary classification:

```bash
python train_classification.py --config config/gat_path.yaml
```

### Training for Regression

Train a Fourier-KAGAT model for regression (e.g., Electron Affinity prediction):

```bash
python train_regression.py --config config/gat_path.yaml
```

### Model Interpretation

Visualize attention weights for specific molecules:

```bash
python interpret_molecules.py \
    --model model_classification.pth \
    --smiles "O=Cc1cc(C=O)c(O)c(C=O)c1" "O=C1c2ccccc2C(=O)C1(O)O" \
    --ids "Active" "Inactive"
```

## Configuration

Edit `config/gat_path.yaml` to adjust model hyperparameters:

- `model_select`: Model type (use "kagat" for Fourier-KAGAT)
- `select_dataset`: Dataset name
- `grid`: KAN grid size
- `head`: Number of attention heads
- `num_layers`: Number of GAT-KAN layers
- `pooling`: Graph pooling method ('avg', 'max', or 'sum')
- `LR`: Learning rate
- `NUM_EPOCHS`: Number of training epochs
- `batch_size`: Batch size

## Data Format

For classification, provide CSV files with:
- `smiles`: SMILES strings
- `label`: Binary labels (0 or 1)

For regression, provide CSV files with:
- `smiles`: SMILES strings
- `EA (V)`: Continuous target values

## Model Architecture

Fourier-KAGAT consists of:
1. **Graph construction**: SMILES → molecular graph with 3D conformers
2. **GAT-KAN layers**: Multiple layers of graph attention with KAN transformations
3. **Graph pooling**: Aggregate node features to graph-level representation
4. **Output layer**: Final prediction with KAN transformations

## Citation

If you use this code in your research, please cite:

```
@article{peivaste2024fourierkagat,
  title={Fourier-KAGAT: Fourier-based Kolmogorov-Arnold Graph Attention Network for Molecular Property Prediction},
  author={Peivaste, Iman},
  journal={Journal Name},
  year={2024}
}
```

## License

[Specify your license here]

## Contact

For questions or issues, please contact Iman Peivaste.

