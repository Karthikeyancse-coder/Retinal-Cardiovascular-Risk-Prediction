# RetiCardNet — Retinal Cardiovascular Risk Prediction Network

A state-of-the-art multi-modal deep learning framework for predicting cardiovascular disease risk from retinal fundus images.

## 🎯 Overview

RetiCardNet combines three powerful modalities to predict cardiovascular risk:
1. **Vision Transformer (ViT)** - Analyzes global retinal image features
2. **Graph Neural Network (GNN)** - Learns vessel topology and structure
3. **Clinical Features** - Integrates patient metadata (Age, BP, BMI)

The model uses a novel **Cross-Attention Fusion** mechanism to intelligently combine these modalities for superior prediction accuracy.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      RetiCardNet                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Fundus Image │  │ Vessel Graph │  │   Clinical   │     │
│  │  (224×224)   │  │  (500 nodes) │  │  (Age,BP,BMI)│     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                  │             │
│         ▼                 ▼                  ▼             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │     ViT      │  │  GNN (GCN)   │  │     MLP      │     │
│  │  (vit_b_16)  │  │  3 Layers    │  │  3 Layers    │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                  │             │
│         └─────────┬───────┴──────────────────┘             │
│                   ▼                                        │
│         ┌─────────────────────┐                            │
│         │  Cross-Attention    │                            │
│         │  Fusion (4 heads)   │                            │
│         └─────────┬───────────┘                            │
│                   ▼                                        │
│         ┌─────────────────────┐                            │
│         │  Prediction Head    │                            │
│         │  (Low/Mod/High)     │                            │
│         └─────────────────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Dataset

- **Source**: Kaggle Fundus Dataset (APTOS, DDR, IDRiD, EyePACs, Messidor)
- **Total Images**: 3,660 retinal fundus images
- **Classes**: 3 cardiovascular risk levels (Low, Moderate, High)
- **Split**: 70% train / 15% val / 15% test
- **Clinical Features**: Simulated Age, Systolic BP, BMI correlated with risk

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install torch torchvision torch_geometric
pip install opencv-python scikit-image scipy pandas numpy
pip install scikit-learn tqdm
```

### Data Preparation

```bash
# Generate clinical data and split dataset
python data_setup.py
```

This creates `clinical_data.csv` with:
- Image paths
- DR grades (0-4)
- CV risk labels (0-2)
- Simulated clinical features

### Training

```bash
# Train the model
python train.py --epochs 20 --batch_size 8 --lr 0.0001
```

**Arguments:**
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 1e-4)
- `--csv_file`: Path to clinical data CSV

### Evaluation

```bash
# Evaluate on test set
python evaluate.py --checkpoint best_reticardnet.pth
```

**Metrics Computed:**
- Accuracy
- F1-Score
- Precision & Recall
- ROC-AUC
- Confusion Matrix

## 📁 Project Structure

```
e:\HD_Model\Antigravity\
├── data_setup.py              # Dataset preparation
├── dataset.py                 # PyTorch Dataset with graph extraction
├── model_components.py        # ViT, GNN, MLP, Fusion modules
├── reticardnet.py            # Main model architecture
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── verify_pipeline.py        # Pipeline verification
├── clinical_data.csv         # Generated clinical data
├── best_reticardnet.pth      # Best model checkpoint
└── dataset/                  # Fundus images
    └── split_dataset/
        └── test/
            ├── 0/            # DR grade 0 (Low risk)
            ├── 1/            # DR grade 1 (Moderate risk)
            ├── 2/            # DR grade 2 (Moderate risk)
            ├── 3/            # DR grade 3 (High risk)
            └── 4/            # DR grade 4 (High risk)
```

## 🔬 Technical Details

### Vessel Graph Construction

1. **Preprocessing**: CLAHE enhancement on green channel
2. **Segmentation**: Adaptive thresholding
3. **Skeletonization**: Morphological thinning
4. **Graph Building**: k-NN graph (k=5) using scipy's cKDTree
5. **Downsampling**: Max 500 nodes per graph for efficiency

### Multi-Modal Fusion

The Cross-Attention layer performs:
```python
# Stack modalities
features = [img_emb, graph_emb, clinical_emb]  # Each: (B, 128)
stacked = stack(features, dim=1)                # (B, 3, 128)

# Multi-head attention
attended = MultiHeadAttention(stacked)          # (B, 3, 128)

# Residual + Norm + Pool
output = mean(LayerNorm(stacked + attended))    # (B, 128)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Loss Function | CrossEntropyLoss |
| Batch Size | 8 |
| Epochs | 20 |

## 🎯 Performance Target

- **Target Accuracy**: ≥90%
- **Current Status**: Training in progress
- **Expected**: High accuracy due to multi-modal fusion

## 🔑 Key Features

✅ **Multi-Modal Learning**: Combines image, graph, and clinical data  
✅ **Vessel Topology**: GNN captures vascular structure  
✅ **Attention Fusion**: Dynamic modality weighting  
✅ **Pretrained ViT**: Transfer learning from ImageNet  
✅ **Clinical Integration**: Seamless fusion of numerical features  
✅ **Automated Pipeline**: End-to-end from images to predictions  

## 📚 Citation

If you use this code, please cite:

```bibtex
@software{reticardnet2025,
  title={RetiCardNet: Multi-Modal Deep Learning for Cardiovascular Risk Prediction},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/reticardnet}
}
```

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

---

**Note**: This model is for research purposes. Clinical deployment requires regulatory approval and extensive validation.
