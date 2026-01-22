# RetiCardNet — Cardiovascular Risk Prediction from Retinal Images

Multi-modal deep learning model for predicting cardiovascular disease risk from retinal fundus images.

## 🎯 Overview

RetiCardNet combines three data sources:
1. **Retinal Images** (EfficientNet-B0 CNN)
2. **Vessel Graph** (Graph Attention Network)
3. **Clinical Data** (Age, BP, BMI, HbA1c, LDL)

**Fusion**: Transformer-based cross-attention layer  
**Performance**: 92.79% accuracy on 1,151 test images

## 📊 Dataset

- **Total**: 5,906 retinal fundus images
- **Classes**: 3 (No Risk, Moderate Risk, High Risk)
- **Split**: Train (4,604) / Val (1,151) / Test (1,151)

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
python train_aggressive.py
```

### Testing
```bash
python evaluate_full_aggressive.py
```

## 📁 Project Structure

```
e:\
├── train_aggressive.py              # Main training script
├── reticardnet_aggressive.py        # Model architecture
├── dataset_aggressive.py            # Data loader
├── evaluate_full_aggressive.py      # Evaluation script
├── clinical_data_aggressive.csv     # Dataset manifest
├── best_reticardnet_aggressive.pth  # Trained weights (92.79% accuracy)
├── requirements.txt                 # Dependencies
└── dataset/                         # Images
    └── split_dataset/
        ├── train/
        ├── val/
        └── test/
```

## 🔬 Technical Details

**Model Components**:
- EfficientNet-B0 (Image features)
- GAT (Graph Attention Network for vessels)
- MLP (Clinical features)
- Transformer Fusion (Cross-attention)

**Training**:
- Optimizer: AdamW
- Loss: Focal Loss + Label Smoothing
- Scheduler: ReduceLROnPlateau
- Batch Size: 8
- Image Size: 256x256

## 📊 Performance

| Metric | Score |
|--------|-------|
| Accuracy | 92.79% |
| F1-Score | 92.77% |
| ROC-AUC | 98.28% |
| Precision | 92.87% |

**Safety**: Zero critical misses (no High Risk classified as No Risk)

## 📄 License

MIT License

---

**Note**: Research prototype. Not for clinical use without validation.
