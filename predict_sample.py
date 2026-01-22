import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dataset_aggressive import RetiCardNetAggressiveDataset
from torch_geometric.data import Batch
import os

# Redefine TransformerFusion and Model to match the Checkpoint (Transformer-based)
class TransformerFusion(nn.Module):
    def __init__(self, dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.modality_tokens = nn.Parameter(torch.randn(1, 3, dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim*4, dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention_pool = nn.Linear(dim, 1)
        
    def forward(self, img_feat, graph_feat, clin_feat):
        x = torch.stack([img_feat, graph_feat, clin_feat], dim=1)
        x = x + self.modality_tokens
        x = self.transformer(x)
        weights = F.softmax(self.attention_pool(x), dim=1)
        fused = torch.sum(x * weights, dim=1)
        return fused

class RetiCardNetAggressive(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        from reticardnet_aggressive import EfficientNetBranch, GATBranch, ClinicalBranch
        self.img_branch = EfficientNetBranch(output_dim=256)
        self.gnn_branch = GATBranch(output_dim=256)
        self.clin_branch = ClinicalBranch(output_dim=256)
        self.fusion = TransformerFusion(dim=256) # RESTORED
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_classes)
        )
        
    def forward(self, img, graph_x, graph_edge_index, graph_batch, clin_data):
        img_feat = self.img_branch(img)
        graph_feat = self.gnn_branch(graph_x, graph_edge_index, graph_batch)
        clin_feat = self.clin_branch(clin_data)
        fused = self.fusion(img_feat, graph_feat, clin_feat)
        logits = self.classifier(fused)
        return logits

def predict_single_sample():
    # 1. Define Sample Data (High Risk Case from Test Set)
    sample_data = {
        'image_path': [r'e:\HD_Model\Antigravity\dataset\split_dataset\test\4\14287_left-600-FA.jpg'],
        'dr_grade': [4],
        'cv_risk_label': [2], # High Risk
        'age': [50.3],
        'systolic_bp': [155.4],
        'bmi': [26.6],
        'hba1c': [9.6],
        'ldl': [157.4],
        'split': ['test'] 
    }
    
    # Create Temp CSV
    # Create Temp CSV
    df = pd.DataFrame(sample_data)
    df.to_csv('temp_sample.csv', index=False)
    
    import time
    time.sleep(1.0) # Wait for file write
    
    # 2. Setup Dataset 
    try:
        dataset = RetiCardNetAggressiveDataset('temp_sample.csv', split='test')
        img, graph, clin, label = dataset[0]
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Dataset Error: {e}")
        return
    
    img = img.unsqueeze(0)
    clin = clin.unsqueeze(0)
    graph_batch = Batch.from_data_list([graph])
    
    # 3. Load Model (Using Locally Defined Class)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RetiCardNetAggressive(num_classes=3).to(device)
    
    checkpoint_path = 'best_reticardnet_aggressive.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully.")
    else:
        print("Error: Checkpoint not found.")
        return

    model.eval()
    
    # 4. Inference
    with torch.no_grad():
        img = img.to(device)
        clin = clin.to(device)
        graph_x = graph_batch.x.to(device)
        graph_edge_index = graph_batch.edge_index.to(device)
        graph_batch_idx = graph_batch.batch.to(device)
        
        logits = model(img, graph_x, graph_edge_index, graph_batch_idx, clin)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        
    # 5. Display
    classes = ['No Risk', 'Moderate Risk', 'High Risk']
    print("\n" + "="*50)
    print("PREDICTION RESULT")
    print("="*50)
    print(f"Image: 14287_left-600-FA.jpg")
    print(f"Clinical: Age: 50.3, BP: 155.4, HbA1c: 9.6")
    print("-" * 30)
    print(f"True Label:      {classes[sample_data['cv_risk_label'][0]]}")
    print(f"Predicted Class: {classes[pred_idx]}")
    print(f"Confidence:      {probs[0][pred_idx]*100:.2f}%")
    print("-" * 30)
    print(f"Probabilities:")
    print(f"  No Risk:       {probs[0][0]*100:.2f}%")
    print(f"  Moderate Risk: {probs[0][1]*100:.2f}%")
    print(f"  High Risk:     {probs[0][2]*100:.2f}%")
    print("="*50 + "\n")
    
    # Clean up
    if os.path.exists('temp_sample.csv'):
        os.remove('temp_sample.csv')

if __name__ == "__main__":
    predict_single_sample()
