import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torchvision import transforms
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
import pandas as pd
import sys

# Import from Aggressive setup
from dataset_aggressive import RetiCardNetAggressiveDataset
# Import from Aggressive setup
from dataset_aggressive import RetiCardNetAggressiveDataset
from reticardnet_aggressive import EfficientNetBranch, GATBranch, ClinicalBranch
import torch.nn.functional as F

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

def custom_collate(batch):
    # Filter out None items
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
        
    # Batch is a list of tuples: (image, graph, clinical, label)
    images = torch.stack([item[0] for item in batch])
    
    graph_list = [item[1] for item in batch]
    graphs = Batch.from_data_list(graph_list)
    
    clinical = torch.stack([item[2] for item in batch])
    labels = torch.stack([torch.tensor(item[3], dtype=torch.long) for item in batch])
    
    return images, graphs, clinical, labels

def evaluate_model(args):
    # Set unicode output for Windows console safety
    sys.stdout.reconfigure(encoding='utf-8')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset & Dataloader
    # Note: dataset_aggressive.py handles transforms internally based on split
    test_dataset = RetiCardNetAggressiveDataset(args.csv_file, split='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=custom_collate, num_workers=0)
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load Model
    model = RetiCardNetAggressive(num_classes=3).to(device)
    
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False) # Safe load
            # Handle dictionary checkpoint vs just state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return
    else:
        print("Warning: No checkpoint provided. Using untrained model.")
    
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Starting inference...", flush=True)
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if batch is None: continue
            
            imgs, graphs, clin, lbls = batch
            imgs, graphs, clin, lbls = imgs.to(device), graphs.to(device), clin.to(device), lbls.to(device)
            
            # Forward pass match signature: img, graph_x, graph_edge_index, graph_batch, clin_data
            outputs = model(imgs, graphs.x, graphs.edge_index, graphs.batch, clin)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if i % 10 == 0:
                print(f"Processed batch {i}/{len(test_loader)}", end='\r')
                
    print("\nInference complete.")
    
    # Calculate Metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='weighted') * 100
    precision = precision_score(all_labels, all_preds, average='weighted') * 100
    recall = recall_score(all_labels, all_preds, average='weighted') * 100
    
    # ROC-AUC (One-vs-Rest for multi-class)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted') * 100
    except:
        auc = 0.0
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Detailed Classification Report
    target_names = ['No Risk', 'Moderate Risk', 'High Risk'] # Assuming 0, 1, 2 mapping
    clf_report = classification_report(all_labels, all_preds, target_names=target_names)
    
    print("\n" + "="*60)
    print("FULL TEST SET EVALUATION (1151 IMAGES)")
    print("="*60)
    print(f"Total Accuracy:  {accuracy:.2f}%")
    print(f"Weighted F1:     {f1:.2f}%")
    print(f"Weighted Precision: {precision:.2f}%")
    print(f"Weighted Recall:    {recall:.2f}%")
    print(f"ROC-AUC:         {auc:.2f}%")
    print("-" * 60)
    print("\nDetailed Classification Report:\n")
    print(clf_report)
    print("-" * 60)
    print("\nConfusion Matrix:")
    print(cm)
    print("="*60)
    
    # Save results to file
    with open("full_test_evaluation_report.txt", "w", encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("FULL TEST SET EVALUATION (1151 IMAGES)\n")
        f.write("="*60 + "\n")
        f.write(f"Total Accuracy:  {accuracy:.2f}%\n")
        f.write(f"Weighted F1:     {f1:.2f}%\n")
        f.write(f"ROC-AUC:         {auc:.2f}%\n")
        f.write("-" * 60 + "\n")
        f.write("\nDetailed Classification Report:\n")
        f.write(clf_report)
        f.write("\n" + "-" * 60 + "\n")
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default=r"e:\HD_Model\Antigravity\clinical_data_aggressive.csv")
    parser.add_argument('--checkpoint', type=str, default=r"e:\HD_Model\Antigravity\best_reticardnet_aggressive.pth")
    parser.add_argument('--batch_size', type=int, default=16)
    
    args = parser.parse_args()
    evaluate_model(args)
