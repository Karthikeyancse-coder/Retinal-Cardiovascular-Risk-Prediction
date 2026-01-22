import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torchvision import models

class EfficientNetBranch(nn.Module):
    """
    Vision Branch using EfficientNet-B4 (Pretrained from Torchvision)
    """
    def __init__(self, output_dim=256, pretrained=True):
        super().__init__()
        # Load EfficientNet-B0 from torchvision (optimized for CPU/Speed)
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # EfficientNet-B0 last layer is 'classifier', we remove it to get features
        # The feature dim before classifier is 1280
        self.feature_dim = 1280
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Projection head to common dimension
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, x):
        features = self.backbone(x) # (B, 1792)
        return self.projection(features) # (B, 256)

class GATBranch(nn.Module):
    """
    Graph Branch using Graph Attention Networks (GATv2)
    """
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=256, heads=4):
        super().__init__()
        
        # GAT Layers
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.2)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.2)
        self.conv3 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.2)
        
        # Projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * heads, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, x, edge_index, batch):
        # x: (Nodes, 2), edge_index: (2, Edges)
        
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        
        # Global Pooling
        x = global_mean_pool(x, batch) # (B, hidden_dim * heads)
        
        return self.projection(x) # (B, 256)

class ClinicalBranch(nn.Module):
    """
    Clinical Branch for 5 High-Fidelity Features
    """
    def __init__(self, input_dim=5, output_dim=256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class ConcatFusion(nn.Module):
    """
    Simple Feature Concatenation Fusion Layer
    Concatenates embeddings from CNN, GNN, and Clinical branches.
    """
    def __init__(self, input_dim=256, output_dim=256):
        super().__init__()
        
        # 3 branches * 256 dim = 768
        concat_dim = input_dim * 3
        
        self.fusion_net = nn.Sequential(
            nn.Linear(concat_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, img_feat, graph_feat, clin_feat):
        # Concatenate along feature dimension (dim=1)
        # img: (B, 256), graph: (B, 256), clin: (B, 256)
        combined = torch.cat([img_feat, graph_feat, clin_feat], dim=1) # (B, 768)
        
        # Project back to standard dimension for classifier
        fused = self.fusion_net(combined) # (B, 256)
        
        return fused

class RetiCardNetAggressive(nn.Module):
    """
    RetiCardNet - Aggressive Optimization Version
    Target Accuracy: 99%
    """
    def __init__(self, num_classes=3):
        super().__init__()
        
        self.img_branch = EfficientNetBranch(output_dim=256)
        self.gnn_branch = GATBranch(output_dim=256)
        self.clin_branch = ClinicalBranch(output_dim=256)
        
        # Replaced Transformer with Concatenation Fusion
        self.fusion = ConcatFusion(input_dim=256, output_dim=256)
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, img, graph_x, graph_edge_index, graph_batch, clin_data):
        # Feature Extraction
        img_feat = self.img_branch(img)
        graph_feat = self.gnn_branch(graph_x, graph_edge_index, graph_batch)
        clin_feat = self.clin_branch(clin_data)
        
        # Fusion
        fused = self.fusion(img_feat, graph_feat, clin_feat)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits
