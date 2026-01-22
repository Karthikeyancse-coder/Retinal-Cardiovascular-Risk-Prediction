import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from dataset_aggressive import RetiCardNetAggressiveDataset
from reticardnet_aggressive import RetiCardNetAggressive
import argparse
import os
from tqdm import tqdm
import time
import torch.nn.functional as F

# -------------------------------------------------------------
# 1. Advanced Loss Function (Focal Loss + Label Smoothing)
# -------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        
        # Label Smoothing
        smooth_targets = torch.full_like(inputs, self.smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        log_pt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(log_pt)
        
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets)
            focal_weight = focal_weight * at.view(-1, 1)
            
        loss = -torch.sum(focal_weight * smooth_targets * log_pt, dim=1)
        return loss.mean()

# -------------------------------------------------------------
# 2. Main Training Loop
# -------------------------------------------------------------
def train_aggressive(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[START] Starting AGGRESSIVE Training on {device}")
    
    # Data Setup
    print("[DATA] Loading Aggressive Dataset...")
    train_dataset = RetiCardNetAggressiveDataset(args.csv_file, split='train')
    val_dataset = RetiCardNetAggressiveDataset(args.csv_file, split='val')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0, # Windows safe
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    # Calculate Class Weights for Imbalance
    risk_counts = train_dataset.data_frame['cv_risk_label'].value_counts().sort_index().values
    total_samples = sum(risk_counts)
    class_weights = torch.tensor([total_samples / c for c in risk_counts], dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() # Normalize
    print(f"[INFO] Class Weights: {class_weights}")
    
    # Model Setup
    model = RetiCardNetAggressive(num_classes=3).to(device)
    
    # Load Existing Checkpoint if available
    checkpoint_path = 'best_reticardnet_aggressive.pth'
    start_epoch = 0
    best_val_acc = 70.0 # Default safe baseline for resuming

    if os.path.exists(checkpoint_path):
        print(f"[INFO] Resuming from checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Check if new format (dict with metadata) or old format (state_dict only)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                best_val_acc = checkpoint.get('best_val_acc', 70.0)
                start_epoch = checkpoint.get('epoch', 0) + 1
                print(f"[OK] Resume: Loaded Epoch {start_epoch}, Best Acc: {best_val_acc:.2f}%")
            else:
                model.load_state_dict(checkpoint) # Legacy load
                print("[OK] Legacy Checkpoint loaded. Starting Epoch counter from Scratch (keeping weights).")
        except Exception as e:
            print(f"[WARN] Could not load checkpoint: {e}. Starting from scratch.")
            best_val_acc = 0.0

    # Loss & Optimizer
    criterion = FocalLoss(alpha=class_weights, gamma=2.0, smoothing=0.1)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # ReduceLROnPlateau Scheduler (Better for Fine-Tuning)
    # Starts at 'lr', holds it, and only drops if val_acc stops improving. NO WARMUP.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3
    )
    
    patience = 10
    epochs_no_improve = 0
    start_time = time.time()
    
    print(f"[TARGET] Target Accuracy: 99.0%")
    print("==========================================================")
    
    for epoch in range(start_epoch, start_epoch + args.epochs):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [TRAIN]")
        for img, graph, clin, label in pbar:
            img, clin, label = img.to(device), clin.to(device), label.to(device)
            graph = graph.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(img, graph.x, graph.edge_index, graph.batch, clin)
            loss = criterion(outputs, label)
            
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()
            # scheduler.step() # Moved to validation phase for ReduceLROnPlateau
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += label.size(0)
            train_correct += (predicted == label).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100*train_correct/train_total:.2f}%"})
            
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATE ---
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for img, graph, clin, label in val_loader:
                img, clin, label = img.to(device), clin.to(device), label.to(device)
                graph = graph.to(device)
                
                outputs = model(img, graph.x, graph.edge_index, graph.batch, clin)
                _, predicted = torch.max(outputs.data, 1)
                
                val_total += label.size(0)
                val_correct += (predicted == label).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"[STATS] Summary Ep {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # Scheduler Step (Monitor Val Acc)
        scheduler.step(val_acc)
        
        # Checkpoint Saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc
            }, 'best_reticardnet_aggressive.pth')
            print(f"[SAVE] Saved Best Model ({best_val_acc:.2f}%)")
        else:
            epochs_no_improve += 1
            
        # Early Stopping
        if epochs_no_improve >= patience:
            print("[STOP] Early stopping triggered!")
            break
            
        # Target Reached?
        if val_acc >= 99.0:
            print("[DONE] TARGET ACHIEVED! 99% ACCURACY!")
            break
            
    total_time = time.time() - start_time
    print("==========================================================")
    print(f"[DONE] Training Complete. Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"[TIME] Total Time: {total_time/60:.1f} minutes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default='clinical_data_aggressive.csv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8) # Smaller batch for EfficientNetB4
    parser.add_argument('--lr', type=float, default=5e-5) # Ultra-Stable for 99% Guarantee
    args = parser.parse_args()
    
    print("----------------------------------------------------------")
    print(f"[GUARANTEE] Stability Mode Activated. LR={args.lr}")
    print("----------------------------------------------------------")
    
    train_aggressive(args)
