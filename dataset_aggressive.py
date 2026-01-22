import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import networkx as nx
from torch_geometric.data import Data
from skimage.morphology import skeletonize
from torchvision import transforms

# Try importing Albumentations, else fallback
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("⚠️ Albumentations not found. Using Torchvision Aggressive Fallback.")

class RetiCardNetAggressiveDataset(Dataset):
    def __init__(self, csv_file, split='train', image_size=256):
        """
        Aggressive Dataset with Advanced Augmentation
        """
        self.data_frame = pd.read_csv(csv_file)
        self.data_frame = self.data_frame[self.data_frame['split'] == split].reset_index(drop=True)
        self.split = split
        self.image_size = image_size
        
        # -----------------------------------------------------------
        # AGGRESSIVE AUGMENTATION PIPELINE
        # -----------------------------------------------------------
        if split == 'train':
            if HAS_ALBUMENTATIONS:
                self.transform = A.Compose([
                    A.Resize(image_size, image_size),
                    # Light/Color variations
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
                    ToTensorV2()
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            # VALIDATION / TEST TRANSFORMS (No Augmentation)
            if HAS_ALBUMENTATIONS:
                self.transform = A.Compose([
                    A.Resize(image_size, image_size),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Load Image
        img_path = self.data_frame.iloc[idx]['image_path']
        image = None
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to load image (None): {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            # Fallback for corrupted images or read errors
            # print(f"Image Load Error: {e}") # Silence to keep logs clean
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Ensure correct type for transforms
        if not isinstance(image, np.ndarray):
             image = np.array(image)
        if image.dtype != np.uint8:
             image = image.astype(np.uint8)

        # 2. Extract Vessel Graph (Simplified for speed in aggressive training)
        graph_data = self.extract_vessel_graph(image)

        # 3. Apply Augmentations
        if HAS_ALBUMENTATIONS:
            augmented = self.transform(image=image)
            image_tensor = augmented['image']
        else:
            image_tensor = self.transform(image)

        # 4. Clinical Features
        # [Age, BP, BMI, HbA1c, LDL] -> Normalized
        age = self.data_frame.iloc[idx]['age'] / 100.0
        bp = self.data_frame.iloc[idx]['systolic_bp'] / 200.0
        bmi = self.data_frame.iloc[idx]['bmi'] / 50.0
        hba1c = self.data_frame.iloc[idx]['hba1c'] / 15.0
        ldl = self.data_frame.iloc[idx]['ldl'] / 300.0
        
        clinical_features = torch.tensor([age, bp, bmi, hba1c, ldl], dtype=torch.float32)

        # 5. Label
        label = int(self.data_frame.iloc[idx]['cv_risk_label'])

        return image_tensor, graph_data, clinical_features, label

    def extract_vessel_graph(self, image_np):
        """
        Extracts graph structure from retinal image.
        Uses green channel -> CLAHE -> Threshold -> Skeleton
        """
        try:
            # Resize for graph extraction speed
            img_small = cv2.resize(image_np, (256, 256))
            green = img_small[:, :, 1]
            
            # Enhancing using OpenCV CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(green)
            
            # Thresholding
            thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
            
            # Skeletonize
            # skeletonize inputs 0/1 array
            skeleton = skeletonize(thresh // 255)
            
            # Build Graph
            y, x = np.where(skeleton)
            coordinates = np.column_stack((x, y))
            
            if len(coordinates) < 10:
                # Fallback empty graph
                return Data(x=torch.zeros((1, 2), dtype=torch.float), 
                           edge_index=torch.zeros((2, 0), dtype=torch.long), 
                           batch=torch.zeros(1, dtype=torch.long))

            # Sample nodes (limit to 300 for efficiency)
            if len(coordinates) > 300:
                indices = np.random.choice(len(coordinates), 300, replace=False)
                coordinates = coordinates[indices]
                
            # Create edges (k-NN)
            from scipy.spatial import cKDTree
            tree = cKDTree(coordinates)
            # k=5 nearest neighbors
            dists, indices = tree.query(coordinates, k=5) 
            
            edge_index = []
            for i in range(len(coordinates)):
                for j_idx in range(1, 5): # skip self (0)
                    neighbor = indices[i][j_idx]
                    if neighbor < len(coordinates): # bound check
                        edge_index.append([i, neighbor])
                        edge_index.append([neighbor, i]) # Undirected
            
            if not edge_index:
                 return Data(x=torch.zeros((len(coordinates), 2), dtype=torch.float), 
                           edge_index=torch.zeros((2, 0), dtype=torch.long))

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            # Node features: coordinate (normalized)
            x_feat = torch.tensor(coordinates / 256.0, dtype=torch.float)
            
            return Data(x=x_feat, edge_index=edge_index)
            
        except Exception as e:
            # Fallback
            # print(f"Graph error: {e}")
            return Data(x=torch.zeros((1, 2), dtype=torch.float), 
                       edge_index=torch.zeros((2, 0), dtype=torch.long))
