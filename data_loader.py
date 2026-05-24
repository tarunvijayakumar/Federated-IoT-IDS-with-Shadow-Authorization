#!/usr/bin/env python3
"""
IoT Data Loader for Federated Learning
Loads CSV datasets and returns PyTorch DataLoaders
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple


class IoTDataset(Dataset):
    """PyTorch Dataset for IoT anomaly detection."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Feature array (N, D)
            y: Label array (N,)
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_iot_partition(csv_path: str, batch_size: int = 32) -> Tuple:
    """
    Load IoT dataset partition from CSV file.
    
    Args:
        csv_path: Path to CSV file (must contain 'label' column)
        batch_size: DataLoader batch size
    
    Returns:
        Tuple: (train_loader, input_dim)
    """
    try:
        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"[OK] Loaded {csv_path}: {df.shape}")
        
        # Validate label column
        if 'label' not in df.columns:
            raise ValueError(f"CSV must have 'label' column. Found: {df.columns.tolist()}")
        
        # Split features and labels
        X = df.drop('label', axis=1).values.astype(np.float32)
        y = df['label'].values.astype(np.int64)
        
        print(f"[OK] Features: {X.shape[1]}, Samples: {X.shape[0]}")
        print(f"[OK] Label distribution: {np.bincount(y)}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Create dataset and loader
        dataset = IoTDataset(X_train, y_train)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        input_dim = X_train.shape[1]
        
        print(f"[OK] Data loaded: {len(dataset)} samples")
        print(f"[OK] Batch size: {batch_size}")
        print(f"[OK] Model input dim: {input_dim}")
        
        return train_loader, input_dim
    
    except FileNotFoundError:
        print(f"[ERROR] File not found: {csv_path}")
        raise
    except ValueError as e:
        print(f"[ERROR] Data validation error: {e}")
        raise
    except Exception as e:
        print(f"[ERROR] Error loading {csv_path}: {e}")
        raise


def load_iot_test_partition(csv_path: str) -> Tuple:
    """
    Load test set from CSV for evaluation.
    
    Returns:
        Tuple: (X_test, y_test, input_dim)
    """
    try:
        df = pd.read_csv(csv_path)
        
        if 'label' not in df.columns:
            raise ValueError(f"CSV must have 'label' column")
        
        X = df.drop('label', axis=1).values.astype(np.float32)
        y = df['label'].values.astype(np.int64)
        
        # Standardize (same as training)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X, y, X.shape[1]
    
    except Exception as e:
        print(f"[ERROR] Failed to load test data: {e}")
        raise


if __name__ == "__main__":
    # Test the loader
    print("[TEST] Testing data loader...")
    
    csv_path = "datasets/feature_vectors/client1.csv"
    train_loader, input_dim = load_iot_partition(csv_path, batch_size=32)
    
    print(f"\n[OK] Data loader test passed!")
    print(f"[OK] Input dimension: {input_dim}")
    print(f"[OK] Number of batches: {len(train_loader)}")
    
    # Show first batch
    for X_batch, y_batch in train_loader:
        print(f"[OK] First batch shape: {X_batch.shape}, Labels: {y_batch.shape}")
        break
