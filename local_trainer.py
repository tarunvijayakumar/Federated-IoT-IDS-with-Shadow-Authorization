from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .data_loader import load_iot_partition
from ..ml_model import IoTAnomalyNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_and_data(
    csv_path: str,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> Tuple[IoTAnomalyNet, DataLoader, nn.Module, torch.optim.Optimizer]:
    train_loader, input_dim = load_iot_partition(csv_path, batch_size)
    model = IoTAnomalyNet(input_dim=input_dim).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, train_loader, criterion, optimizer

def train_one_epoch(
    model: IoTAnomalyNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    running_loss = 0.0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)
    return running_loss / len(loader.dataset)

def evaluate(
    model: IoTAnomalyNet,
    loader: DataLoader,
    criterion: nn.Module,
) -> Dict[str, float]:
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            loss += criterion(logits, y).item() * X.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return {
        "loss": loss / total,
        "accuracy": correct / total if total > 0 else 0.0,
    }
