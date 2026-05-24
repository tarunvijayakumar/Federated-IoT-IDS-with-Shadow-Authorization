#!/usr/bin/env python3
"""
Flower IoT Client - Federated Learning + ThingsBoard TLS MQTT
fl-client-1 TOKEN: CGTWc6JHuFogN5oh3rRp
"""

import argparse
import sys
import time
import os
import json
import torch
import torch.nn as nn
import numpy as np
import flwr as fl
import paho.mqtt.client as mqtt
import ssl
from typing import Tuple, List, Dict

from federated_learning_framework.ml_model import IoTAnomalyNet
from federated_learning_framework.fl_client.local_trainer import get_model_and_data

# ========== YOUR TOKENS ==========
TOKEN = "CGTWc6JHuFogN5oh3rRp"  # fl-client-1 ThingsBoard token


class IoTClient(fl.client.NumPyClient):
    """Flower NumPyClient for IoT anomaly detection + ThingsBoard telemetry."""
    
    def __init__(self, csv_path: str):
        """Initialize client with data, model, and MQTT TLS client."""
        try:
            self.model, self.train_loader, self.criterion, self.optimizer = get_model_and_data(
                csv_path=csv_path,
                batch_size=32
            )
            print(f"[OK] Data loaded: {len(self.train_loader.dataset)} samples")
            
            # MQTT TLS Client Setup
            self.mqtt_client = mqtt.Client(client_id=f"fl-node-{os.getpid()}")
            self.mqtt_client.username_pw_set(TOKEN)
            self.mqtt_client.tls_set(
                ca_certs="./ca.crt",
                tls_version=ssl.PROTOCOL_TLSv1_2
            )
            self.mqtt_client.tls_insecure_set(True)  # self-signed localhost
            self.mqtt_client.connect("host.docker.internal", 8883, 60)  # TB MQTT
            self.mqtt_client.loop_start()
            
            time.sleep(2)
            print(f"[OK] MQTT TLS connected: token {TOKEN[:8]}...")
            
            # Heartbeat
            self.send_telemetry({"status": "FL_CLIENT_1_STARTED"})
            
            # Model info
            input_dim = next((m.in_features for m in self.model.modules() if isinstance(m, nn.Linear)), "unknown")
            print(f"[OK] Model input dim: {input_dim}")
            self.device = torch.device("cpu")
            self.model.to(self.device)
            
        except Exception as e:
            print(f"[ERROR] Init failed: {e}")
            raise
    
    def send_telemetry(self, data: Dict):
        """Publish to ThingsBoard."""
        try:
            payload = json.dumps(data)
            self.mqtt_client.publish("v1/devices/me/telemetry", payload)
            print(f"[MQTT] {payload}")
        except Exception as e:
            print(f"[MQTT ERR] {e}")
    
    def get_parameters(self, config) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=False)
    
    def fit(self, parameters, config) -> Tuple[List[np.ndarray], int, dict]:
        self.set_parameters(parameters)
        
        epochs = config.get("epochs", 1)
        lr = config.get("learning_rate", 0.01)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        print(f"[TRAIN] {epochs} epochs, lr={lr}")
        
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            self.model.train()
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * X_batch.size(0)
                total_samples += X_batch.size(0)
            
            avg_loss = epoch_loss / len(self.train_loader.dataset)
            total_loss += avg_loss
            
            self.send_telemetry({
                "epoch": epoch+1,
                "loss": float(avg_loss),
                "node": "fl-client-1"
            })
            print(f"[EPOCH {epoch+1}] loss: {avg_loss:.4f}")
        
        self.send_telemetry({
            "training_complete": True,
            "total_loss": float(total_loss/epochs),
            "total_samples": total_samples,
            "node": "fl-client-1"
        })
        
        return self.get_parameters(config), total_samples, {"loss": total_loss/epochs}
    
    def evaluate(self, parameters, config) -> Tuple[float, int, dict]:
        self.set_parameters(parameters)
        
        total_loss, total_correct, total_samples = 0.0, 0, 0
        
        self.model.eval()
        with torch.no_grad():
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                
                total_loss += loss.item() * X_batch.size(0)
                predictions = torch.argmax(logits, dim=1)
                total_correct += torch.sum(predictions == y_batch).item()
                total_samples += X_batch.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        self.send_telemetry({
            "eval_loss": float(avg_loss),
            "eval_accuracy": float(accuracy),
            "node": "fl-client-1"
        })
        
        print(f"[EVAL] loss: {avg_loss:.4f}, acc: {accuracy:.4f}")
        return avg_loss, total_samples, {"accuracy": accuracy}


def main(csv_path: str, server_address: str = "host.docker.internal:8080"):
    print("🚀 fl-client-1 IDS Training + TB Telemetry")
    print(f"Data: {csv_path} | Server: {server_address} | Token: {TOKEN[:8]}...")
    
    client = IoTClient(csv_path)
    fl.client.start_numpy_client(server_address, client=client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--server", type=str, default="host.docker.internal:8080")
    args = parser.parse_args()
    main(args.data, args.server)
