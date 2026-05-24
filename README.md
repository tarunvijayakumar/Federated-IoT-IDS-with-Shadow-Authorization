Privacy-Preserving IoT Security Framework using Federated Learning
A lightweight edge-based federated learning project for IoT intrusion detection with shadow authorization protection. The system trains intrusion detection models across distributed edge nodes without sending raw data to a central server, while also protecting shared-device credentials through a proxy-based authorization layer.

Overview
This project is designed for privacy-preserving IoT security in environments such as smart cities, healthcare, and industrial systems. It combines federated learning, deep learning-based anomaly detection, edge deployment, and a shadow authorization mechanism to reduce both data leakage risk and centralized exposure.

Key Features
Federated learning-based intrusion detection using Flower (FLWR)

CNN-LSTM model for IoT anomaly and attack detection

No raw telemetry sharing between clients and server

Shadow authorization proxy for protecting device credentials

Edge-friendly deployment using Docker and Docker Compose

Real-time monitoring support through ThingsBoard

Support for IoT traffic and telemetry-driven security analysis

Architecture
The framework contains the following major components:

Data Generation / Collection Module
Collects or prepares IoT traffic and telemetry data for local client training.

Feature Engineering Module
Performs preprocessing and feature selection for efficient training and inference.

Federated Learning Module
Coordinates distributed model training across multiple clients using FedAvg.

Shadow Mapping and Proxy Module
Protects credential usage and authorization flows during device access or sharing.

Edge IDS Deployment Module
Deploys the trained global model for low-latency intrusion detection at the edge.

Monitoring and Alerting Module
Streams outputs to dashboards for attack visibility, drift monitoring, and response.

Tech Stack
Language: Python 3.x

Deep Learning: PyTorch

Federated Learning: Flower (FLWR)

Containerization: Docker, Docker Compose

Monitoring: ThingsBoard CE

Data Processing: Pandas, NumPy

Messaging / IoT Integration: MQTT

Workflow
Prepare client-specific IoT datasets.

Start the federated server and client containers.

Train the global intrusion detection model over multiple federated rounds.

Aggregate local model updates using FedAvg.

Deploy the final global model to the edge IDS container.

Run real-time inference on incoming IoT telemetry.

Visualize alerts and monitoring metrics through dashboards.

Project Structure
bash
.
├── datasets/
├── federated_learning/
├── models/
├── shadow_proxy/
├── edge_ids/
├── monitoring/
├── docker-compose.yml
└── README.md
Expected Outcomes
Privacy-preserving collaborative training

Fast edge-side intrusion detection

Reduced dependency on centralized raw data collection

Protection against credential exposure in shared IoT environments

Scalable architecture for heterogeneous IoT deployments

Use Cases
Smart home and smart city device security

Industrial IoT monitoring

Healthcare IoT anomaly detection

Secure shared-device ecosystems

Future Improvements
Differential privacy or secure aggregation integration

Online learning for concept drift adaptation

Lightweight model compression for constrained edge devices

Automated response and mitigation engine

Kubernetes-based large-scale deployment

Author Notes
This project was developed as a cybersecurity-focused academic/research implementation for federated IoT security and privacy-preserving intrusion detection. It is suitable for final-year project demonstration, publication-oriented work, and further extension into IEEE-style research output.
