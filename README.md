# AI-Driven Dynamic Testing for IoT Device Security: Detection of Network Attacks

## Overview

This project explores AI-driven detection and classification of network attacks in IoT environments. It combines network-traffic simulation, packet capture, flow-feature extraction, deep-learning models, and Flask-based inference applications.

The project contains two detection approaches:

- **Double Deep Q-Network (DDQN):** a reinforcement-learning-based approach for network-traffic classification.
- **Graph Neural Network (GNN):** a graph-based approach that learns from relationships represented in network traffic data.

The DDQN and GNN pipelines have separate model implementations and supporting datasets. The GNN workflow includes packet captures inspected with Wireshark and flow-feature extraction using CICFlowMeter. The project also contains NS-3 simulation material for benign and DDoS traffic scenarios.

## Project workflow

<img width="1211" height="528" alt="image" src="https://github.com/user-attachments/assets/0b8de616-c5d5-4656-9b44-692ccd19cc9a" />

1. **Traffic simulation:** Generate benign and DDoS IoT network scenarios using NS-3.
2. **Packet capture and inspection:** Capture network traffic and inspect PCAP files using Wireshark.
3. **Flow-feature extraction:** Process packet captures with CICFlowMeter to obtain structured traffic features.
4. **Dataset preparation:** Prepare the relevant traffic data and labels for the DDQN and GNN pipelines.
5. **Model development:** Train and evaluate the DDQN and GNN models using their respective implementations and datasets.
6. **Inference:** Use the Flask applications to load the trained models, process test data, and display predictions or evaluation outputs.

## Dataset

All related training and inference datasets for both the models are located in this link.The complete datasets used for the DDQN and GNN models, including network traffic captures and processed datasets, are hosted externally.

Please download the datasets from:

[Dataset Repository](https://drive.google.com/drive/folders/1Rs_FurDMvhra_KOHFQt0Upd0w_Fu1PrF)

## Models

### Double Deep Q-Network (DDQN)

The DDQN component applies a Double Deep Q-Network approach to network-traffic classification. Its implementation, notebooks, and related files are located in `ddqn_model/`.

### Graph Neural Network (GNN)

The GNN component represents network communication as graph-structured data and applies a graph neural network for classification. Its implementation, notebooks, model files, and related data are located under `gnn_model/`.

Placeholders are given for dataset directories within the code for each model. Add directory path to the code based on the local folder structure after dataset download.

## Tools and technologies

| Tool / technology | Role in the project |
|---|---|
| NS-3 | IoT network-traffic simulation |
| NetAnim | Visualization of simulated network activity |
| Wireshark | Packet capture inspection and analysis |
| CICFlowMeter | Extraction of flow-based traffic features |
| Python | Data processing, model development and inference |
| DDQN | Reinforcement-learning-based traffic classification |
| GNN | Graph-based traffic classification |
| Flask | Model-inference applications and web interface |
| Jupyter Notebook | Model development and experimentation |

## Results

`out/` contain demo video on DDoS and Benign traffic attack scenarious for an IoT-based environment. 

<img width="703" height="232" alt="image" src="https://github.com/user-attachments/assets/a5c51ddc-66d3-4e02-aeb1-1bf6aa2c468c" />

<img width="718" height="372" alt="image" src="https://github.com/user-attachments/assets/fac39be5-7f0e-4dc8-9533-923d2992bfd8" />

<img width="713" height="407" alt="image" src="https://github.com/user-attachments/assets/da23e1fa-000a-4ffd-ae40-98557a7af31a" />


## Responsible use

This project is intended for educational, research, and authorized security-testing purposes.


