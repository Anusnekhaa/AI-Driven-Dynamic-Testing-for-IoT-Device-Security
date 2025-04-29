from flask import render_template, Flask, request, jsonify
import numpy as np
import pandas as pd
import json
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')  # This disables the GUI and uses backend suited for Flask/server
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch_geometric
from torch_geometric.utils import from_networkx, add_self_loops, degree
from torch_geometric.nn import MessagePassing
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.loader import NeighborSampler
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import networkx as nx
import pandas as pd
import socket
import struct
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

app = Flask(__name__)

label_col = "Label"

# --- Load label dictionary ---
with open(r"", "r") as f: #upload path for json label dictionary 
    label_dict = json.load(f)
    inv_label_dict = {v: k for k, v in label_dict.items()}

attack_types = list(label_dict.keys())
map_type = {label: idx for idx, label in enumerate(attack_types)}
inv_map_type = {v: k for k, v in map_type.items()}

# --- Register huber loss ---
@tf.keras.utils.register_keras_serializable(package="Custom", name="huber_loss")
def huber_loss(y_true, y_pred, clip_value=1.0):
    x = y_true - y_pred
    condition = tf.abs(x) < clip_value
    squared_loss = 0.5 * tf.square(x)
    linear_loss = clip_value * (tf.abs(x) - 0.5 * clip_value)
    return tf.where(condition, squared_loss, linear_loss)

# --- Load model ---
json_path = r"" #upload gnn model .json file 
weights_path = r"" #upload gnn model weights .pth file 

with open(json_path) as f:
    model = model_from_json(json.load(f))

model.load_weights(weights_path)
model.compile(loss=huber_loss, optimizer="sgd")

# --- Helper to convert plots to base64 ---
def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return encoded

X_test = pd.read_csv(r"") #X_test.csv - upload the path for your inference dataset
cols_to_norm = ['LONGEST_FLOW_PKT', 'RETRANSMITTED_IN_PKTS', 'MAX_IP_PKT_LEN', 'TCP_WIN_MAX_IN', 'RETRANSMITTED_OUT_BYTES', 'DST_TO_SRC_SECOND_BYTES', 'OUT_PKTS', 'DURATION_OUT', 'IN_PKTS', 'FLOW_DURATION_MILLISECONDS', 'MIN_IP_PKT_LEN', 'NUM_PKTS_512_TO_1024_BYTES', 'RETRANSMITTED_IN_BYTES', 'DURATION_IN', 'SHORTEST_FLOW_PKT', 'IN_BYTES', 'DST_TO_SRC_AVG_THROUGHPUT', 'RETRANSMITTED_OUT_PKTS', 'SRC_TO_DST_SECOND_BYTES', 'TCP_WIN_MAX_OUT', 'SRC_TO_DST_AVG_THROUGHPUT', 'NUM_PKTS_128_TO_256_BYTES', 'NUM_PKTS_1024_TO_1514_BYTES', 'OUT_BYTES', 'NUM_PKTS_256_TO_512_BYTES', 'NUM_PKTS_UP_TO_128_BYTES', 'MAX_TTL', 'MIN_TTL']

from sklearn.preprocessing import LabelEncoder

test_le = LabelEncoder()
X_test['Attack'] = test_le.fit_transform(X_test['Attack'])
test_class_map = test_le.classes_

X_test['h'] = X_test[ cols_to_norm ].values.tolist()

G_nx_test = nx.from_pandas_edgelist(X_test, "saddr", "daddr", ['h', 'Label', 'Attack'], create_using=nx.MultiDiGraph())

G_pyg_test = from_networkx(G_nx_test)

test_num_nodes = G_pyg_test.num_nodes
test_num_edges = G_pyg_test.num_edges

G_pyg_test.x = th.ones(test_num_nodes, len(X_test['h'].iloc[0]))

test_edge_attr_list = []
test_edge_label_list = []
test_edge_class_list = []

for u, v, key, data in G_nx_test.edges(keys=True, data=True):
    test_edge_attr_list.append(data['h']) 
    test_edge_label_list.append(data['Label']) 
    test_edge_class_list.append(data['Attack'])

G_pyg_test.edge_attr = th.tensor(test_edge_attr_list, dtype=th.float32)
benign_class_index = test_le.transform(['Benign'])[0]
edge_labels_binary = [0 if attack == benign_class_index else 1 for attack in test_edge_class_list]
G_pyg_test.edge_label = th.tensor(edge_labels_binary, dtype=th.long)

G_pyg_test.edge_class = th.tensor(test_edge_class_list, dtype=th.long)

print("Number of edges in G_pyg_test:", G_pyg_test.num_edges)
print("Number of node in G_pyg_test:", G_pyg_test.num_nodes)
print("Shape of node in G_pyg_test:", G_pyg_test.x.shape)
print("Shape of edge attr in G_pyg_test:", G_pyg_test.edge_attr.shape)
print("Shape of edge label in G_pyg_test:", G_pyg_test.edge_label.shape)
print("Shape of edge class in G_pyg_test:", G_pyg_test.edge_class.shape)

# --- Flask Route for Inference ---
@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "CSV file missing"}), 400

    file = request.files['file']
    try:
        df = pd.read_csv(file)

        if label_col not in df.columns:
            return jsonify({"error": f"Label column '{label_col}' missing in the CSV."}), 400

        true_labels = df[label_col].map(label_dict).map(map_type).to_numpy()
        features = df.drop(columns=[label_col]).to_numpy()

        preds = model.predict(features)
        pred_classes = np.argmax(preds, axis=1)

        graph_info = {
            "num_edges": G_pyg_test.num_edges,
            "num_nodes": G_pyg_test.num_nodes,
            "node_shape": list(G_pyg_test.x.shape),
            "edge_attr_shape": list(G_pyg_test.edge_attr.shape),
            "edge_label_shape": list(G_pyg_test.edge_label.shape),
            "edge_class_shape": list(G_pyg_test.edge_class.shape),
        }

        # Confusion Matrix
        # Create the confusion matrix
        cm = confusion_matrix(true_labels, pred_classes)

        # Plot as heatmap
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax,cmap=sns.cubehelix_palette(as_cmap=True))
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")
        cm_img = plot_to_base64(fig)  


        # Normalized Confusion Matrix
        cm_norm = confusion_matrix(true_labels, pred_classes, normalize="true")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=attack_types).plot(ax=ax2, cmap="Blues", values_format=".2f")
        norm_cm_img = plot_to_base64(fig2)

        cm = confusion_matrix(true_labels, pred_classes, labels=list(inv_map_type.keys()))

        # Get counts for each class
        estimated_correct_labels = np.diag(cm)
        false_negatives = np.sum(cm, axis=1) - estimated_correct_labels
        false_positives = np.sum(cm, axis=0) - estimated_correct_labels

        # Prepare bar positions and labels
        pos = np.arange(len(estimated_correct_labels))
        width = 0.35

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))

        # Green: Correct
        p1 = plt.bar(pos, estimated_correct_labels, width, color='g')

        # Red: False Negatives
        p2 = plt.bar(pos + width, false_negatives, width, color='r')

        # Blue: False Positives stacked on False Negatives
        p3 = plt.bar(pos + width, false_positives, width,
             bottom=false_negatives, color='b')

        # X-axis labels
        ax.set_xticks(pos + width / 2)
        ax.set_xticklabels([inv_map_type[i] for i in range(len(estimated_correct_labels))], rotation='vertical')

        # Title & Legend
        accuracy = np.sum(estimated_correct_labels) / len(true_labels)
        ax.set_title('Test set scores, Acc = {:.2f}'.format(100 * accuracy))
        plt.legend(('Correct estimated', 'False negative', 'False positive'))
        bar_img = plot_to_base64(fig)

        # Performance Metrics
        acc = accuracy_score(true_labels, pred_classes)
        f1 = f1_score(true_labels, pred_classes, average='weighted')
        precision = precision_score(true_labels, pred_classes, average='weighted')
        recall = recall_score(true_labels, pred_classes, average='weighted')

        # Use prediction confidence as reward (sum of max predicted probabilities)
        total_reward = int(np.sum(pred_classes == true_labels))
        num_samples = len(features)
        accuracy_percent = float(100 * total_reward / num_samples)

        # Console-style print
        print('\r\nNumber of samples: {} | Accuracy = {}%'.format(total_reward, num_samples, accuracy_percent))


        return jsonify({
            "graph_info": graph_info,
            "num_samples": num_samples,
            "accuracy_percent": round(accuracy_percent, 2),
            "predicted_classes": [inv_map_type[i] for i in pred_classes.tolist()],
            "confusion_matrix": cm.tolist(),
            "normalized_confusion_matrix": cm_norm.tolist(),
            "cm_image_base64": cm_img,
            "norm_cm_image_base64": norm_cm_img,
            "bar_chart_base64": bar_img,
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
