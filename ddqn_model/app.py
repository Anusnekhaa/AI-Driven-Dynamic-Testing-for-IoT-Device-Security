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

app = Flask(__name__)

label_col = "Label"

# --- Load label dictionary ---
with open(r"C:\Projects\Dynamic Attack Detection\ddqn_model\ddqn_model\label_dict.json", "r") as f:
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
json_path = r"C:\Projects\Dynamic Attack Detection\ddqn_model\ddqn_model\DDQN_model.json"
weights_path = r"C:\Projects\Dynamic Attack Detection\ddqn_model\ddqn_model\DDQN_model.weights.h5"

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
        print('\r\nTotal reward: {} | Number of samples: {} | Accuracy = {}%'.format(total_reward, num_samples, accuracy_percent))


        return jsonify({
            "total_reward": total_reward,
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
