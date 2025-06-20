import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import joblib
import tensorflow as tf
import io
import matplotlib
matplotlib.use('Agg')  # Fix tkinter issue
import matplotlib.pyplot as plt
from tensorflow.keras.utils import img_to_array, load_img
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from collections import OrderedDict

app = Flask(__name__)

# === Configuration Paths ===
MODEL_PATH = os.path.join(os.getcwd(), "models/efficientnet_v2_l_dust_detector.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2  # Change according to your dataset
CLASS_NAMES = {0: "Clean", 1: "Dusty"}
UPLOAD_FOLDER = 'static/uploads'
PLOT_FOLDER = 'static/plots'
INVERTER_GRAPHS_FOLDER = 'static/inverter_fault_graphs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

# === Load Models ===
def build_model():
    model_weights = torchvision.models.EfficientNet_V2_L_Weights.DEFAULT
    model = torchvision.models.efficientnet_v2_l(weights=model_weights).to(DEVICE)
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(in_features=1280, out_features=256, bias=True),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=NUM_CLASSES, bias=True)
    ).to(DEVICE)
    return model

model = build_model()
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

# Remove 'module.' prefix if the model was saved using DataParallel
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "")
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.eval()

# Load LSTM Model for Anomaly Detection
model_lstm = tf.keras.models.load_model('my_model.h5')
label_encoder = joblib.load('label_encoder.pkl')

# === Utility Functions ===
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# === Routes ===
@app.route("/")
def home():
    return render_template("index.html")  # Homepage with navigation

@app.route("/upload")
def upload_page():
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # Read image and transform
        image = Image.open(io.BytesIO(file.read()))
        image_tensor = transform_image(image)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()

        return jsonify({
            "predicted_class": CLASS_NAMES[pred_class],
            "confidence": float(confidence)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/inverter", methods=["GET", "POST"])
def inverter_fault():
    def get_available_graphs():
        graphs = []
        if not os.path.exists(INVERTER_GRAPHS_FOLDER):
            return graphs  # Return empty if folder does not exist

        for filename in os.listdir(INVERTER_GRAPHS_FOLDER):
            if filename.endswith(".png"):
                parts = filename.split("_")
                if len(parts) < 3:  # Ensure correct file naming format
                    continue
                inverter = parts[1]  # Extract inverter ID
                date = parts[-1].replace(".png", "")  # Extract date
                graphs.append({
                    "inverter": inverter, 
                    "date": date, 
                    "file": filename
                })
        return graphs

    graphs = get_available_graphs()
    inverters = sorted(set(g['inverter'] for g in graphs))
    dates = sorted(set(g['date'] for g in graphs))
    
    selected_inverter = request.form.get('inverter')
    selected_date = request.form.get('date')
    
    # Filter graphs based on user selection
    filtered_graphs = [g for g in graphs if (not selected_inverter or g['inverter'] == selected_inverter) and
                                                 (not selected_date or g['date'] == selected_date)]

    # Debugging logs
    print(f"Selected Inverter: {selected_inverter}, Selected Date: {selected_date}")
    print(f"Filtered Graphs: {filtered_graphs}")

    return render_template("inverter.html", 
                           inverters=inverters, 
                           dates=dates, 
                           graphs=filtered_graphs, 
                           selected_inverter=selected_inverter, 
                           selected_date=selected_date)


# ✅ FIXED: Updated `/anomaly` route (Allows GET & POST)
@app.route("/anomaly", methods=["GET", "POST"])
def upload_file():
    print(f"Received request with method: {request.method}")  # Debugging line

    if request.method == "GET":
        return render_template("anomaly.html")  # Serve the form on GET request

    if "file" not in request.files:
        return render_template("anomaly.html", error="No file uploaded")

    file = request.files["file"]
    if file.filename == "":
        return render_template("anomaly.html", error="No selected file")

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Load and preprocess image
    img = load_img(filepath, target_size=(40, 24))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction using LSTM model
    prediction = model_lstm.predict(img_array)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]

    # Plot prediction probabilities
    class_probabilities = prediction[0]
    class_labels = label_encoder.classes_

    plt.figure(figsize=(8, 5))
    bars = plt.bar(class_labels, class_probabilities, color='skyblue', edgecolor='black')
    plt.xlabel('Classes', fontsize=12, fontweight='bold')
    plt.ylabel('Probability', fontsize=12, fontweight='bold')
    plt.title('Prediction Probabilities', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add labels on bars
    for bar, prob in zip(bars, class_probabilities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{prob:.2f}', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

    # Save the probability plot
    plot_filename = f'plot_{os.path.splitext(filename)[0]}.png'
    plot_path = os.path.join(PLOT_FOLDER, plot_filename)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    return render_template('anomaly.html', 
                           filename=filename, 
                           predicted_class=predicted_class, 
                           plot_path=f'plots/{plot_filename}')


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
