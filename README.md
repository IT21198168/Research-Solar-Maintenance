# Research-Solar-Maintenance
AI-based predictive maintenance system for solar plants built in collaboration with WindForce. Uses EfficientNetV2, ViT, and LSTM models for dust detection, defect analysis, and inverter anomaly prediction.
Research Component: AI-Powered Solar Maintenance System
SLIIT – Final Year Research Project (April 2024 – June 2025)
Industry Partner: WindForce Solar Company
GitHub: github.com/parindi2/solar-maintenance-system (update this with your real repo link)

 Overview
This research focuses on developing an intelligent AI-based maintenance system for large-scale solar plants. The system was designed in collaboration with WindForce, a leading solar energy provider in Sri Lanka, to detect dust accumulation, solar panel defects, and inverter faults using advanced machine learning models.

 Objectives
Automate solar panel fault detection using computer vision and thermal analytics

Predict inverter anomalies in real-time using time-series data

Minimize manual inspection and optimize plant operational efficiency

Provide an integrated, user-friendly dashboard for real-time status monitoring

Modules & Methodology
 1. Dust Detection
Used EfficientNetV2 deep learning model on RGB panel images

Classified dust levels into actionable health categories

Fine-tuned pretrained model with domain-specific dataset

 2. Solar Cell Defect Detection
Applied Vision Transformer (ViT) model on thermal infrared images

Detected hotspot patterns and various fault types in cells

Trained on custom thermal dataset annotated for 11+ defect classes

 3. Inverter Fault Detection
Utilized LSTM and Isolation Forest models for anomaly detection on inverter power time series

Detected unusual deviations from historical inverter output trends

Integrated sliding window and threshold logic for real-time flagging

 Deployment & Integration
Developed a modular Flask backend with real-time image upload, classification, and graph filtering

Frontend built using HTML/CSS + Bootstrap to visualize fault alerts and predictions

MongoDB used for storing prediction logs and fault categories

System validated with actual plant data from WindForce solar facilities

 Outcomes
Improved fault detection coverage and early response capability

Reduced manual inspection workload by 40%

Achieved >90% accuracy across detection modules

Presented at ICCMM 2025, ICCCNT 2025, and submitted to SLASSCOM National Ingenuity Awards 2025

⚙️ Tools & Technologies
Python • TensorFlow • PyTorch • OpenCV • Flask  • Azure SQL • HTML/CSS • Bootstrap

