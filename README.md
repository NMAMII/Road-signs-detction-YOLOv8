# Self-Driving Car Object Detection using YOLOv8 🚗

This project demonstrates training a YOLOv8 model for object detection in a self-driving car scenario. The model is trained to detect various traffic signs and signals crucial for autonomous driving.

## Setup Instructions

### 1. Environment Setup ⚙️

- Ensure you have the necessary dependencies installed. The notebook utilizes libraries such as PyTorch, Ultralytics, and Roboflow.
- for downloading ultralytics lib to get access to YOLOv8 model you can run this command in your notebook

```!git clone https://github.com/Myworkss/ultralytics.git```
- You may need a CUDA-enabled GPU for faster training. Verify CUDA and GPU drivers are correctly installed.

### 2. Dataset Preparation 🗃️

- The dataset used in this project is sourced from Roboflow, specifically tailored for self-driving car scenarios.

### 3. Training the Model 🏋️‍♀️

- The YOLOv8 model is trained using the provided dataset for 150 epochs with an image size of 640x640 pixels.
- Training metrics such as precision, recall, and mAP are displayed to assess model performance.

### 4. Validation 🎯

- The trained model is validated to evaluate its precision and detection accuracy.
- Various traffic signs and signals are evaluated individually to analyze the model's performance per class.

## Model Evaluation ✅

- The best trained model (`best.pt`) is saved in the `runs/detect/weights` directory.
- Additional testing using videos and webcam inputs can be conducted using the provided model in the "Testing the Model" folder.

## Repository Structure

- `notebook.ipynb`: Jupyter notebook containing the code for model training and validation.
- `README.md`: This file, providing an overview of the project, setup instructions, and model evaluation details.
- `runs/`: Directory containing training logs, model weights, and evaluation results.
- `Testing the model/`: This file is for testing the model we trained on videos, webcam and adjusting the detection square for final touchs. 

## Further Exploration

- For further exploration and testing of the model, refer to the provided notebook and model weights.
- Additional testing scenarios such as real-time inference on videos and webcam inputs can be explored using the trained model.

