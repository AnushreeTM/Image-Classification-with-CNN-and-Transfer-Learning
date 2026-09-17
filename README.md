# Image Classification with CNN and Transfer Learning (MobileNetV2)

A deep-learning project that classifies images into **6 categories** using two approaches — a **custom Convolutional Neural Network (CNN)** built from scratch and a **transfer-learning model based on MobileNetV2** — and compares their performance.

## Overview

The goal of this project is to build, train, and evaluate image classification models on a multi-class image dataset, while handling real-world challenges such as **class imbalance**. Two models are trained and benchmarked against each other to determine which performs best.

## Key Features

- **End-to-end pipeline** — image extraction, resizing (224×224), train/validation split, and performance-optimized data loading with TensorFlow `tf.data`.
- **Class imbalance handling** — computes class weights with `sklearn.utils.class_weight` so under-represented classes (e.g. `Product_2`, `Background`) don't get ignored by the model.
- **Two models compared:**
  - A **custom CNN** with stacked Conv2D / MaxPooling layers.
  - **Transfer learning with MobileNetV2** (ImageNet weights), including a **fine-tuning** stage at a lower learning rate.
- **Comprehensive evaluation** — accuracy/loss curves, model-vs-model accuracy comparison, confusion matrices, and precision / recall / F1-score classification reports.

## Tech Stack

- **Python**
- **TensorFlow / Keras** — model building and training
- **scikit-learn** — class weights and evaluation metrics
- **Matplotlib & Seaborn** — visualization
- **NumPy, Pandas, PIL** — data handling
- Developed in **Google Colab**

## Workflow

1. **Data preparation** — extract images from a zip archive, resize to 224×224, and organize by class.
2. **Exploratory analysis** — inspect class distribution and visualize sample images per class.
3. **Data loading** — create training (80%) and validation (20%) datasets with caching, shuffling, and prefetching.
4. **Handle imbalance** — calculate class weights to balance training.
5. **Model 1 — Custom CNN** — build, train (20 epochs), and evaluate.
6. **Model 2 — MobileNetV2** — train the frozen base (10 epochs), then fine-tune (5 epochs) at a reduced learning rate.
7. **Compare & evaluate** — accuracy comparison, confusion matrices, and classification reports for both models.

## Results

Both models are evaluated on the validation set and compared on training and validation accuracy. The transfer-learning model (MobileNetV2) is used to benchmark against the custom CNN baseline. *(See the notebook for the final accuracy figures and plots.)*

## How to Run

1. Open `ML_IMAGE_CLASSIFICATION.ipynb` in **Google Colab** (or Jupyter).
2. Update the dataset path to point to your image zip file.
3. Run the cells top to bottom.

> **Note:** The notebook was built in Google Colab and mounts Google Drive to load the dataset. If running locally, replace the Drive-mount and path cells with your local dataset path.

## Author

**Anushree** — [GitHub](https://github.com/AnushreeTM)
