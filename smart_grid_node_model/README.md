# ⚡ Transformer-Based Fault Detection for Smart Grid Nodes

> Real-time, interpretable anomaly detection on embedded power systems using self-supervised Transformers.

![AUC ROC](images/roc_curve.png)
![Latent PCA](images/pca_latent_projection.png)

Paper Link: https://drive.google.com/file/d/1hBmqx_9sWqOwcMe0_67193EGOkG2q-n8/view?usp=sharing

---

## 🔍 Overview

This project implements a **self-supervised Transformer model** for fault detection in high-frequency voltage and current signals from Smart Grid Nodes. Unlike traditional thresholding methods or supervised classifiers, our approach requires **no labeled fault data** and offers **interpretability through attention visualizations** — all while being **deployable on microcontroller-class edge devices**.

---

## 🧠 Key Features

- ✅ **Self-supervised learning** via next-timestep forecasting
- ✅ **Transformer encoder** with positional encoding + multi-head attention
- ✅ **Zero-shot generalization** to unseen faults (flicker, overvoltage, drift, etc.)
- ✅ **Visual interpretability**: attention heatmaps, PCA latent space projection
- ✅ **Edge deployable** using TensorFlow Lite Micro (ESP32, STM32)
- ✅ **Threshold-free anomaly scoring** using dynamic percentile-based calibration

---

## 🧪 Dataset

- 300 time-series sequences (1 kHz, 1-second duration)
- 6 classes: normal, overvoltage, undervoltage, disconnect, flicker, drift
- Simulated with controlled fault injection and added noise
- See `/simulated_grid_data/` or generate your own via `data_simulation.py`

---

## 🧱 Architecture

```markdown
[Voltage/Current Input]
↓
[Linear Projection]
↓
[Positional Encoding]
↓
[Transformer Encoder (3 Layers, 4 Heads)]
↓
[Output Projection]
↓
[Reconstructed Signal]
```

## 📈 Results
```yaml
| Metric        | Value        |
|---------------|--------------|
| AUC (ROC)     | > 0.90       |
| Accuracy @ τ  | High (>90%)  |
| Latent Clustering | ✔️ PCA shows clear fault separation |
| Attention Focus | ✔️ Fault-localized heatmaps |
```
Key visualizations:
- `images/roc_curve.png`
- `images/error_histogram.png`
- `images/attention_flicker_L1_H1.png`
- `images/pca_latent_projection.png`
- `images/confusion_matrix.png`

---

## 🚀 Deployment

The model is compatible with TensorFlow Lite for Microcontrollers.

Steps:
1. Export model using `export_tflite.py`
2. Flash to embedded device (ESP32, STM32)
3. Use BLE or UART for real-time inference + logging

---

## 📂 Repository Structure
```markdown
├── script.ipynb # Full training & visualization notebook
├── simulated_grid_data/ # Synthetic dataset
├── images/ # All result visualizations (need to run script to generate)
├── export_tflite.py # TFLite conversion (optional)
├── README.md
```