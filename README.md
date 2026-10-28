# 🔬 SteelScan AI: Real-Time Industrial Surface Defect Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)
![License](https://img.shields.io/badge/License-MIT-green)

**SteelScan AI** is an industrial-grade, deep learning-based surface defect inspection framework designed for high-speed quality control in steel and metallurgy production lines. Built using a custom **DualPath-AFNet** (Attention Fusion Network) architecture, it achieves end-to-end, high-precision detection and segmentation of complex surface defects (crazing, inclusion, patches, pitted surfaces, rolled-in scale, and scratches).

---

## 📌 Features

- **High-Precision Defect Detection:** Leverages Multi-Scale Feature Aggregation and Spatial-Channel Attention mechanisms tailored for fine-grained metallic texture anomalies.
- **Dual-Path Architecture:** Decouples global contextual feature extraction from fine-grained localized edge maps.
- **Production-Ready REST API:** Fast API inference endpoints supporting single-image analysis, batch processing, and live video stream frames.
- **Robust Preprocessing Pipeline:** Automated contrast adjustment (CLAHE), noise suppression, and dynamic image cropping for non-uniform lighting conditions.
- **Extensible Export Formats:** Export models to ONNX and TensorRT for low-latency edge device deployments (NVIDIA Jetson / Edge TPU).

---

## 🛠️ System Architecture
