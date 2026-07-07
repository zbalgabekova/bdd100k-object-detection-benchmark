# BDD100K Object Detection Benchmark

Benchmarking **YOLO11** and **RT-DETR** for object detection on the **BDD100K** dataset with robustness evaluation across different weather, lighting, and road scene conditions.

---

## Overview

This project implements a complete pipeline for training, inference, and evaluation of modern object detection models on the BDD100K dataset.

The main objective is to compare the performance of different object detectors under varying environmental conditions, including:

- ☀️ Weather (clear, rainy, snowy, overcast, etc.)
- 🌙 Time of day (daytime, night, dawn/dusk)
- 🛣️ Road scene (city street, highway, residential, etc.)

## Motivation

Object detectors are typically evaluated on overall benchmark metrics. However, real-world driving environments vary significantly in weather, illumination, and scene complexity.

This project investigates how detector performance changes under different environmental conditions using the BDD100K dataset. The goal is to provide a more comprehensive assessment of model robustness beyond standard validation metrics.


The project includes:

- Dataset preparation
- Train/validation/test split generation
- Conversion to YOLO format
- Model training
- Prediction generation
- Detailed evaluation by environmental conditions

---

## Models

Currently supported models:

- YOLO11n (done)
- YOLO11s (in the process)
- RT-DETR (in the process)

---

## Dataset

This project uses the **BDD100K** object detection dataset.
Only the object detection task is used.

---

