# BDD100K Object Detection Benchmark

Benchmarking **YOLO11** and **RT-DETR** for object detection on the **BDD100K** dataset with robustness evaluation across different weather, lighting, and road scene conditions.

---

## Overview

This project implements a complete pipeline for training, inference, and evaluation of modern object detection models on the BDD100K dataset. Unlike standard BDD100K benchmarks, the models are trained only on clear daytime images and evaluated on a dedicated test set containing diverse weather, lighting, and road scene conditions to assess their robustness.

The main objective is to compare the performance of different object detectors under varying environmental conditions, including:

- ☀️ Weather (clear, rainy, snowy, overcast, etc.)
- 🌙 Time of day (daytime, night, dawn/dusk)
- 🛣️ Road scene (city street, highway, residential, etc.)

---

## Motivation

Object detectors are typically evaluated on overall benchmark metrics. However, real-world driving environments vary significantly in weather, illumination, and scene complexity.

This project investigates how detector performance changes under different environmental conditions using the BDD100K dataset. The goal is to provide a more comprehensive assessment of model robustness beyond standard validation metrics.

---

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

- YOLO11n 
- YOLO11s 
- RT-DETR 

---

## Dataset

This project uses the **BDD100K** object detection dataset.
Only the object detection task is used. The benchmark focuses on four object categories:

- Car
- Pedestrian
- Traffic Light
- Traffic Sign

### Dataset Split

Unlike the standard BDD100K split, this project creates a custom train/validation/test partition to evaluate model robustness under different environmental conditions.

#### Training Set

The training set contains **only clear daytime images**.

This allows the models to learn under ideal driving conditions without exposure to adverse weather or low-light environments.

#### Validation Set

The validation set also contains **only clear daytime images** and is used during training for model selection and monitoring.

#### Test Set

The test set contains **all remaining environmental conditions**, including:

**Weather**
- Rainy
- Snowy
- Overcast
- Partly cloudy
- Foggy
- Undefined weather
- Remaining clear images not used for training

**Time of day**
- Night
- Dawn/Dusk
- Remaining daytime images

**Road scene**
- City street
- Highway
- Residential
- Parking lot
- Tunnel
- Gas station

A fixed test set of **5,000 images** is used for all experiments.

### Motivation

The purpose of this split is to evaluate how well object detection models trained under ideal conditions generalize to unseen environmental conditions.

Instead of measuring only overall detection accuracy, the evaluation reports performance separately for:

- Weather conditions
- Time of day
- Road scene

This enables a detailed robustness analysis and allows direct comparison of model performance under different driving environments.

---

## Installation

Clone the repository

```bash
git clone https://github.com/zbalgabekova/bdd100k-object-detection-benchmark.git

cd bdd100k-object-detection-benchmark
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Pipeline

### 1. Compute dataset statistics, generate metadata, and produce distribution plots

```bash
python scripts/prepare_dataset.py
```

### 2. Create dataset splits

```bash
python scripts/create_splits.py --metadata outputs/metadata.csv --output splits --weather clear --timeofday daytime
```


