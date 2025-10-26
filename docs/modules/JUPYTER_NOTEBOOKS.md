# Jupyter Notebooks - Experimental Analysis

This document describes the Jupyter notebooks in the `Jupyter/` folder, which were used during the 2020 research period for exploratory data analysis, feature engineering, and model experimentation.

## Overview

The notebooks represent the interactive exploration phase of the research, where various approaches to anomaly detection and classification were tested before being converted into production Python scripts.

## Feature Engineering

### fisher.ipynb

**Purpose**: Fisher score calculation for feature selection

**What it does**:
- Loads all benign, Gafgyt, and Mirai attack data across all devices
- Calculates Fisher scores for all 115 features
- Ranks features by discriminative power for binary classification (benign vs attack)
- Generates ranked feature list used in `data/fisher/fisher.csv`

**Key outputs**:
- Top features identified: MI_dir_L1_weight, H_L1_weight, MI_dir_L3_weight
- Fisher scores quantify how well each feature separates benign from attack traffic

**Dataset statistics**:
```
Gafgyt attacks:  1,032,056 samples
Mirai attacks:   3,668,402 samples
Benign traffic:    555,932 samples
Total:          5,256,390 samples across 9 devices
```

## Anomaly Detection Experiments

### anomaly.ipynb

**Purpose**: Initial autoencoder exploration for anomaly detection

**What it does**:
- Basic autoencoder training on benign traffic
- Reconstruction error analysis
- Threshold determination experiments

### Anomaly_detection_standard_norm.ipynb

**Purpose**: Autoencoder with StandardScaler normalization

**What it does**:
- Tests StandardScaler for feature normalization
- Compares reconstruction errors with standardized features
- Evaluates anomaly detection thresholds

### Anomaly_detection_stand_norm.ipynb

**Purpose**: Alternative StandardScaler experiments

**What it does**:
- Variant implementation of standard normalization
- Different hyperparameter configurations
- Threshold sensitivity analysis

### Anomaly_detection_minmax_scaling.ipynb

**Purpose**: Autoencoder with MinMaxScaler normalization

**What it does**:
- Tests MinMaxScaler (0-1 range) instead of StandardScaler
- Compares reconstruction error distributions
- Evaluates impact of normalization choice on detection

### Anomaly_detection_minmax_scaling2.ipynb

**Purpose**: Extended MinMax scaling experiments

**What it does**:
- Follow-up experiments with MinMaxScaler
- Additional hyperparameter tuning
- Cross-device performance comparison

## Classification Experiments

### botnet_type.ipynb

**Purpose**: Multi-class classification (benign, Gafgyt, Mirai)

**What it does**:
- Trains neural network classifiers
- Tests different architectures and feature counts
- Generates confusion matrices
- Evaluates classification accuracy across attack types

### botnet_type2.ipynb

**Purpose**: Refined classification experiments

**What it does**:
- Improved classification models
- Feature selection impact analysis
- Per-device performance evaluation

### mirai_attack_type.ipynb

**Purpose**: Fine-grained Mirai attack classification

**What it does**:
- Classifies specific Mirai attack types (scan, ack, syn, udp, udpplain)
- Evaluates if different Mirai variants are distinguishable
- Tests classification on individual devices vs combined

### mirai_attack_type2.ipynb

**Purpose**: Extended Mirai type classification

**What it does**:
- Follow-up experiments on Mirai attack type classification
- Different model architectures
- Feature importance for attack type detection

## Utilities

### utils.py

**Purpose**: Shared utility functions for notebooks

**Functions**:
- `plot_confusion_matrix()`: Visualizes confusion matrices with optional normalization
- Used across classification notebooks for consistent visualizations

## Generated Artifacts

The notebooks produced several artifacts saved in the `Jupyter/` folder:

### Model Files
- `autoencoder_traffic.h5`: Trained autoencoder model (496 KB)

### Visualization Files
- `model.png`: Model architecture diagram (58 KB)
- `classification_model.png`: Classification model architecture (36 KB)

### Results Files
- `anomaly_scores.csv`: Anomaly detection evaluation results
- `classification_scores.csv`: Multi-class classification results
- `atk_classification_scores.csv`: Attack type classification results

### Subfolders
- `anomaly/`: Additional anomaly detection experiments
- `models/`: Saved model checkpoints from experiments

## Relationship to Production Code

The notebooks served as the experimental foundation for:

1. **Anomaly Detection** (`anomaly-detection/` module):
   - StandardScaler chosen over MinMaxScaler based on notebook experiments
   - Threshold calculation methodology refined in notebooks

2. **Classification** (`classification/` module):
   - Feature selection counts (3, 5, 10) determined from notebook experiments
   - Fisher score feature ranking from `fisher.ipynb`

3. **Feature Engineering**:
   - Top features identified in notebooks became default configurations
   - Fisher scores saved to `data/fisher/fisher.csv`

## Technical Notes

### Deprecated Code

The notebooks contain 2020-era code with deprecated patterns:
- `pandas.DataFrame.append()` (deprecated in pandas 1.4+)
- Mixed `keras.` and `tensorflow.keras.` imports
- Direct keras imports without TensorFlow wrapper

**Note**: These notebooks are preserved as-is for historical reference. They document the exploration process but are not maintained for execution in modern environments.

### Environment Requirements

To run these notebooks (not recommended for production):
- Python 3.9
- TensorFlow 2.10.0
- Pandas 1.3.5
- Jupyter/IPython
- See `environment-archive.yaml` for complete dependencies

## Usage Notes

These notebooks are **reference material** showing the research exploration process. For reproducible experiments, use the production scripts:
- `anomaly-detection/train_autoencoder.py`
- `classification/train_classifier.py`

The notebooks demonstrate the iterative process of:
1. Loading and exploring data
2. Testing different preprocessing approaches
3. Experimenting with model architectures
4. Evaluating performance across configurations
5. Selecting best approaches for production implementation

## Historical Context

**Created**: May-June 2020
**Purpose**: Graduate research exploration and experimentation
**Status**: Preserved for reference, not actively maintained

---

**See also**:
- `docs/modules/ANOMALY_DETECTION.md` - Production anomaly detection module
- `docs/modules/CLASSIFICATION.md` - Production classification module
- `docs/project-analysis/RETROSPECTIVE.md` - 2024 analysis of 2020 research
