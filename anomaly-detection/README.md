# Anomaly Detection Module

This module implements autoencoder-based deep learning for detecting IoT botnet attacks. The approach reproduces the models from Meidan et al.'s paper ["N-BaIoT—Network-Based Detection of IoT Botnet Attacks Using Deep Autoencoders"](https://arxiv.org/pdf/1805.03409.pdf).

## Overview

The anomaly detection system:

- Trains autoencoders on benign (normal) network traffic patterns
- Detects attacks by identifying deviations from learned normal behavior
- Uses reconstruction error (MSE) as the anomaly detection metric
- Evaluates against Mirai and BASHLITE (Gafgyt) botnet attacks

Device configurations and hyperparameters are defined in `../config/devices.json`.

## Environment Setup

Create and activate the conda environment:

```bash
conda env create -f ../environment-archive.yaml
conda activate botnet-archive-2020
```

See `../environment-archive.yaml` for complete dependency specifications.

## Data Preparation

Download the N-BaIoT dataset:

```bash
python ../scripts/download_data.py
```

This script downloads benign traffic and attack data for each IoT device. You will need to manually extract the `.rar` archives - place extracted CSV files in the same folder as the benign traffic data.

**Dataset source**: UCI Machine Learning Repository - N-BaIoT dataset

## Training Models

### Per-Device Models

Train models for specific devices:

```bash
python train.py                    # Train for all devices
python train.py Danmini_Doorbell   # Train for specific device
```

Training parameters (epochs, learning rate) are loaded from `../config/devices.json` for each device. Successfully trained models are saved as `model.h5` in the respective device folder.

### Combined Model

Train a single model on all benign traffic:

```bash
python train_combined.py
```

Output: `combined_model.h5`

### Federated Learning Experiments

Experimental TensorFlow Federated implementations:

```bash
python train_v04.py              # Latest FL implementation (simulation)
python run_experiment_*.py       # Various FL experiments
```

**Note**: These are simulation-based implementations, not production federated deployments. See `../docs/archived/experimental/README.md` for details on the experimental evolution.

## Training Logs

Training logs are saved to:

- Per-device: `{device}/logs/`
- Combined model: `combined_model_logs/`

View training progress with TensorBoard:

```bash
tensorboard --logdir={device}/logs
```

## Testing and Evaluation

Test trained models:

```bash
python test.py                   # Test device-specific models
python test_combined.py          # Test combined model
```

The test scripts:

1. Calculate anomaly detection threshold using validation data
2. Compare threshold to paper's reported values
3. Evaluate false positive rate on benign traffic
4. Evaluate false negative rate on attack traffic (equal size to test set)
5. Apply window-based detection as specified in the paper

## Implementation Notes

The original paper does not specify:

- **Batch size**: Default Keras batch size (32) was used
- **Activation functions**: `tanh` chosen for hidden layers (standard choice for autoencoders)
- **Feature scaling**: `sklearn.preprocessing.StandardScaler` used

These choices represent reasonable defaults based on deep learning best practices.

## File Structure

```bash
anomaly-detection/
├── train.py                    # Per-device training script
├── train_og.py                 # Original centralized training
├── train_combined.py           # Combined model training
├── train_v04.py               # Latest FL implementation
├── test.py                    # Testing script
├── test_combined.py           # Combined model testing
├── run_experiment_*.py        # FL experiments
└── {device}/
    ├── benign_traffic.csv
    ├── model.h5               # Trained model
    └── logs/                  # TensorBoard logs
```

## References

Meidan, Y., Bohadana, M., Mathov, Y., Mirsky, Y., Breitenbacher, D., Shabtai, A., & Elovici, Y. (2018). N-BaIoT—Network-Based Detection of IoT Botnet Attacks Using Deep Autoencoders. IEEE Pervasive Computing, 17(3), 12-22.
