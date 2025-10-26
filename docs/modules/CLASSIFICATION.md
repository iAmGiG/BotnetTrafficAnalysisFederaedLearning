# Classification Module

This module implements multi-class neural network classification to distinguish between benign traffic and specific IoT botnet attack types (Gafgyt and Mirai).

## Overview

The classification system:

- Performs 3-class classification: benign, Gafgyt, and Mirai
- Achieves 99.98% accuracy using all 115 features
- Maintains 99.94% accuracy with only top 3 features (Fisher score selection)
- Demonstrates that feature reduction significantly decreases training time with minimal accuracy loss

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

Extract the `.rar` files into the appropriate directories:

- `data/{device_name}/gafgyt_attack/`
- `data/{device_name}/mirai_attack/`

**Dataset source**: UCI Machine Learning Repository - N-BaIoT dataset

## Training and Evaluation

### Train with All Features

```bash
python train.py
```

### Train with Top N Features

Specify the number of top features (by Fisher score) to use:

```bash
python train.py 5    # Use top 5 features
python train.py 3    # Use top 3 features
```

The trained model will be saved as `model_{N}.h5` where N is the number of features used.

## Testing

Test a previously trained model:

```bash
python test.py 5 'model_5.h5'    # Test model trained on top 5 features
```

Arguments:

1. Number of top features used during training
2. Model filename to load

## Performance Results

### All Features (115)

**Training**: 20 epochs, 42 minutes

**Metrics**:

- Loss: 0.001033
- Accuracy: 99.982%

**Confusion Matrix**:

```bash
                Predicted
              Benign  Gafgyt   Mirai
Actual Benign 111369     114       2
       Gafgyt     34  567253       7
       Mirai       9      87  733647
```

### Top 5 Features

**Training**: 5 epochs, ~8 minutes

**Metrics**:

- Loss: 0.004696
- Accuracy: 99.919%

**Confusion Matrix**:

```bash
                Predicted
              Benign  Gafgyt   Mirai
Actual Benign 111436      40       9
       Gafgyt    647  566647       0
       Mirai      63     388  733292
```

### Top 3 Features (Recommended)

**Training**: 5 epochs, ~8 minutes (99s per epoch)

**Metrics**:

- Loss: 0.003247
- Accuracy: 99.947%

**Confusion Matrix**:

```bash
                Predicted
              Benign  Gafgyt   Mirai
Actual Benign 111439      40       6
       Gafgyt    460  566834       0
       Mirai      80     162  733501
```

**Analysis**: Top 3 features provide the optimal balance between accuracy (99.947%) and training efficiency (5x faster than all features).

### Top 2 Features (Insufficient)

**Training**: 5 epochs, ~8 minutes (99s per epoch)

**Metrics**:

- Loss: 0.335397
- Accuracy: 84.302%

**Confusion Matrix**:

```bash
                Predicted
              Benign  Gafgyt   Mirai
Actual Benign  58989   52297     199
       Gafgyt    523  436574  130197
       Mirai     486   38033  695224
```

**Analysis**: Two features are insufficient for reliable classification, demonstrating the importance of feature selection threshold.

## Feature Selection

Features are ranked using Fisher score, which measures the discriminative power of each feature for classification. Fisher scores are pre-computed and stored in `../data/fisher/fisher.csv`.

## File Structure

```bash
classification/
├── train.py              # Training script with feature selection
├── test.py               # Testing/evaluation script
└── model_{N}.h5          # Trained models (N = number of features)
```

## Key Findings

1. **Feature reduction is highly effective**: 3 features achieve 99.947% accuracy vs 99.982% with all 115 features
2. **Training efficiency**: 5x faster training with minimal accuracy loss
3. **Threshold matters**: Dropping to 2 features causes significant accuracy degradation (84.3%)
4. **Practical deployment**: Top 3-5 features enable real-time classification with reduced computational requirements

## References

Meidan, Y., Bohadana, M., Mathov, Y., Mirsky, Y., Breitenbacher, D., Shabtai, A., & Elovici, Y. (2018). N-BaIoT—Network-Based Detection of IoT Botnet Attacks Using Deep Autoencoders. IEEE Pervasive Computing, 17(3), 12-22.
