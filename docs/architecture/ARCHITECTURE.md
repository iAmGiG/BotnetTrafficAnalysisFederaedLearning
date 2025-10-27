# System Architecture

**Project**: IoT Botnet Traffic Analysis with Federated Learning
**Version**: 2.0 (Modernization)
**Last Updated**: 2025-10-26

---

## Table of Contents

1. [Overview](#overview)
2. [System Components](#system-components)
3. [Data Architecture](#data-architecture)
4. [Model Architectures](#model-architectures)
5. [Federated Learning Architecture](#federated-learning-architecture)
6. [Deployment Architecture](#deployment-architecture)
7. [Technology Stack](#technology-stack)

---

## Overview

This system implements a comprehensive IoT botnet detection solution using deep learning and federated learning techniques. It consists of three main approaches:

1. **Anomaly Detection** - Autoencoder-based unsupervised learning
2. **Classification** - Supervised multi-class neural network
3. **Federated Learning** - Distributed learning across IoT devices

### High-Level Architecture

```
┌─────────────┐
│   Dataset   │  N-BaIoT (9 IoT Devices)
│  (UCI ML)   │  - Benign traffic
└──────┬──────┘  - Mirai attacks
       │         - Gafgyt attacks
       ▼
┌─────────────┐
│Preprocessing│  - Feature extraction (115 features)
│   Pipeline  │  - Fisher score selection
└──────┬──────┘  - Normalization
       │
       ├──────────┬──────────┐
       ▼          ▼          ▼
   ┌───────┐  ┌───────┐  ┌───────┐
   │Anomaly│  │Classi-│  │Feder- │
   │Detect │  │  fier │  │ ated  │
   └───┬───┘  └───┬───┘  └───┬───┘
       │          │          │
       └──────────┴──────────┘
                  ▼
       ┌──────────────────┐
       │   Evaluation &   │
       │  Explainability  │
       └──────────────────┘
```

See [diagrams/images/](diagrams/images/) for detailed visualizations.

---

## System Components

### 1. Data Layer

#### Data Sources

- **N-BaIoT Dataset** (UCI Machine Learning Repository)
  - 9 commercial IoT devices
  - Network traffic captures
  - Labeled benign and malicious traffic

#### Data Storage

```bash
data/
├── raw/                     # Downloaded .rar archives
│   ├── Danmini_Doorbell/
│   ├── Ecobee_Thermostat/
│   └── ... (7 more devices)
├── processed/               # Preprocessed CSV files
│   ├── combined/
│   └── per_device/
└── fisher/                  # Feature selection
    ├── fisher.csv          # All features ranked
    └── top_features_fisherscore.csv
```

#### Key Features

- **115 statistical features** extracted using AfterImage framework
- Features include:
  - Packet statistics (mean, variance, std dev)
  - Jitter calculations
  - Connection patterns
  - Protocol distributions

### 2. Preprocessing Layer

#### Components

**a. Data Loader** (`src/data/loaders/`)

- Loads device-specific CSV files
- Combines multiple devices
- Handles missing data
- Manages memory efficiently

**b. Feature Engineering** (`src/data/preprocessing/`)

- Extracts 115 statistical features
- Temporal windowing
- Stream-based computation

**c. Feature Selection** (`src/data/preprocessing/feature_selection.py`)

- Fisher score calculation
- Top-N feature selection
- Dimensionality reduction
- Performance: 3 features achieve 99.94% accuracy

**d. Normalization** (`src/data/preprocessing/normalization.py`)

- StandardScaler (zero mean, unit variance)
- Fit on training data only (no leakage)
- Transform train/validation/test consistently

#### Data Pipeline Flow

```bash
Raw CSV → Load → Combine → Label → Feature Select → Split → Scale → NumPy Arrays
```

**Critical**: Scaler must be fit ONLY on training data to prevent data leakage.

### 3. Model Layer

Three distinct model architectures:

#### A. Anomaly Detection (Autoencoder)

**Purpose**: Detect anomalous (malicious) traffic patterns

**Architecture**:

```bash
Input (115) → Dense(100, ReLU) → Dense(90, ReLU) → Bottleneck(90)
                                                         ↓
Output (115) ← Dense(100, ReLU) ← [Reconstruction Loss]
```

**Training**:

- Loss: Mean Squared Error (MSE)
- Optimizer: Adam
- Trained on benign traffic only
- Learns to reconstruct normal patterns

**Inference**:

- Reconstruction error > threshold → Anomaly (malicious)
- Threshold determined per-device using validation set

**Files**: `src/anomaly_detection/`

#### B. Classification (Neural Network)

**Purpose**: Classify traffic into benign/Mirai/Gafgyt

**Architecture**:

```bash
Input (3-115) → Dense(64, ReLU) → Dropout(0.3) →
                Dense(32, ReLU) → Dropout(0.3) →
                Output(3, Softmax)
```

**Training**:

- Loss: Categorical Crossentropy
- Optimizer: Adam
- Classes: [Benign, Mirai, Gafgyt]
- Results: 99.85% accuracy (validated)

**Feature Analysis**:

| Features | Accuracy | Notes |
|----------|----------|-------|
| 115 (all) | 99.98% | Original (data leakage) |
| 115 (fixed) | 99.85% | After fixing leakage |
| 3 (top) | 99.94% | Optimal balance |

**Files**: `src/classification/`

#### C. Federated Learning (Flower)

**Purpose**: Train models in a privacy-preserving distributed manner

**Architecture**:

```bash
Server (Aggregation)
  ↓ Broadcast Model
Client 1 (Device 1) →
Client 2 (Device 2) → FedAvg → Updated Global Model
...
Client 9 (Device 9) →
  ↑ Send Updates
```

**Components**:

- **Server**: Orchestrates training rounds, aggregates updates
- **Clients**: 9 IoT devices, each with local data
- **Strategy**: FedAvg (weighted average based on data size)

**Files**: `src/federated_learning/`

### 4. Evaluation Layer

#### Metrics (`src/*/evaluation/metrics.py`)

- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC

#### Explainability (`src/classification/evaluation/explainability.py`)

- LIME (Local Interpretable Model-agnostic Explanations)
- Generates HTML reports
- Shows feature importance per prediction

#### Analysis (`src/analysis/`)

- Overfitting analysis
- Cross-device validation
- Feature importance studies
- Data leakage detection

### 5. Output Layer

```bash
outputs/
├── models/                  # Trained models (.keras format)
│   ├── autoencoder_device1.keras
│   ├── classifier_3features.keras
│   └── federated_global.keras
├── logs/                    # Training logs (TensorBoard)
│   ├── autoencoder/
│   ├── classifier/
│   └── federated/
├── lime_explanations/       # LIME HTML reports
├── figures/                 # Generated plots
│   ├── confusion_matrix.png
│   ├── learning_curves.png
│   └── feature_importance.png
└── results/                 # Experiment results (JSON)
    └── experiment_metrics.json
```

---

## Data Architecture

### Dataset Specification

**Source**: N-BaIoT (Network-based Detection of IoT Botnet Attacks)
**Citation**: Meidan et al., IEEE Pervasive Computing, 2018

**Devices (9)**:

1. Danmini Doorbell
2. Ecobee Thermostat
3. Ennio Doorbell
4. Philips B120N/10 Baby Monitor
5. Provision PT-737E Security Camera
6. Provision PT-838 Security Camera
7. Samsung SNH 1011 N Webcam
8. SimpleHome XCS7-1002 WHT Security Camera
9. SimpleHome XCS7-1003 WHT Security Camera

**Attack Types**:

- **Mirai**: TCP flood, UDP flood, ACK flood, HTTP flood
- **Gafgyt** (BASHLITE): Various DDoS vectors

**Traffic Characteristics**:

- Benign: Normal device operation (IoT protocols, updates)
- Malicious: Botnet command-and-control + attack traffic

### Data Pipeline Details

#### 1. Download Phase

```python
# scripts/download_data.py
def download_device_data(device_name: str):
    """
    Downloads .rar archives from UCI repository
    Structure: {device}/benign.csv, {device}/gafgyt/*.csv, {device}/mirai/*.csv
    """
```

#### 2. Loading Phase

```python
# src/data/loaders/nbaiot_loader.py
def load_device_data(
    device: str,
    attack_type: str,
    feature_count: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads CSV files for specific device and attack type
    Returns: (features, labels)
    """
```

#### 3. Feature Engineering Phase

```python
# Already done by N-BaIoT dataset
# 115 features extracted using AfterImage
# Damped incremental statistics over network streams
```

#### 4. Feature Selection Phase

```python
# src/data/preprocessing/feature_selection.py
def select_top_features(n: int) -> List[str]:
    """
    Uses pre-computed Fisher scores
    Returns top N most discriminative features
    """
```

#### 5. Normalization Phase

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)  # ONLY training data
x_train_scaled = scaler.transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)
```

### Data Splits

**Standard Split**:

- Training: 70%
- Validation: 15%
- Test: 15%

**Federated Split**:

- Per-device splits (each device is a client)
- Each client has own train/val/test

**Cross-Device Split**:

- Train on 8 devices
- Test on 1 held-out device
- Repeat for all 9 devices (leave-one-out)

---

## Model Architectures

### Autoencoder (Anomaly Detection)

#### Architecture Details

```python
# src/anomaly_detection/models/autoencoder.py

def build_autoencoder(input_dim: int = 115) -> keras.Model:
    """
    Builds autoencoder model for anomaly detection.
    """
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(100, activation='relu')(input_layer)
    encoder = Dense(90, activation='relu')(encoder)

    # Bottleneck
    bottleneck = Dense(90, activation='relu')(encoder)

    # Decoder
    decoder = Dense(100, activation='relu')(bottleneck)
    output_layer = Dense(input_dim, activation='linear')(decoder)

    # Model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')

    return model
```

#### Training Process

1. **Data Preparation**: Use benign traffic only
2. **Training**: Model learns to reconstruct normal patterns
3. **Threshold Calculation**: MSE on validation benign + malicious
4. **Inference**: If MSE > threshold → Anomaly

#### Threshold Determination

```python
def calculate_threshold(model, x_benign_val, x_malicious_val):
    """
    Calculate optimal threshold using validation set.
    """
    benign_mse = np.mean((model.predict(x_benign_val) - x_benign_val)**2, axis=1)
    malicious_mse = np.mean((model.predict(x_malicious_val) - x_malicious_val)**2, axis=1)

    # Threshold: point that maximizes separation
    threshold = np.percentile(benign_mse, 95)  # 95th percentile of benign errors
    return threshold
```

### Classifier (Multi-class)

#### Architecture Details

```python
# src/classification/models/classifier.py

def build_classifier(
    input_dim: int,
    num_classes: int = 3
) -> keras.Model:
    """
    Builds multi-class classifier.
    """
    model = keras.Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
```

#### Training Process

1. **Data Preparation**: All traffic (benign + malicious)
2. **Label Encoding**: [Benign: 0, Mirai: 1, Gafgyt: 2]
3. **Training**: Supervised learning with categorical crossentropy
4. **Evaluation**: Multi-class metrics

#### Hyperparameters

```yaml
# configs/model_config.yaml
classifier:
  architecture:
    hidden_layers: [64, 32]
    dropout_rate: 0.3
    activation: relu
    output_activation: softmax

  training:
    batch_size: 128
    epochs: 20
    learning_rate: 0.001
    optimizer: adam

  callbacks:
    early_stopping:
      monitor: val_loss
      patience: 5
    reduce_lr:
      monitor: val_loss
      factor: 0.2
      patience: 3
```

---

## Federated Learning Architecture

### Flower Framework

**Why Flower?**:

- Modern, actively maintained
- Framework-agnostic (TF, PyTorch)
- Better documentation than TFF
- Production-ready

### Components

#### Server

```python
# src/federated_learning/flower/server.py

class BotnetStrategy(fl.server.strategy.FedAvg):
    """Custom federated averaging strategy."""

    def aggregate_fit(self, server_round, results, failures):
        """
        Aggregate model updates from clients.
        Weighted by number of samples.
        """
        # Weights from each client
        weights = [r[1].parameters for r in results]
        num_samples = [r[1].num_examples for r in results]

        # Weighted average
        aggregated = weighted_average(weights, num_samples)

        return aggregated
```

#### Client

```python
# src/federated_learning/flower/client.py

class DeviceClient(fl.client.NumPyClient):
    """Flower client for IoT device."""

    def __init__(self, device_name: str, model: keras.Model):
        self.device_name = device_name
        self.model = model
        self.x_train, self.y_train = load_device_data(device_name)

    def get_parameters(self, config):
        """Return current model parameters."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Train on local device data."""
        self.model.set_weights(parameters)
        history = self.model.fit(
            self.x_train, self.y_train,
            epochs=config["local_epochs"],
            batch_size=config["batch_size"]
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        """Evaluate on local test data."""
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}
```

### Training Process

1. **Initialization**: Server creates global model
2. **Broadcast**: Send global model to clients
3. **Local Training**: Each client trains on local data
4. **Aggregation**: Server aggregates updates (FedAvg)
5. **Update**: Global model updated
6. **Repeat**: For N rounds

### Communication Protocol

```
Round 1:
  Server → Client 1: global_model_weights
  Server → Client 2: global_model_weights
  ...
  Client 1 → Server: updated_weights (5000 samples)
  Client 2 → Server: updated_weights (3000 samples)
  ...
  Server: aggregate(weights, num_samples)

Round 2:
  [Repeat]
```

### Privacy Considerations

- **Data Privacy**: Client data never leaves device
- **Model Privacy**: Only model updates shared
- **Differential Privacy** (future): Add noise to updates

---

## Deployment Architecture

### Production Inference Pipeline

```
IoT Device Traffic → Feature Extraction → Normalization → Model Serving → Prediction
                                                              ↓
                                                    [Threshold Check]
                                                              ↓
                                              ┌──────────────┴──────────────┐
                                              ▼                             ▼
                                          Benign                       Malicious
                                        (Normal)                    (Generate Alert)
```

### Model Serving

**TensorFlow Serving**:

```bash
docker run -p 8501:8501 \
  --mount type=bind,source=/models/classifier,target=/models/classifier \
  -e MODEL_NAME=classifier \
  tensorflow/serving
```

**REST API**:

```python
import requests

# Prepare features
features = extract_features(traffic_data)

# Make prediction
response = requests.post(
    'http://localhost:8501/v1/models/classifier:predict',
    json={'instances': [features]}
)

prediction = response.json()['predictions'][0]
```

### Monitoring

**Metrics to Track**:

- Prediction latency
- Model accuracy (online)
- False positive rate
- False negative rate
- System throughput

**Tools**:

- Prometheus (metrics collection)
- Grafana (visualization)
- CloudWatch/ELK (logging)

---

## Technology Stack

### Current (2020-era)

- Python 3.8
- TensorFlow 2.10
- Pandas 1.3.5
- NumPy 1.21

### Target (Modern)

- Python 3.11+
- TensorFlow 2.17+
- Keras 3.0+
- Pandas 2.2+
- NumPy 2.0+
- Flower 1.10+

### Development Tools

- pytest (testing)
- black (formatting)
- pylint (linting)
- mypy (type checking)
- pre-commit (git hooks)

### Infrastructure

- Docker (containerization)
- GitHub Actions (CI/CD)
- TensorBoard (training visualization)
- MLflow (experiment tracking)

---

## Design Decisions

### Why Autoencoder for Anomaly Detection?

**Advantages**:

- Unsupervised: No need for labeled anomaly data
- Generalization: Can detect new attack types
- Interpretability: Reconstruction error is intuitive

**Disadvantages**:

- Threshold tuning required
- May miss subtle anomalies
- Requires sufficient benign data

### Why Fisher Score for Feature Selection?

**Advantages**:

- Fast computation
- Univariate (independent features)
- Interpretable (statistical measure)

**Formula**:

```
Fisher(f) = (μ₁ - μ₂)² / (σ₁² + σ₂²)
```

Where:

- μ₁, μ₂: Class means
- σ₁², σ₂²: Class variances

### Why Flower for Federated Learning?

**Advantages**:

- Modern framework (2020+)
- Better docs than TFF
- Production-ready
- Active community

**Disadvantages**:

- Less research adoption than TFF
- Fewer examples for complex scenarios

---

## Performance Characteristics

### Anomaly Detection

- Training time: ~5-10 min per device (CPU)
- Inference: < 1ms per sample
- Memory: ~100 MB model size

### Classification

- Training time: ~40 min (all features, 20 epochs, CPU)
- Inference: < 1ms per sample
- Memory: ~50 MB model size

### Federated Learning

- Training time: Depends on rounds and clients
- Communication: ~10 MB per round (9 clients)
- Convergence: 20-50 rounds typical

---

## Future Enhancements

1. **Real-time Inference**: Stream processing with Apache Kafka
2. **Distributed Training**: Multi-GPU training
3. **Model Compression**: Pruning and quantization
4. **Advanced FL**: Personalization, differential privacy
5. **Deployment**: Kubernetes orchestration

---

## References

1. N-BaIoT Dataset: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT)
2. Meidan et al., "N-BaIoT", IEEE Pervasive Computing, 2018
3. Flower Documentation: [flower.dev](https://flower.dev/docs/)
4. LIME: Ribeiro et al., "Why Should I Trust You?", KDD 2016

---

**Document Version**: 1.0
**Last Updated**: 2025-10-26
**Maintained By**: Project Team
