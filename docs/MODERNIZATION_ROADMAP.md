# Modernization Roadmap - Main Branch Overhaul

**Status**: Planning Phase
**Target Branch**: `main`
**Last Updated**: 2025-10-26
**Related Issues**: #17, #22, #23

---

## Executive Summary

This document outlines a complete remaster and overhaul of the IoT Botnet Traffic Analysis project. 
This is not just a refactor - it's a comprehensive modernization bringing 2020-era research code to 2025 production standards.

**Key Principles**:

1. Scientific integrity first - validate all results
2. Production-ready code quality
3. Comprehensive documentation
4. Automated workflows
5. Modern Python ecosystem (3.11+, TF 2.17+)

---

## Phase 1: Foundation & Architecture Documentation

### 1.1 Architecture Visualization (Priority: HIGH)

**Goal**: Create comprehensive system diagrams using Graphviz to document current and target architectures.

#### Diagrams to Create

**A. Current System Architecture**

- Component diagram showing all modules
- Data flow diagram (dataset → preprocessing → training → evaluation)
- Module dependency graph
- File structure tree visualization

**B. Data Pipeline Architecture**

- Data download and extraction flow
- Feature engineering pipeline (115 features → Fisher selection)
- Training data preparation
- Test/validation split strategy

**C. Model Architectures**

- Autoencoder architecture (anomaly detection)
- Classifier architecture (multi-class classification)
- Federated learning simulation architecture

**D. Deployment Architecture** (Future)

- Modern production deployment
- Federated learning with Flower
- Real-time inference pipeline

#### Implementation Plan

```bash
# Location: docs/architecture/
├── diagrams/
│   ├── generate_diagrams.py      # Automated diagram generation
│   ├── system_architecture.gv    # Graphviz source
│   ├── data_pipeline.gv
│   ├── model_architecture.gv
│   └── deployment_architecture.gv
├── images/                        # Generated PNG/SVG outputs
└── ARCHITECTURE.md                # Comprehensive guide
```

**Deliverables**:

- [ ] Graphviz diagram generation script
- [ ] System architecture diagram
- [ ] Data pipeline diagram
- [ ] Model architecture diagrams (both autoencoder and classifier)
- [ ] Federated learning architecture diagram
- [ ] Comprehensive ARCHITECTURE.md document
- [ ] Integration with main README.md

**Tools**:

- Python `graphviz` library
- Automated generation in CI/CD
- SVG/PNG outputs for documentation

---

### 1.2 Project Structure Reorganization (Priority: HIGH)

**Current Issues**:

- Inconsistent naming (`anomaly-detection` vs `classification`)
- Mixed concerns (models, logs, lime outputs in module dirs)
- No clear separation of concerns
- No tests directory
- No proper package structure

**Target Structure**:

```bash
botnet-traffic-analysis-fl/
├── .github/
│   ├── workflows/
│   │   ├── ci.yaml               # Continuous Integration
│   │   ├── tests.yaml            # Automated testing
│   │   ├── lint.yaml             # Code quality checks
│   │   └── docs.yaml             # Documentation generation
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
│
├── src/                           # All source code
│   ├── __init__.py
│   ├── anomaly_detection/         # Renamed from anomaly-detection
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── autoencoder.py   # Model definition
│   │   │   └── federated_autoencoder.py
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── train.py
│   │   │   └── train_federated.py
│   │   ├── evaluation/
│   │   │   ├── __init__.py
│   │   │   └── test.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── threshold.py
│   │
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── classifier.py
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   └── train.py
│   │   ├── evaluation/
│   │   │   ├── __init__.py
│   │   │   ├── test.py
│   │   │   └── explainability.py  # LIME integration
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── metrics.py
│   │
│   ├── federated_learning/        # New module
│   │   ├── __init__.py
│   │   ├── flower/                # Flower FL implementation
│   │   │   ├── __init__.py
│   │   │   ├── server.py
│   │   │   ├── client.py
│   │   │   └── strategy.py
│   │   ├── simulation/            # Simulation framework
│   │   │   ├── __init__.py
│   │   │   └── run_simulation.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── aggregation.py
│   │
│   ├── data/                      # Data utilities
│   │   ├── __init__.py
│   │   ├── loaders/
│   │   │   ├── __init__.py
│   │   │   ├── nbaiot_loader.py
│   │   │   └── device_loader.py
│   │   ├── preprocessing/
│   │   │   ├── __init__.py
│   │   │   ├── feature_engineering.py
│   │   │   ├── normalization.py
│   │   │   └── feature_selection.py  # Fisher score
│   │   └── validation/
│   │       ├── __init__.py
│   │       └── validators.py
│   │
│   ├── common/                    # Shared utilities
│   │   ├── __init__.py
│   │   ├── config.py              # Configuration management
│   │   ├── logging.py             # Structured logging
│   │   └── paths.py               # Path management
│   │
│   └── analysis/                  # Analysis tools
│       ├── __init__.py
│       ├── overfitting_analysis.py
│       ├── data_leakage_analysis.py
│       └── visualization.py
│
├── tests/                         # Comprehensive test suite
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_models.py
│   │   ├── test_data_loaders.py
│   │   └── test_preprocessing.py
│   ├── integration/
│   │   ├── test_training_pipeline.py
│   │   └── test_evaluation_pipeline.py
│   ├── fixtures/
│   │   └── sample_data.py
│   └── conftest.py
│
├── scripts/                       # Standalone utilities
│   ├── download_data.py
│   ├── setup_environment.py
│   └── run_experiments.py
│
├── configs/                       # Configuration files
│   ├── devices.yaml              # Converted from JSON
│   ├── model_config.yaml         # Model hyperparameters
│   ├── training_config.yaml      # Training configuration
│   └── federated_config.yaml     # FL configuration
│
├── data/                          # Data directory (gitignored)
│   ├── raw/                       # Downloaded datasets
│   ├── processed/                 # Preprocessed data
│   ├── fisher/                    # Feature selection results
│   └── README.md                  # Data structure documentation
│
├── outputs/                       # All outputs (gitignored)
│   ├── models/                    # Trained models
│   ├── logs/                      # Training logs
│   ├── lime_explanations/         # Explainability outputs
│   ├── figures/                   # Generated plots
│   └── results/                   # Experiment results
│
├── notebooks/                     # Jupyter notebooks
│   ├── exploratory/
│   ├── analysis/
│   └── visualization/
│
├── docs/                          # Documentation
│   ├── architecture/              # Architecture diagrams & docs
│   │   ├── diagrams/
│   │   ├── images/
│   │   └── ARCHITECTURE.md
│   ├── api/                       # API documentation (auto-generated)
│   ├── guides/
│   │   ├── GETTING_STARTED.md
│   │   ├── TRAINING_GUIDE.md
│   │   ├── DEPLOYMENT_GUIDE.md
│   │   └── CONTRIBUTING.md
│   ├── research/
│   │   ├── OVERFITTING_ANALYSIS.md
│   │   ├── FEATURE_SELECTION.md
│   │   └── FEDERATED_LEARNING.md
│   ├── references/                # External references
│   │   ├── N_BaIoT_dataset.md
│   │   └── papers.md
│   └── archived/                  # Historical documentation
│       ├── 2020-research/
│       └── experimental/
│
├── environment.yaml               # Modern conda environment
├── requirements.txt               # Production dependencies
├── requirements-dev.txt           # Development dependencies
├── pyproject.toml                 # Python project config
├── setup.py                       # Package installation
├── pytest.ini                     # pytest configuration
├── .pylintrc                      # Linting configuration
├── .pre-commit-config.yaml        # Pre-commit hooks
├── Makefile                       # Common commands
├── README.md                      # Main documentation
└── LICENSE
```

**Migration Tasks**:

- [ ] Create new directory structure
- [ ] Migrate anomaly-detection → src/anomaly_detection
- [ ] Migrate classification → src/classification
- [ ] Create federated_learning module
- [ ] Create data utilities module
- [ ] Move all outputs to outputs/ directory
- [ ] Update all import paths
- [ ] Create **init**.py files
- [ ] Add proper package metadata

---

## Phase 2: Code Modernization

### 2.1 Dependency Updates (Priority: CRITICAL)

**Target Stack**:

- Python 3.11+ (3.12 preferred)
- TensorFlow 2.17+
- Keras 3.0+ (standalone)
- Pandas 2.2+
- NumPy 2.0+
- scikit-learn 1.5+
- Flower (flwr) 1.10+ for federated learning

**Breaking Changes to Address**:

#### A. Pandas 2.0+ Migration

```python
# OLD (deprecated)
df = df.append(other_df, ignore_index=True)

# NEW (modern)
df = pd.concat([df, other_df], ignore_index=True)
```

**Files to update**: All training and data loading scripts

#### B. Keras 3.0 Migration

```python
# OLD (mixed imports)
from keras.models import load_model
from tensorflow.keras.layers import Dense

# NEW (unified - use standalone Keras 3.0)
from keras.models import load_model
from keras.layers import Dense
```

**Files to update**: All model definitions and training scripts

#### C. NumPy 2.0 Changes

- Update array type annotations
- Fix deprecated array functions
- Update dtype handling

#### D. TensorFlow 2.17+ Updates

- Update optimizer API calls
- Fix deprecated callbacks
- Update model saving format (.keras instead of .h5)

**Deliverables**:

- [ ] Create modern environment.yaml
- [ ] Update all import statements
- [ ] Fix pandas deprecations
- [ ] Migrate to Keras 3.0
- [ ] Test all functionality
- [ ] Document migration guide

---

### 2.2 Code Quality Improvements (Priority: HIGH)

#### A. Type Hints

Add comprehensive type hints to all functions:

```python
from typing import Tuple, Optional
import numpy as np
import pandas as pd

def load_device_data(
    device_name: str,
    attack_type: str,
    feature_count: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare device data for training.

    Args:
        device_name: Name of IoT device
        attack_type: Type of attack ('mirai', 'gafgyt', or 'benign')
        feature_count: Number of top features to use (default: all)

    Returns:
        Tuple of (features DataFrame, labels Series)
    """
    ...
```

#### B. Docstrings

Add comprehensive Google-style docstrings:

```python
def train_autoencoder(
    x_train: np.ndarray,
    learning_rate: float = 0.01,
    epochs: int = 100
) -> keras.Model:
    """Train autoencoder model for anomaly detection.

    This function implements the autoencoder training pipeline for
    detecting botnet attacks in IoT network traffic.

    Args:
        x_train: Training data of shape (n_samples, n_features)
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs

    Returns:
        Trained Keras model

    Raises:
        ValueError: If x_train has invalid shape

    Examples:
        >>> x_train = load_training_data()
        >>> model = train_autoencoder(x_train, learning_rate=0.01)
        >>> model.save('outputs/models/autoencoder.keras')
    """
    ...
```

#### C. Linting and Formatting

- Use `black` for code formatting
- Use `pylint` for code quality
- Use `mypy` for type checking
- Use `isort` for import sorting

**Configuration**:

```toml
# pyproject.toml
[tool.black]
line-length = 88
target-version = ['py311']

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
```

#### D. Error Handling

Add proper exception handling:

```python
class DataLoadError(Exception):
    """Exception raised when data loading fails."""
    pass

class ModelTrainingError(Exception):
    """Exception raised during model training."""
    pass

def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise DataLoadError(f"Data file not found: {path}")
    except pd.errors.EmptyDataError:
        raise DataLoadError(f"Data file is empty: {path}")
    return df
```

**Deliverables**:

- [ ] Add type hints to all functions
- [ ] Add comprehensive docstrings
- [ ] Set up black, pylint, mypy
- [ ] Add proper error handling
- [ ] Create coding standards document

---

### 2.3 Configuration Management (Priority: MEDIUM)

**Current Issues**:

- Hard-coded paths
- Magic numbers throughout code
- No centralized configuration

**Solution**: Use Hydra or OmegaConf for configuration management

```python
# configs/config.yaml
project:
  name: "botnet-traffic-analysis"
  version: "2.0.0"

data:
  root_dir: "data"
  devices:
    - Danmini_Doorbell
    - Ecobee_Thermostat
    # ... etc

model:
  autoencoder:
    input_dim: 115
    encoding_dim: 90
    hidden_layers: [100, 90]
    activation: "relu"

  classifier:
    hidden_layers: [64, 32]
    dropout: 0.3
    activation: "relu"

training:
  batch_size: 128
  epochs: 100
  learning_rate: 0.001
  validation_split: 0.2

federated:
  num_clients: 9
  rounds: 50
  fraction_fit: 0.8
  min_fit_clients: 7
```

**Usage**:

```python
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig):
    model = build_model(
        input_dim=cfg.model.autoencoder.input_dim,
        hidden_layers=cfg.model.autoencoder.hidden_layers
    )
    ...
```

**Deliverables**:

- [ ] Convert all configs to YAML
- [ ] Implement configuration management
- [ ] Remove hard-coded values
- [ ] Create config validation
- [ ] Document configuration system

---

## Phase 3: Testing & Validation

### 3.1 Unit Testing (Priority: HIGH)

**Coverage Target**: 80%+

**Test Structure**:

```python
# tests/unit/test_data_loaders.py
import pytest
import pandas as pd
from src.data.loaders import load_device_data

class TestDataLoaders:
    """Test suite for data loading functions."""

    def test_load_device_data_valid(self, sample_device_data):
        """Test loading valid device data."""
        df, labels = load_device_data("Danmini_Doorbell", "mirai")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert df.shape[1] == 115  # All features

    def test_load_device_data_invalid_device(self):
        """Test loading data for non-existent device."""
        with pytest.raises(ValueError):
            load_device_data("NonExistent_Device", "mirai")

    def test_feature_selection(self, sample_device_data):
        """Test Fisher score feature selection."""
        df, labels = load_device_data("Danmini_Doorbell", "mirai", feature_count=10)
        assert df.shape[1] == 10
```

**Test Categories**:

- Data loading and preprocessing
- Model building and training
- Feature selection
- Metrics calculation
- Configuration loading

**Deliverables**:

- [ ] Set up pytest framework
- [ ] Create test fixtures
- [ ] Write unit tests for all modules
- [ ] Achieve 80%+ code coverage
- [ ] Set up coverage reporting

---

### 3.2 Integration Testing (Priority: MEDIUM)

Test complete pipelines:

```python
# tests/integration/test_training_pipeline.py
def test_anomaly_detection_pipeline(tmp_path):
    """Test complete anomaly detection training pipeline."""
    # Setup
    config = load_config("configs/model_config.yaml")
    data_dir = tmp_path / "data"

    # Run pipeline
    model = train_anomaly_detection(
        device="Danmini_Doorbell",
        config=config,
        output_dir=tmp_path
    )

    # Validate
    assert model is not None
    assert (tmp_path / "model.keras").exists()
    assert (tmp_path / "training_history.json").exists()
```

**Test Scenarios**:

- End-to-end training pipeline
- End-to-end evaluation pipeline
- Data preprocessing pipeline
- Federated learning simulation

**Deliverables**:

- [ ] Write integration tests
- [ ] Test all major pipelines
- [ ] Validate outputs
- [ ] Document test procedures

---

### 3.3 Overfitting Analysis (Priority: CRITICAL)

**Related**: Issue #23

This is critical for scientific integrity.

**Tests to Implement**:

1. **Learning Curves** ✅ (already implemented)
2. **Cross-Validation** ✅ (already implemented)
3. **Feature Ablation** ✅ (already implemented)
4. **Dropout Testing** ✅ (already implemented)
5. **Cross-Device Generalization** ⏳ (CRITICAL - needs implementation)
6. **Feature Perturbation** ⏳ (needs implementation)
7. **Data Scaling Tests** ⏳ (needs implementation)

**Priority Implementation**:

```python
# src/analysis/cross_device_validation.py
def cross_device_validation(model_type: str = "classifier"):
    """
    Leave-one-device-out cross-validation.

    Train on 8 devices, test on 1 held-out device.
    Repeat for all 9 devices to ensure generalization.
    """
    devices = load_device_list()
    results = []

    for held_out_device in devices:
        # Train on other 8 devices
        training_devices = [d for d in devices if d != held_out_device]
        model = train_model(devices=training_devices)

        # Test on held-out device
        accuracy = evaluate_model(model, device=held_out_device)
        results.append({
            'held_out': held_out_device,
            'accuracy': accuracy
        })

    return pd.DataFrame(results)
```

**Deliverables**:

- [ ] Implement cross-device validation
- [ ] Implement feature perturbation tests
- [ ] Implement data scaling tests
- [ ] Generate comprehensive analysis report
- [ ] Update docs/research/OVERFITTING_ANALYSIS.md

---

## Phase 4: Automation & CI/CD

### 4.1 GitHub Actions Workflows (Priority: HIGH)

#### A. Continuous Integration

```yaml
# .github/workflows/ci.yaml
name: Continuous Integration

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install black pylint mypy isort
      - name: Format check
        run: black --check src tests
      - name: Lint
        run: pylint src
      - name: Type check
        run: mypy src

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.11', '3.12']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Run tests
        run: pytest --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

#### B. Automated Testing

```yaml
# .github/workflows/tests.yaml
name: Test Suite

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly
  workflow_dispatch:

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup environment
        run: |
          conda env create -f environment.yaml
      - name: Run integration tests
        run: pytest tests/integration

  overfitting-analysis:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run overfitting analysis
        run: python src/analysis/overfitting_analysis.py
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: analysis-results
          path: outputs/analysis/
```

#### C. Documentation Generation

```yaml
# .github/workflows/docs.yaml
name: Documentation

on:
  push:
    branches: [main]

jobs:
  build-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Generate architecture diagrams
        run: python docs/architecture/diagrams/generate_diagrams.py
      - name: Build API docs
        run: |
          pip install pdoc3
          pdoc --html --output-dir docs/api src
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs
```

#### D. Release Automation

```yaml
# .github/workflows/release.yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build package
        run: python setup.py sdist bdist_wheel
      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          files: dist/*
```

**Deliverables**:

- [ ] Create CI workflow
- [ ] Create testing workflow
- [ ] Create documentation workflow
- [ ] Create release workflow
- [ ] Set up branch protection rules
- [ ] Configure GitHub secrets

---

### 4.2 Pre-commit Hooks (Priority: MEDIUM)

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.10.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 7.1.1
    hooks:
      - id: flake8
        args: ['--max-line-length=88']

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
```

**Setup**:

```bash
pip install pre-commit
pre-commit install
```

**Deliverables**:

- [ ] Create pre-commit configuration
- [ ] Document pre-commit setup
- [ ] Enforce in CI

---

### 4.3 Makefile for Common Commands (Priority: LOW)

```makefile
# Makefile
.PHONY: install test lint format clean docs

install:
 pip install -e ".[dev]"
 pre-commit install

test:
 pytest tests/ -v --cov=src --cov-report=html

test-unit:
 pytest tests/unit/ -v

test-integration:
 pytest tests/integration/ -v

lint:
 pylint src tests
 mypy src

format:
 black src tests
 isort src tests

clean:
 find . -type d -name "__pycache__" -exec rm -rf {} +
 find . -type f -name "*.pyc" -delete
 rm -rf .pytest_cache .mypy_cache .coverage htmlcov
 rm -rf dist build *.egg-info

docs:
 python docs/architecture/diagrams/generate_diagrams.py
 pdoc --html --output-dir docs/api src

train-autoencoder:
 python src/anomaly_detection/training/train.py

train-classifier:
 python src/classification/training/train.py

analysis:
 python src/analysis/overfitting_analysis.py
```

**Deliverables**:

- [ ] Create Makefile
- [ ] Document all commands
- [ ] Add to README

---

## Phase 5: Federated Learning Modernization

### 5.1 Replace TensorFlow Federated with Flower (Priority: HIGH)

**Related**: Issue #22

**Why Flower**:

- Modern, actively maintained
- Better documentation
- Framework-agnostic (works with TensorFlow, PyTorch)
- Easier to use than TFF
- Production-ready deployment

**Architecture**:

```python
# src/federated_learning/flower/client.py
import flwr as fl
from typing import Dict, List, Tuple
import numpy as np

class BotnetClient(fl.client.NumPyClient):
    """Flower client for IoT device training."""

    def __init__(self, device_name: str, model: keras.Model):
        self.device_name = device_name
        self.model = model
        self.x_train, self.y_train = self.load_device_data()

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return current model parameters."""
        return self.model.get_weights()

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """Train model on device data."""
        self.model.set_weights(parameters)
        history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=config["local_epochs"],
            batch_size=config["batch_size"],
            verbose=0
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """Evaluate model on device data."""
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}
```

```python
# src/federated_learning/flower/server.py
import flwr as fl
from typing import Dict, List, Tuple, Optional

class BotnetStrategy(fl.server.strategy.FedAvg):
    """Custom federated averaging strategy."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException]
    ) -> Tuple[Optional[fl.common.Parameters], Dict]:
        """Aggregate model weights using weighted average."""
        # Custom aggregation logic
        aggregated_weights = super().aggregate_fit(server_round, results, failures)

        # Log metrics
        print(f"Round {server_round} complete")

        return aggregated_weights

def start_server(num_rounds: int = 50):
    """Start Flower server."""
    strategy = BotnetStrategy(
        fraction_fit=0.8,
        min_fit_clients=7,
        min_available_clients=9
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )
```

```python
# src/federated_learning/simulation/run_simulation.py
import flwr as fl
from src.federated_learning.flower.client import BotnetClient

def run_simulation(num_rounds: int = 50):
    """Run federated learning simulation."""

    # Create clients for each IoT device
    def client_fn(cid: str) -> BotnetClient:
        device_name = DEVICES[int(cid)]
        model = build_model()
        return BotnetClient(device_name, model)

    # Run simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=9,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=BotnetStrategy()
    )
```

**Deliverables**:

- [ ] Implement Flower client
- [ ] Implement Flower server
- [ ] Implement custom aggregation strategy
- [ ] Create simulation framework
- [ ] Compare results: centralized vs federated
- [ ] Document FL implementation
- [ ] Create deployment guide

---

### 5.2 Federated Learning Research (Priority: MEDIUM)

**Research Questions**:

1. Does federated learning maintain accuracy vs centralized?
2. Can we achieve device-specific models?
3. What is the communication cost?
4. How does it perform with heterogeneous devices?

**Experiments to Run**:

1. **Baseline Comparison**
   - Centralized training (all devices combined)
   - Local training (per-device models)
   - Federated training (Flower)

2. **Personalization**
   - Personalized FL (device-specific fine-tuning)
   - Federated multi-task learning

3. **Communication Efficiency**
   - Measure communication rounds
   - Implement gradient compression
   - Implement sparse updates

**Deliverables**:

- [ ] Design experiments
- [ ] Implement baseline comparisons
- [ ] Run federated experiments
- [ ] Analyze results
- [ ] Write research report (docs/research/FEDERATED_LEARNING.md)
- [ ] Consider publication

---

## Phase 6: Documentation & User Experience

### 6.1 Comprehensive Documentation (Priority: HIGH)

**Documentation Structure**:

```bash
docs/
├── README.md                      # Documentation index
├── GETTING_STARTED.md             # Quick start guide
├── ARCHITECTURE.md                # System architecture
├── API_REFERENCE.md               # API documentation
├── guides/
│   ├── INSTALLATION.md            # Installation guide
│   ├── TRAINING_GUIDE.md          # Training guide
│   ├── EVALUATION_GUIDE.md        # Evaluation guide
│   ├── DEPLOYMENT_GUIDE.md        # Deployment guide
│   └── CONTRIBUTING.md            # Contribution guidelines
├── research/
│   ├── OVERFITTING_ANALYSIS.md    # Overfitting analysis results
│   ├── FEATURE_SELECTION.md       # Fisher score analysis
│   ├── FEDERATED_LEARNING.md      # FL research results
│   └── EXPLAINABILITY.md          # LIME interpretability
├── architecture/
│   ├── ARCHITECTURE.md            # Detailed architecture
│   ├── DATA_PIPELINE.md           # Data pipeline details
│   └── diagrams/                  # Architecture diagrams
└── tutorials/
    ├── 01_data_preparation.md
    ├── 02_training_models.md
    ├── 03_evaluation.md
    └── 04_federated_learning.md
```

**Key Documents to Create**:

#### A. GETTING_STARTED.md

- Installation instructions
- Quick start examples
- Common use cases
- Troubleshooting

#### B. ARCHITECTURE.md

- System overview
- Component descriptions
- Data flow
- Architecture diagrams
- Design decisions

#### C. API_REFERENCE.md

- Auto-generated from docstrings
- Module reference
- Function signatures
- Examples

#### D. Training Guides

- Anomaly detection training
- Classification training
- Federated learning training
- Hyperparameter tuning

**Deliverables**:

- [ ] Write GETTING_STARTED.md
- [ ] Write ARCHITECTURE.md
- [ ] Generate API documentation
- [ ] Write training guides
- [ ] Write deployment guide
- [ ] Create tutorials
- [ ] Update main README.md

---

### 6.2 Naming Convention Standardization (Priority: MEDIUM)

**Current Issues**:

- Inconsistent module names (hyphens vs underscores)
- Inconsistent function names
- Unclear variable names

**Standards to Implement**:

#### A. Module Naming

```python
# Use snake_case for all modules
anomaly_detection  # NOT anomaly-detection
classification
federated_learning
```

#### B. Function Naming

```python
# Use descriptive snake_case
def load_device_data() -> pd.DataFrame:
    ...

def train_autoencoder_model() -> keras.Model:
    ...

def evaluate_model_performance() -> Dict[str, float]:
    ...
```

#### C. Class Naming

```python
# Use PascalCase
class DataLoader:
    ...

class AutoencoderModel:
    ...

class FederatedClient:
    ...
```

#### D. Variable Naming

```python
# Use descriptive names
device_name = "Danmini_Doorbell"  # NOT dn or device
feature_count = 115               # NOT fc or n_features
learning_rate = 0.001            # NOT lr
```

#### E. Constant Naming

```python
# Use UPPER_SNAKE_CASE
MAX_EPOCHS = 100
DEFAULT_BATCH_SIZE = 128
NUM_DEVICES = 9
```

**Deliverables**:

- [ ] Create CODING_STANDARDS.md
- [ ] Refactor all code to follow standards
- [ ] Set up linter to enforce standards
- [ ] Document naming conventions

---

### 6.3 Logging & Monitoring (Priority: MEDIUM)

**Current Issues**:

- Print statements everywhere
- No structured logging
- No experiment tracking

**Solution**: Implement structured logging and experiment tracking

```python
# src/common/logging.py
import logging
import json
from pathlib import Path
from datetime import datetime

class ExperimentLogger:
    """Structured logging for experiments."""

    def __init__(self, experiment_name: str, output_dir: Path):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.start_time = datetime.now()

        # Setup file logging
        self.logger = logging.getLogger(experiment_name)
        handler = logging.FileHandler(output_dir / "experiment.log")
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # Metrics storage
        self.metrics = {}

    def log_parameters(self, params: Dict):
        """Log experiment parameters."""
        self.logger.info(f"Parameters: {json.dumps(params, indent=2)}")
        self.metrics['parameters'] = params

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric value."""
        self.logger.info(f"Metric {name}: {value} (step {step})")
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({'value': value, 'step': step})

    def log_model(self, model: keras.Model, name: str):
        """Save and log model."""
        model_path = self.output_dir / f"{name}.keras"
        model.save(model_path)
        self.logger.info(f"Model saved: {model_path}")

    def finish(self):
        """Finish experiment and save results."""
        duration = (datetime.now() - self.start_time).total_seconds()
        self.metrics['duration_seconds'] = duration

        # Save metrics
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
```

**Usage**:

```python
logger = ExperimentLogger("anomaly_detection_exp1", Path("outputs/experiments/exp1"))

logger.log_parameters({
    'device': 'Danmini_Doorbell',
    'learning_rate': 0.01,
    'epochs': 100
})

for epoch in range(epochs):
    loss = train_epoch()
    logger.log_metric('loss', loss, step=epoch)

logger.log_model(model, 'autoencoder')
logger.finish()
```

**Integration with MLflow (Optional)**:

```python
import mlflow

with mlflow.start_run(run_name="anomaly_detection"):
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.keras.log_model(model, "model")
```

**Deliverables**:

- [ ] Implement ExperimentLogger
- [ ] Replace all print statements
- [ ] Add structured logging to all scripts
- [ ] Optional: Integrate MLflow
- [ ] Create logging guide

---

## Phase 7: Performance & Optimization

### 7.1 Performance Profiling (Priority: LOW)

**Tools**:

- cProfile for CPU profiling
- memory_profiler for memory usage
- TensorBoard for training metrics

```python
# Profile training
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run training
train_model()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

**Deliverables**:

- [ ] Profile data loading
- [ ] Profile training pipelines
- [ ] Identify bottlenecks
- [ ] Optimize critical paths
- [ ] Document optimizations

---

### 7.2 Data Pipeline Optimization (Priority: LOW)

**Optimizations**:

1. **Use tf.data.Dataset**:

```python
def create_dataset(x, y, batch_size=128):
    """Create optimized TensorFlow dataset."""
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.cache()  # Cache after expensive operations
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch next batch
    return dataset
```

2. **Parallel Data Loading**:

```python
from concurrent.futures import ThreadPoolExecutor

def load_all_devices_parallel(devices: List[str]) -> pd.DataFrame:
    """Load multiple devices in parallel."""
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(load_device_data, d) for d in devices]
        results = [f.result() for f in futures]
    return pd.concat(results)
```

3. **Efficient Feature Selection**:

```python
# Cache Fisher scores
@lru_cache(maxsize=1)
def get_top_features(n: int) -> List[str]:
    """Get top N features (cached)."""
    fisher_scores = pd.read_csv("data/fisher/fisher.csv")
    return fisher_scores.nlargest(n, 'score')['feature'].tolist()
```

**Deliverables**:

- [ ] Optimize data loading
- [ ] Use tf.data.Dataset
- [ ] Implement caching
- [ ] Parallelize where possible
- [ ] Measure improvements

---

## Implementation Timeline

### Sprint 1 (Weeks 1-2): Foundation

- [ ] Phase 1.1: Architecture diagrams
- [ ] Phase 1.2: Directory restructuring
- [ ] Phase 2.1: Dependency updates
- [ ] Set up development environment

### Sprint 2 (Weeks 3-4): Code Quality

- [ ] Phase 2.2: Code quality improvements
- [ ] Phase 2.3: Configuration management
- [ ] Phase 3.1: Unit testing setup
- [ ] Phase 4.2: Pre-commit hooks

### Sprint 3 (Weeks 5-6): Testing & Validation

- [ ] Phase 3.1: Unit tests (complete)
- [ ] Phase 3.2: Integration tests
- [ ] Phase 3.3: Overfitting analysis (complete)
- [ ] Validate all results

### Sprint 4 (Weeks 7-8): Automation

- [ ] Phase 4.1: GitHub Actions (all workflows)
- [ ] Phase 4.3: Makefile
- [ ] Set up CI/CD completely
- [ ] Test automation

### Sprint 5 (Weeks 9-10): Federated Learning

- [ ] Phase 5.1: Implement Flower FL
- [ ] Phase 5.2: FL experiments
- [ ] Compare centralized vs federated
- [ ] Document FL findings

### Sprint 6 (Weeks 11-12): Documentation & Polish

- [ ] Phase 6.1: Complete documentation
- [ ] Phase 6.2: Naming standardization
- [ ] Phase 6.3: Logging & monitoring
- [ ] Final testing and validation

### Sprint 7 (Weeks 13-14): Optimization & Release

- [ ] Phase 7.1: Performance profiling
- [ ] Phase 7.2: Data pipeline optimization
- [ ] Final polishing
- [ ] Release v2.0.0

---

## Success Metrics

### Code Quality

- [ ] 80%+ test coverage
- [ ] 0 linting errors
- [ ] Type hints on all functions
- [ ] Comprehensive docstrings

### Scientific Integrity

- [ ] Overfitting analysis complete
- [ ] Cross-device validation passed
- [ ] Results reproducible
- [ ] All findings documented

### Automation

- [ ] CI/CD passing
- [ ] Automated testing
- [ ] Automated documentation
- [ ] Pre-commit hooks working

### Documentation

- [ ] Complete API documentation
- [ ] User guides for all features
- [ ] Architecture diagrams
- [ ] Contributing guidelines

### Federated Learning

- [ ] Flower FL implemented
- [ ] Experiments completed
- [ ] Results documented
- [ ] Comparison with centralized

---

## GitHub Issues to Create/Update

### New Issues to Create

1. **Architecture Visualization** (#24)
   - Create Graphviz diagrams
   - Generate system architecture
   - Data pipeline visualization
   - Model architecture diagrams

2. **Directory Restructuring** (#25)
   - Migrate to src/ layout
   - Separate concerns (models, training, evaluation)
   - Create proper package structure
   - Update all imports

3. **Dependency Modernization** (#26)
   - Python 3.11+
   - TensorFlow 2.17+
   - Pandas 2.2+, NumPy 2.0+
   - Keras 3.0

4. **Code Quality Improvements** (#27)
   - Add type hints
   - Add docstrings
   - Set up linting (black, pylint, mypy)
   - Error handling

5. **Configuration Management** (#28)
   - Migrate to YAML configs
   - Implement Hydra/OmegaConf
   - Remove hard-coded values

6. **Unit Testing Suite** (#29)
   - Set up pytest
   - Write unit tests
   - Achieve 80%+ coverage

7. **Integration Testing** (#30)
   - Test complete pipelines
   - End-to-end testing

8. **GitHub Actions CI/CD** (#31)
   - CI workflow
   - Testing workflow
   - Documentation workflow
   - Release workflow

9. **Flower Federated Learning** (#32)
   - Implement Flower client
   - Implement Flower server
   - Run experiments
   - Compare with centralized

10. **Documentation Overhaul** (#33)
    - Getting started guide
    - Architecture documentation
    - API reference
    - Training guides

11. **Logging & Monitoring** (#34)
    - Structured logging
    - Experiment tracking
    - MLflow integration (optional)

12. **Performance Optimization** (#35)
    - Profile code
    - Optimize data pipeline
    - Measure improvements

### Existing Issues to Update

- **#23**: Update with overfitting analysis progress
- **#22**: Convert to Flower implementation plan
- **#17**: Reference this roadmap document
- **#16**: Plan to address in testing phase

---

## Notes & Considerations

### Timeline Flexibility

This is an aggressive 14-week timeline. Adjust based on:

- Availability
- Complexity encountered
- Additional requirements

### Priorities

If time is limited, focus on:

1. **Critical**: Phases 1, 2, 3.3 (architecture, modernization, validation)
2. **High**: Phases 3, 4, 5.1 (testing, automation, FL)
3. **Medium**: Phases 6, 5.2 (documentation, FL research)
4. **Low**: Phase 7 (optimization)

### Incremental Approach

- Work in feature branches
- Frequent PRs
- Keep main branch always working
- Document as you go

### Research vs Engineering

Balance research goals (FL experiments, overfitting analysis) with engineering goals (code quality, testing, documentation).

### Portfolio Value

This overhaul transforms a 2020 research project into a 2025 production-ready system, demonstrating:

- Scientific rigor
- Software engineering best practices
- Modern ML engineering skills
- Documentation and communication skills

---

## Next Steps

1. Review this roadmap
2. Create GitHub issues (use this document as reference)
3. Set up project board for tracking
4. Start with Sprint 1: Architecture & Foundation
5. Work iteratively, adjust as needed

---

**Document Status**: Draft v1.0
**Next Review**: After Sprint 1 completion
**Feedback**: Create issues or PRs for roadmap updates
