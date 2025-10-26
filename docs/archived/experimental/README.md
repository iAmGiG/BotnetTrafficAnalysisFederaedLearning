# Experimental Federated Learning Attempts (2020)

This folder contains experimental code from the 2020 research phase attempting to implement federated learning for IoT botnet detection. These files represent the evolution and various dead-end approaches tried during development.

## Context

The primary goal was to convert a centralized deep learning approach (anomaly detection using autoencoders) into a federated learning setup using TensorFlow Federated (TFF). Multiple iterations were attempted with varying degrees of success.

## Files Overview

### `tff_initial_attempt.py` (formerly `train_v00.py`) - Initial TFF Simulation Attempt

**Date**: ~May 2020
**Status**: Experimental / Non-functional
**Approach**: First attempt at using TensorFlow Federated's simulation API

**What it tried**:

- Basic TFF client-server broadcasting model
- Attempted to use `tff.learning.build_federated_averaging_process()`
- Struggled with data format conversion for TFF compatibility

**Why it was abandoned**:

- Complexity of TFF API for the use case
- Data preprocessing challenges for federated context
- Simulation-only approach didn't align with intended deployment

---

### `manual_fedavg_attempt.py` (formerly `train_v01.py`) - Simplified Federation Approach

**Date**: ~May 2020
**Status**: Partial implementation
**Approach**: Attempted manual implementation of FedAvg without full TFF framework

**What it tried**:

- Manual implementation of Federated Averaging (FedAvg)
- Simpler client-server architecture
- Per-device model training with aggregation

**Why it was abandoned**:

- Reinventing the wheel (duplicating TFF functionality)
- Aggregation logic became complex
- Moved back to TFF approach in later versions

---

### `pysyft_exploration.py` (formerly `train_v02.py`) - PySyft Exploration

**Date**: ~May-June 2020
**Status**: Non-functional / Abandoned
**Approach**: Attempted to use PySyft library for federated learning

**What it tried**:

- Import `syft` and `syft_tensorflow`
- TensorFlow + PySyft integration
- Privacy-preserving federated learning

**Why it was abandoned**:

- PySyft-TensorFlow integration was immature in 2020
- Documentation and examples were scarce
- API instability and compatibility issues
- See `../../PYSYFT_RESEARCH.md` for full analysis

**Note**: This file was later renamed from `train_syft.py` during refactoring.

---

### `tff_refinement.py` (formerly `train_v03.py`) - TFF Refinement Attempt

**Date**: ~June 2020
**Status**: Partial implementation
**Approach**: Refined TFF approach with better data handling

**What it tried**:

- Improved TFF client data preparation
- Better separation of client datasets
- Attempted to fix TFF type signature issues

**Why it was abandoned**:

- Still struggled with TFF complexity
- Simulation limitations became apparent
- Performance issues with multiple clients

---

### `tff_reference_example.py` (formerly `simple_fedavg_test_reviewing_edit.py`) - Reference Implementation

**Date**: June 2020
**Status**: Reference code (pulled from TFF examples)
**Approach**: Unmodified/lightly modified example from TensorFlow Federated research section

**What it is**:

- Clean FedAvg implementation from TFF experimental examples
- Intended as reference for understanding TFF patterns
- Not directly integrated into project

**Note from README++.MD (June 1, 2020)**:
> "Simple fed ave test pulled from experimental section, under review... haven't touched it YET."

This file was pulled for study but never adapted for the botnet detection use case.

---

## What Actually Worked

The working federated learning implementation is in:

- **`../../../anomaly-detection/train_v04.py`** - Latest iteration (June 2020)
- **`../../../anomaly-detection/run_experiment_00.py`** - Experimental runner
- **`../../../anomaly-detection/run_experiment_01.py`** - With compression techniques

These files in the main `anomaly-detection/` directory represent the "best effort" federated implementations that were tested and documented in the research.

## Lessons Learned

1. **TensorFlow Federated Complexity**: TFF's API was powerful but complex for this use case. The simulation approach didn't translate well to real distributed deployment.

2. **PySyft Immaturity (2020)**: PySyft-TensorFlow integration was too experimental. PyTorch would have been required, necessitating a complete rewrite.

3. **Federated Learning Overhead**: The simulation approach added significant complexity without clear production deployment path.

4. **Data Heterogeneity Challenges**: IoT device-specific data characteristics made federated aggregation challenging.

## Evolution Timeline

```bash
tff_initial_attempt.py (Initial TFF)
    ↓
manual_fedavg_attempt.py (Manual FedAvg)
    ↓
pysyft_exploration.py (PySyft attempt - dead end)
    ↓
tff_refinement.py (TFF refinement)
    ↓
train_v04.py (Working implementation - kept in main directory)
```

## References

- **TensorFlow Federated**: <https://www.tensorflow.org/federated>
- **PySyft Research**: `../../PYSYFT_RESEARCH.md`
- **Research Notes**: `../../../anomaly-detection/README++.MD`
- **Published Paper**: <https://www.sciencedirect.com/science/article/pii/S2666827022000081>

## Why These Are Archived

These files are preserved for:

- **Research transparency**: Showing the iterative development process
- **Future reference**: Documenting what approaches were tried and why they failed
- **Portfolio context**: Demonstrating problem-solving evolution
- **Learning resource**: Understanding federated learning challenges in practice

They are not intended to be run or maintained, but rather to provide historical context for the research process.

---

*Last Updated*: October 2024
*Original Research Period*: May-June 2020
*Researcher*: iAmGiG (Kennesaw State University)
