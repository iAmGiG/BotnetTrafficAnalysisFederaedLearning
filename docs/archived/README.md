# Archived Materials

This directory contains archived materials from the 2020 research project.

## Contents

### `experimental/`
Experimental federated learning attempts that didn't make it into the final implementation. Each file represents a different approach:

- **tff_initial_attempt.py** - First attempt using TensorFlow Federated
- **manual_fedavg_attempt.py** - Manual Federated Averaging implementation
- **pysyft_exploration.py** - PySyft framework exploration (abandoned)
- **tff_refinement.py** - Refined TFF approach
- **tff_reference_example.py** - Reference code from TFF examples

See `experimental/README.md` for detailed evolution timeline.

### `federated_learning_architecture.png`
**Original filename**: `someKindaDiagram.png`

Simple architectural diagram showing the federated learning server-client model:
- **Server**: Passes back the expected threshold type
- **Client**: Receives model and data type
- **Communication**: Server passes model, learn-rate, and client data type

This was likely a conceptual sketch for understanding TensorFlow Federated's architecture where:
1. Server broadcasts model and hyperparameters to clients
2. Clients train locally on their data
3. Server aggregates updates

**Context**: This represents the TFF distributed aggregation protocol concept where:
> "Inputs live on the clients, outputs live on the server"

The diagram captures the essence of the challenge mentioned in the README:
> "This way does not make a flow of client first easy to approach. You still need a master controller to broadcast controls to individual clients."

## Why These Are Archived

These materials represent:
- **Research process** - Shows iterative development and learning
- **Dead ends** - Approaches that didn't work out but were valuable to try
- **Historical context** - Documents the state of FL frameworks in 2020
- **Learning artifacts** - Diagrams and notes from understanding complex concepts

They're preserved for transparency and as a record of the research journey.

---

*See also*:
- `../../RETROSPECTIVE.md` - Comprehensive project retrospective
- `../../PYSYFT_RESEARCH.md` - Analysis of PySyft evolution 2020-2025
