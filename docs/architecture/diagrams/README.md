# Architecture Diagrams

This directory contains architecture visualization scripts and sources.

## Diagram Sets

We maintain TWO sets of diagrams:

### 1. Current System (As-Is) - 2020 Architecture

Documents the existing system before modernization:

- Uses deprecated dependencies (TF 2.10, Pandas 1.3.5)
- Original file structure with `anomaly-detection/` hyphenated names
- Mixed keras imports
- TensorFlow Federated (experimental, incomplete)

**Purpose**: Historical reference, understanding current system

**Files**:

- `generate_current_diagrams.py` - Generates current system diagrams
- `current_*.gv` - Graphviz source files
- Images saved to `images/current/`

### 2. Target System (To-Be) - 2025 Modernized Architecture

Documents the target modernized system:

- Modern dependencies (TF 2.17+, Pandas 2.2+)
- New `src/` package layout
- Unified keras 3.0 imports
- Flower federated learning
- Production-ready deployment

**Purpose**: Modernization goal, architectural vision

**Files**:

- `generate_target_diagrams.py` - Generates target system diagrams
- `target_*.gv` - Graphviz source files
- Images saved to `images/target/`

## Generating Diagrams

### Current System Diagrams

```bash
python generate_current_diagrams.py
```

Output: `images/current/*.png` and `images/current/*.svg`

### Target System Diagrams

```bash
python generate_target_diagrams.py
```

Output: `images/target/*.png` and `images/target/*.svg`

### Both

```bash
python generate_diagrams.py  # Generates both current and target
```

## Diagram Types

Both sets include:

1. **System Architecture** - High-level component overview
2. **Data Pipeline** - Data processing flow
3. **Autoencoder Architecture** - Anomaly detection model
4. **Classifier Architecture** - Multi-class classifier
5. **Federated Learning Architecture** - FL implementation
6. **File Structure** - Directory organization

## Dependencies

```bash
pip install graphviz
```

System graphviz installation also required:

- **Ubuntu/Debian**: `sudo apt-get install graphviz`
- **macOS**: `brew install graphviz`
- **Windows**: Download from [graphviz.org](https://graphviz.org/download/)

## CI/CD Integration

Diagrams are automatically regenerated on:

- Push to main branch
- Documentation updates
- Manual workflow dispatch

See: `.github/workflows/docs.yaml`

## Directory Structure

```
diagrams/
├── README.md                          # This file
├── generate_diagrams.py              # Master script (generates both)
├── generate_current_diagrams.py      # Current system diagrams
├── generate_target_diagrams.py       # Target system diagrams
├── current_*.gv                      # Graphviz sources (current)
├── target_*.gv                       # Graphviz sources (target)
└── images/
    ├── current/                      # Current system diagrams
    │   ├── system_architecture.png
    │   ├── system_architecture.svg
    │   └── ...
    └── target/                       # Target system diagrams
        ├── system_architecture.png
        ├── system_architecture.svg
        └── ...
```

## Usage in Documentation

### Current System

```markdown
![Current System Architecture](architecture/diagrams/images/current/system_architecture.png)
```

### Target System

```markdown
![Target System Architecture](architecture/diagrams/images/target/system_architecture.png)
```

### Comparison

Show both side-by-side to illustrate modernization:

```markdown
| Current (2020) | Target (2025) |
|----------------|---------------|
| ![Current](images/current/system_architecture.png) | ![Target](images/target/system_architecture.png) |
```

## Maintenance

- Update current diagrams only for corrections, not changes
- Update target diagrams as design evolves
- Keep both sets in sync structurally for comparison
- Archive old versions when major changes occur
