# Priority Adjustments Based on User Feedback

**Date**: 2025-10-26
**Context**: User review of MODERNIZATION_ROADMAP.md

---

## Key Decisions

### 1. Framework Choice: **Stay with TensorFlow**

**Decision**: Do NOT rewrite in PyTorch

**Reasoning**:

- Existing working code (anomaly detection, classification)
- Simple models (MLP, autoencoder) - framework doesn't matter
- Time savings: 4-6 weeks saved
- Flower supports both TF and PyTorch equally
- This is a modernization, not bleeding-edge research
- Keras 3.0 makes framework-agnostic code possible

**Future**: Can consider PyTorch for new projects

### 2. Federated Learning: **Flower is the Winner**

**Decision**: Use Flower, skip TensorFlow Federated and PySyft

**Reasoning**:

- TFF: Still problematic (user's 2020 experience was correct)
- PySyft: Too complex, privacy-focused (not needed here)
- Flower: Modern (2020-2025), production-ready, most popular
- Flower: Best documentation, simple API, active development

### 3. Dataset: **N-BaIoT Primary, IoT-23 Stretch**

**Decision**: Keep N-BaIoT as primary, add IoT-23 as stretch goal

**Reasoning**:

- N-BaIoT: Still valid, standard benchmark, comparable results
- N-BaIoT: Well-documented, manageable size
- IoT-23: Good for generalization testing (stretch goal)
- Newer datasets (Edge-IIoTset, TON-IoT): Too large, overkill

**Stretch Goal**: Cross-dataset validation with IoT-23

### 4. Explainability: **Add SHAP, Keep LIME**

**Decision**: Add SHAP as primary, keep LIME for comparison

**Reasoning**:

- LIME: Still works, already implemented
- SHAP: Modern standard (2025), theoretically grounded
- SHAP: Better visualizations, more accurate
- Both together: Shows comprehensiveness

### 5. CI/CD Priority: **HIGH → MEDIUM**

**Decision**: Simplify CI/CD, focus on developer experience

**Adjusted Priorities**:

**HIGH** (Keep):

- Pre-commit hooks (black, isort, mypy)
- Testing framework (pytest)
- Type hints and docstrings
- Code quality

**MEDIUM** (Downgrade):

- GitHub Actions workflows
- Automated testing on PR
- Documentation generation
- Basic CI

**LOW** (Nice to have):

- Advanced CI/CD (releases, deployment)
- Docker containers
- Kubernetes
- Production monitoring

**Reasoning**:

- This is a Python research project, not production software
- Focus on code quality, not infrastructure
- Developer experience > automated deployment

### 6. Synthetic Data: **New Stretch Goal**

**Decision**: Add synthetic data generation as stretch goal

**Created**: Issue #30

**Reasoning**:

- Controlled testing
- Class balancing
- Privacy-preserving dataset
- Research contribution value

**Timeline**: 6 weeks (optional)

### 7. Deliverable Framing: **5-Year Modernization**

**Decision**: Frame as modernization, not new research

**Language**:

- ✅ "Modernization of 2020 research"
- ✅ "Engineering upgrade with modern tooling"
- ✅ "Validation and FL extension"
- ❌ "Novel approach" (it's not novel)
- ❌ "State-of-the-art" (transformers are SOTA, not this)

**Reasoning**:

- Honesty shows integrity
- Demonstrates 5 years of growth
- Portfolio piece: engineering maturity
- No need to claim novelty

---

## Updated Priority Tiers

### CRITICAL (Must Have)

1. **Architecture Diagrams** (#25) - Foundation for understanding
2. **Directory Restructuring** (#26) - Modern Python package layout
3. **Overfitting Analysis** (#23) - Scientific validation
4. **Dependency Modernization** - Python 3.11, TF 2.17, Pandas 2.2

**Timeline**: Weeks 1-4

### HIGH (Should Have)

5. **Flower FL Implementation** (#28) - Core feature
6. **Testing Framework** - pytest, fixtures, 80% coverage
7. **Code Quality** - Type hints, docstrings, black formatting
8. **SHAP Explainability** - Modern standard

**Timeline**: Weeks 5-8

### MEDIUM (Nice to Have)

9. **Basic CI/CD** (#27) - Simplified, developer-focused
10. **Documentation** (#29) - User guides, API docs
11. **Data Leakage Fix** (#13) - Already done, document impact
12. **Configuration Management** - YAML configs, remove hard-coding

**Timeline**: Weeks 9-12

### STRETCH (Optional)

13. **Synthetic Data** (#30) - Research contribution
14. **IoT-23 Dataset** - Cross-dataset validation
15. **Performance Optimization** - Profiling, data pipeline
16. **MLflow Integration** - Experiment tracking

**Timeline**: Weeks 13-14+ (optional)

---

## Timeline Options

### MVP (4-6 Weeks)

**Scope**: CRITICAL items only

- Architecture diagrams
- Directory restructuring
- Dependency updates
- Overfitting analysis
- Basic testing

**Outcome**: Working modernized code, validated results

### Full Modernization (10-12 Weeks)

**Scope**: CRITICAL + HIGH

- Everything in MVP
- Flower FL implementation
- Comprehensive testing (80% coverage)
- Code quality (types, docs, formatting)
- SHAP explainability

**Outcome**: Production-ready, portfolio-quality code

### Complete + Research (14+ Weeks)

**Scope**: CRITICAL + HIGH + MEDIUM + STRETCH

- Everything in Full Modernization
- Simplified CI/CD
- Complete documentation
- Synthetic data generation
- Cross-dataset validation

**Outcome**: Publication-ready, research contribution

---

## What Changed from Original Roadmap

### Removed/Downgraded

1. **PyTorch Migration**: Removed entirely (stay with TF)
2. **Advanced CI/CD**: Downgraded from HIGH to MEDIUM
3. **Production Deployment**: Downgraded from MEDIUM to LOW
4. **TensorFlow Federated**: Removed (use Flower)
5. **PySyft Exploration**: Removed (not needed)

### Added/Upgraded

1. **SHAP Explainability**: Added (upgrade from LIME)
2. **Synthetic Data**: Added as stretch goal (#30)
3. **IoT-23 Dataset**: Added as stretch goal
4. **Tooling Evolution Doc**: Added (TOOLING_EVOLUTION_2020_2025.md)
5. **Cross-Dataset Validation**: Added as research goal

### Kept/Refined

1. **Flower FL**: Confirmed as the right choice
2. **Directory Restructuring**: Still CRITICAL
3. **Overfitting Analysis**: Still CRITICAL
4. **Testing**: Still HIGH priority
5. **Documentation**: Still important (MEDIUM)

---

## Rationale Summary

**Core Philosophy**:
This is an **engineering modernization** of **scientifically sound 2020 research**, not cutting-edge new research.

**Focus On**:

- Validating original results (overfitting analysis)
- Modern engineering practices (testing, types, docs)
- Adding FL capability (Flower)
- Code quality and maintainability

**Don't Focus On**:

- Claiming novelty (it's 5 years old)
- Production deployment (research project)
- Bleeding-edge ML (transformers, LLMs)
- Complex infrastructure (K8s, advanced CI/CD)

**Outcome**:
Portfolio piece demonstrating:

- Scientific integrity (validation)
- Engineering growth (2020 → 2025)
- Modern best practices
- Federated learning capability

---

## Next Steps

1. ✅ Create modern conda environment
2. ⏳ Generate architecture diagrams (both current and target)
3. ⏳ Review and internalize priority adjustments
4. ⏳ Start Sprint 1: Foundation

**Sprint 1 Focus**:

- Architecture diagrams
- Directory restructuring planning
- Environment setup
- Development tools (black, pytest, mypy)

**NOT in Sprint 1**:

- Coding (not yet)
- FL implementation (later)
- CI/CD setup (later)

---

## Questions Answered

| Question | Answer |
|----------|--------|
| **PySyft evolution?** | Still active, more complex, privacy-focused |
| **TFF deprecated?** | No, but still problematic (skip it) |
| **Flower history?** | Created 2020, now most popular FL framework |
| **PyTorch switch?** | No - stay with TensorFlow (time savings) |
| **Tooling changes?** | See TOOLING_EVOLUTION_2020_2025.md |
| **Newer datasets?** | IoT-23 as stretch goal, N-BaIoT primary |
| **Synthetic data?** | Yes, added as stretch goal (#30) |
| **CI/CD priority?** | Downgraded to MEDIUM (simplified) |
| **LIME still good?** | Yes, but add SHAP (modern standard) |
| **src/ folder?** | Absolutely - modern Python packaging |
| **Dating deliverables?** | Frame as modernization, not new research |

---

**Status**: Priorities adjusted, ready to proceed with Sprint 1
