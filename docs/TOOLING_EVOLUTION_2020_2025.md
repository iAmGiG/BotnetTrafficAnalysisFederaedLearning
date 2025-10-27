# Federated Learning & ML Tooling Evolution: 2020 → 2025

**Purpose**: Document how the FL/ML landscape changed since your 2020 research
**Date**: 2025-10-26

---

## Executive Summary

**TL;DR**:

- **PySyft**: Still active, shifted to privacy research (more complex)
- **TensorFlow Federated**: Not deprecated, but still problematic for production
- **Flower**: NEW winner (2020-2025), production-ready, most popular
- **LIME**: Still works, but **SHAP** is now dominant for explainability
- **PyTorch vs TF**: PyTorch winning in research, but **stay with TF** for this project
- **N-BaIoT dataset**: Still valid, but newer datasets exist (stretch goal)

---

## Federated Learning Frameworks

### 1. PySyft

**2020 Status**: Cutting-edge, unstable, OpenMined project

**2025 Status**: **Still active, more mature**

**Evolution**:

```
2020: Young, breaking changes, research-focused
2021: Acquired stable API (v0.5)
2022: Shifted focus to privacy-preserving FL
2023-2025: Mature, but complex (differential privacy, secure multiparty computation)
```

**Current Use Cases**:

- Privacy research (differential privacy)
- Secure aggregation
- Medical/financial FL (high privacy requirements)

**Pros** (2025):

- Most privacy-preserving option
- Strong cryptographic guarantees
- Good for sensitive data

**Cons** (2025):

- Complex setup
- Steeper learning curve than 2020
- Overkill for IoT botnet detection (privacy not the goal)

**Verdict**: **Skip** for this project. Use Flower instead. Consider PySyft if you add a "Privacy-Preserving FL" research angle.

### 2. TensorFlow Federated (TFF)

**2020 Status**: Google's official FL framework, promising

**2025 Status**: **Not deprecated, but not recommended**

**What Changed**:

```
2020: New, complex API, dependency hell (your experience)
2021-2022: Incremental improvements, still research-focused
2023-2025: Maintained but stagnant, no major breakthroughs
```

**Current Reality**:

- Still official Google project
- Used in Google research papers
- **Still hard to install** (you weren't wrong in 2020!)
- Simulation-focused, weak deployment story
- Complex API didn't improve much

**Why it failed to gain traction**:

1. Too research-oriented (not production-ready)
2. TensorFlow-only (PyTorch users excluded)
3. Steep learning curve
4. Flower emerged as better alternative

**Verdict**: **Skip**. Your 2020 experience was correct - it's not worth the hassle.

### 3. Flower (flwr) - THE WINNER

**2020 Status**: Didn't exist! (Created late 2020)

**2025 Status**: **Most popular FL framework**

**History**:

```bash
2020 (Nov): First release by Oxford/Cambridge researchers
2021: Rapid adoption, simple API wins hearts
2022: Flower Labs founded, $3.5M seed funding
2023: Used by Porsche, Accenture, others
2024-2025: Dominant FL framework (5k+ GitHub stars)
```

**Why Flower Won**:

1. **Framework Agnostic**: Works with TensorFlow, PyTorch, JAX, scikit-learn
2. **Simple API**: Much easier than TFF or PySyft
3. **Production Ready**: Actual deployment, not just simulation
4. **Active Development**: Regular releases, responsive team
5. **Good Documentation**: Examples, tutorials, community

**Example** (how simple it is):

```python
# Server
import flwr as fl
fl.server.start_server(config=fl.server.ServerConfig(num_rounds=10))

# Client
class MyClient(fl.client.NumPyClient):
    def fit(self, parameters, config):
        # Train your model
        return updated_parameters, num_samples, {}

fl.client.start_numpy_client(server_address="localhost:8080", client=MyClient())
```

**Companies Using Flower** (2025):

- Porsche (vehicle FL)
- Accenture (enterprise FL)
- Adap (medical FL)
- Various healthcare/finance companies

**Verdict**: **Use Flower**. This is the clear choice for 2025.

---

## PyTorch vs TensorFlow Decision

### The Landscape Shift (2020 → 2025)

**2020**:

- TensorFlow dominant (60% market share)
- PyTorch growing (35% market share)
- Keras part of TensorFlow

**2025**:

- PyTorch dominant in **research** (70%+ of papers)
- TensorFlow still strong in **production** (40% market share)
- Keras 3.0 is **framework-agnostic** (works with both!)

### PyTorch Advantages (2025)

**Why researchers prefer it**:

1. More "Pythonic" API (feels natural)
2. Dynamic computation graphs (better debugging)
3. Better for cutting-edge research (transformers, LLMs)
4. Dominant in academic papers

**Example**:

```python
# PyTorch feels more natural
import torch

model = torch.nn.Sequential(
    torch.nn.Linear(115, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 3)
)

loss = model(x).cross_entropy(y)  # Clean, intuitive
loss.backward()
```

### TensorFlow Advantages (2025)

**Why production uses it**:

1. Better deployment story (TF Serving, TFLite)
2. More mature ecosystem
3. Better for mobile/edge (TFLite)
4. Keras 3.0 makes it framework-agnostic

**Example**:

```python
# TensorFlow with Keras 3.0
import keras

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy')
# Can run on TF, PyTorch, or JAX backend!
```

### **Recommendation for This Project: Stay with TensorFlow**

**Reasons**:

1. **You have working code** (anomaly detection, classification)
2. **Simple models** (MLP, autoencoder) - framework doesn't matter
3. **Time savings**: Rewriting would add 4-6 weeks
4. **Flower supports both equally** - no FL advantage to PyTorch
5. **Keras 3.0** makes it easy to switch later if needed
6. **This is a portfolio/modernization project**, not cutting-edge research

**When PyTorch makes sense**:

- New research with transformers/LLMs
- Cutting-edge architectures
- Learning PyTorch for career goals

**For your use case**: TensorFlow + Keras 3.0 is perfect.

---

## Explainability: LIME vs SHAP

### LIME (2016)

**Status** (2025): Still maintained, still used, but **SHAP is better**

**What it does**:

- Local interpretable model-agnostic explanations
- Perturbs inputs, fits local linear model
- Shows feature importance per prediction

**Pros**:

- Simple concept
- Works with any model
- Good visualizations

**Cons** (compared to SHAP):

- Not theoretically grounded
- Can be unstable (different runs give different results)
- Less accurate than SHAP

### SHAP (2017-2025) - THE WINNER

**Status** (2025): **Dominant explainability framework**

**Why it's better**:

1. **Theoretically grounded**: Based on Shapley values (game theory)
2. **Consistent**: Same input always gives same explanation
3. **Accurate**: Better reflects true feature importance
4. **Better visualizations**: Summary plots, waterfall plots, force plots
5. **Actively maintained**: Microsoft/community support

**Example**:

```python
import shap

# For neural networks
explainer = shap.DeepExplainer(model, x_train[:100])
shap_values = explainer.shap_values(x_test)

# Beautiful visualizations
shap.summary_plot(shap_values, x_test)  # Global importance
shap.force_plot(explainer.expected_value, shap_values[0])  # Single prediction
shap.waterfall_plot(shap_values[0])  # Feature contributions
```

**Visualization Quality** (SHAP is much better):

- Summary plots: See global feature importance
- Force plots: See how features push prediction
- Waterfall plots: Step-by-step feature contributions
- Dependence plots: Feature interactions

### **Recommendation: Add SHAP, Keep LIME**

**Approach**:

1. Keep LIME (already implemented, works)
2. **Add SHAP** (better science, better visuals)
3. Compare both in documentation

**Benefits**:

- Shows you know modern tools (SHAP)
- Validates with multiple methods (LIME + SHAP agreement)
- Better for papers/presentations (SHAP is standard now)

---

## Dataset Evolution

### N-BaIoT (2018) - YOUR CURRENT DATASET

**Status** (2025): **Still valid, still cited**

**Pros**:

- Well-documented
- Clean labels
- Real botnet attacks (Mirai, Gafgyt)
- 9 diverse IoT devices
- Many research papers use it (comparable results)

**Cons**:

- Attacks from 2016-2017 (dated)
- Only 2 botnet types
- Limited device diversity

**Verdict**: **Keep as primary dataset** - it's still the standard.

### Newer Datasets (2020-2024)

**1. IoT-23** (Stratosphere Lab, 2020)

**What it is**:

- 23 IoT devices
- More recent attacks
- Better device diversity

**Pros**:

- More devices
- Modern attack types
- Good documentation

**Cons**:

- Harder to process than N-BaIoT
- Less widely used (fewer comparable results)

**Use Case**: **Stretch goal** - test generalization across datasets

**2. Edge-IIoTset** (2022)

**What it is**:

- 10 IoT devices
- Modern attack types (DDoS, ransomware, injection)
- Very comprehensive

**Pros**:

- Recent attacks
- Good diversity
- Multiple attack categories

**Cons**:

- Very large (GB+ of data)
- Complex preprocessing
- Overwhelming for a modernization project

**Use Case**: **Future research** if pursuing publications

**3. TON-IoT** (2021)

**What it is**:

- Testbed data
- Multiple attack vectors
- Comprehensive

**Pros**:

- Very complete
- Well-structured
- Good for research

**Cons**:

- Massive size
- Requires significant preprocessing
- Overkill for your project

**Use Case**: **Skip** - too large for modernization scope

### **Recommendation: N-BaIoT Primary + IoT-23 Stretch**

**Approach**:

**Phase 1** (Core): N-BaIoT

- All model development
- All validation
- All FL experiments

**Phase 2** (Stretch Goal): IoT-23 validation

- Test trained models on IoT-23
- Measure cross-dataset generalization
- Publish results

**Benefits**:

- Comparable to existing research (N-BaIoT standard)
- Shows generalization (IoT-23 validation)
- Manageable scope

**Timeline**:

- N-BaIoT: Weeks 1-12
- IoT-23: Weeks 13-14 (optional)

---

## ML Tooling Evolution (2020 → 2025)

### What Existed in 2020-2021

**Not Available**:

- ChatGPT / LLMs (GPT-2 existed, but not mainstream)
- Stable Diffusion / DALL-E
- GitHub Copilot
- Transformers were new (BERT was 2018)
- Few FL frameworks (TFF only)

**Available**:

- CNNs dominant for vision
- RNNs/LSTMs for sequences
- Basic transformers (research)
- Supervised learning standard

### What Changed by 2025

**Revolutionary Changes**:

1. **LLMs Everywhere** (2022-2025)
   - GPT-3, GPT-4, Claude, Llama
   - Transformers dominant for NLP
   - Transfer learning standard

2. **FL Went Mainstream** (2022-2025)
   - Apple uses FL (keyboards, Siri)
   - Google uses FL (Gboard)
   - Flower emerged as standard
   - Production deployments common

3. **MLOps Matured** (2021-2025)
   - MLflow standard for tracking
   - Weights & Biases popular
   - DVC for data versioning
   - Great Expectations for data validation

4. **Model Serving Standardized** (2022-2025)
   - TensorFlow Serving mature
   - Triton Inference Server (NVIDIA)
   - ONNX more popular
   - Edge deployment easier (TFLite, ONNX)

5. **Explainability Standard** (2020-2025)
   - SHAP dominant
   - Integrated into scikit-learn
   - Expected in production

### What DIDN'T Change for Your Use Case

**Your problem**: IoT botnet detection with tabular data

**Still valid** (2020 → 2025):

- MLPs and autoencoders work great
- Fisher score feature selection valid
- Tabular data handling similar
- Supervised learning approach sound

**Your 2020 research is scientifically sound** - the modernization is about **engineering**, not the core ML.

---

## Priority Adjustments Based on Your Feedback

### CI/CD: HIGH → MEDIUM

**Agreed**. For a Python research project:

**Keep (HIGH Priority)**:

- Pre-commit hooks (black, isort, mypy)
- Basic linting
- Type hints
- Testing framework (pytest)

**Downgrade to MEDIUM**:

- Full GitHub Actions CI/CD
- Automated deployment
- Docker containers
- Release automation

**Downgrade to LOW**:

- Production monitoring
- Kubernetes
- Advanced DevOps

**Rationale**: Focus on **developer experience** and **code quality**, not production infrastructure.

### Updated Priority List

**CRITICAL**:

1. Architecture diagrams (#25)
2. Directory restructuring (#26)
3. Overfitting analysis (#23)
4. Modern dependencies

**HIGH**:
5. Flower FL implementation (#28)
6. Testing framework (pytest)
7. Code quality (type hints, docstrings)

**MEDIUM**:
8. Basic CI/CD (#27 - simplified)
9. Documentation (#29)
10. SHAP explainability

**STRETCH**:
11. Synthetic data (#30)
12. IoT-23 dataset validation
13. Performance optimization

---

## Deliverable Framing (5-Year Revision)

**You're right**: This is a **modernization**, not new research.

**Framing**:

### What This IS

- **2020 Research**: Original work (scientifically sound)
- **2025 Modernization**: Engineering update, validation, FL implementation
- **Portfolio Piece**: Shows growth and engineering maturity

### What This IS NOT

- New research claiming novelty
- Invalidation of 2020 work
- Cutting-edge ML research

### Documentation Language

**Use**:

- "Modernization of 2020 research"
- "Engineering upgrade with modern tooling"
- "Validation and federated learning extension"
- "Production-ready implementation"

**Avoid**:

- "Novel approach" (it's not, it's 2020)
- "State-of-the-art" (transformers are SOTA, not MLPs)
- "New research" (it's a revision)

### Honest Framing is STRONG

**Example** (README):

```markdown
## About This Project

This project **modernizes** 2020 graduate research on IoT botnet detection. The original research was scientifically sound, achieving 99.85% classification accuracy on the N-BaIoT dataset.

**2025 Modernization**:
- Updated to Python 3.11, TensorFlow 2.17
- Fixed data leakage bug (minimal impact: 0.13% accuracy drop)
- Added federated learning with Flower framework
- Comprehensive overfitting analysis and validation
- Production-ready code quality (tests, docs, CI/CD)

**Original Work** (2020-2022):
- See `archive-2020-research` branch for original implementation
- Published paper: [ScienceDirect Link]
- Thesis: [KSU Digital Commons Link]
```

**This shows**:

- **Honesty**: Clear about what's old vs new
- **Growth**: Shows 5 years of engineering skill development
- **Integrity**: Validates old work, doesn't claim novelty
- **Value**: Demonstrates modernization skills

---

## Final Recommendations

### Immediate Actions (This Week)

1. ✅ Install mamba/conda
2. ✅ Create `botnet-modern` environment
3. ⏳ Generate architecture diagrams
4. ⏳ Review roadmap and adjust priorities

### Sprint 1 (Weeks 1-2): Foundation

**Focus**:

- Generate diagrams
- Plan directory restructuring
- Set up modern environment
- Install development tools (black, pytest)

**NOT**:

- Don't start coding yet
- Don't rush into FL
- Don't worry about CI/CD

### Tool Decisions (FINAL)

| Category | Tool | Why |
|----------|------|-----|
| **Language** | Python 3.11 | Modern, stable |
| **Framework** | TensorFlow 2.17 | Keep existing code |
| **FL Framework** | Flower | Best choice 2025 |
| **Explainability** | SHAP (+ LIME) | Modern standard |
| **Testing** | pytest | Python standard |
| **Formatting** | black + isort | Auto-format |
| **Type Checking** | mypy | Catch bugs early |
| **Experiment Tracking** | MLflow (optional) | Nice to have |

### What NOT to Do

**Don't**:

- Rewrite in PyTorch (waste of time)
- Use TensorFlow Federated (broken)
- Claim this is new research (it's modernization)
- Overcomplicate CI/CD (research project, not production)
- Rush (take time to do it right)

**Do**:

- Stay with TensorFlow
- Use Flower for FL
- Be honest about 5-year revision
- Focus on validation and code quality
- Document everything

---

## Summary

**Your 2020 work was good**. The tools evolved, but your core approach is still valid.

**2025 Modernization = Engineering Upgrade**:

- Modern dependencies (Python 3.11, TF 2.17)
- Better FL framework (Flower)
- Code quality (tests, types, docs)
- Validation (overfitting analysis)

**Time Investment**: 12-14 weeks for full modernization, 4-6 weeks for MVP

**Outcome**: Portfolio piece showing engineering growth from 2020 → 2025

---

**Next**: Check `environment-modern.yaml` and create the conda env!
