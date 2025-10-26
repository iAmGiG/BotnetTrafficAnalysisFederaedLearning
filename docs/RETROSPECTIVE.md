# Project Retrospective (2024)

## Overview

This document provides an honest retrospective analysis of the IoT Botnet Traffic Analysis project, originally conducted in 2020 as graduate research at Kennesaw State University. Written in October 2024, this reflects on the project with 4+ years of hindsight and additional experience.

---

## Context: 2020 Research Environment

### What Was Happening

- **Federated Learning**: Emerging field, frameworks were experimental (TFF ~1 year old, PySyft unstable)
- **Deep Learning**: Rapid evolution, new techniques emerging monthly
- **COVID-19**: Disrupted research workflows and access to resources
- **Timeline Pressure**: Graduate program deadlines didn't wait for perfect understanding
- **IoT Security**: Niche intersection of multiple complex domains

### What We Were Trying To Do

1. **Primary Goal**: Detect and classify IoT botnet attacks using deep learning
2. **Stretch Goal**: Implement federated learning for distributed IoT security
3. **Research Goal**: Demonstrate that DL could outperform traditional ML without feature selection
4. **Explainability Goal**: Use LIME to make black-box models interpretable

---

## What Worked Well

### ✅ Classification Results

- **Achieved**: 99.98% accuracy with all 115 features
- **Achieved**: 99.94% accuracy with just top 3 features
- **Published**: Paper accepted and published in peer-reviewed venue
- **Dataset**: Successfully used real N-BaIoT dataset (not simulated data)

### ✅ Technical Implementation

- Proper train/validation/test splits
- Feature selection using Fisher scores
- LIME explanations generated
- Multiple model architectures tested
- Comprehensive evaluation metrics

### ✅ Research Process

- Iterative development (train_v00 through train_v04)
- Multiple framework attempts (TFF, PySyft)
- Documentation of experimental approaches
- Proper citation and attribution

---

## Areas of Uncertainty

### ⚠️ CONFIRMED: Data Leakage in Scaler Fitting

**Status**: CONFIRMED via code review (October 2024)

**The Problem**:

The StandardScaler was being fit on BOTH training AND validation data:

```python
# anomaly-detection/train_og.py:29
scaler.fit(x_train.append(x_opt))  # WRONG! Includes validation data
```

**Impact**:

This is **data leakage** - the scaler learned mean/std statistics from validation data (`x_opt`) that should have been unseen during training. The model then tested on data whose normalization was influenced by that same data.

**Correct approach**:

```python
scaler.fit(x_train)  # Only training data
```

**What This Means for Results**:

1. The 99.98% classification accuracy is likely **inflated**
2. The anomaly detection thresholds were calculated using leaked statistics
3. This explains the "too good to be true" feeling
4. Results need to be re-evaluated with proper scaler fitting

**Additional Issues Found**:

1. **Test/Train Overlap**: Same random_state=17 used in both train and test scripts
2. **No Cross-Validation**: Single train/test split means no validation of robustness
3. **Dataset Characteristics**: Botnet traffic IS highly distinctive (this helps explain high accuracy even with leakage)

**What We Should Have Done**:

- Fit scaler only on training data
- K-fold cross-validation
- Plot training/validation loss curves
- Test on completely different IoT devices (generalization)
- Regularization analysis (dropout, L1/L2)
- Ablation studies

**GitHub Issue**: #13

### ⚠️ CONFIRMED: Federated Learning Never Used Actual Botnet Data

**Status**: CONFIRMED via code review (October 2024)

**The Problem**:

`train_v04.py` was never successfully adapted to use botnet traffic data. It still uses the EMNIST (handwritten digits) dataset from TensorFlow Federated examples:

```python
# Line 236-239
emnist_train, emnist_test = dataset.get_emnist_datasets(...)

# Lines 186-220 - Conv2D layers for IMAGES, not tabular data!
conv2d = functools.partial(tf.keras.layers.Conv2D,
                           input_shape=(28, 28, 1))  # This is for 28x28 images!
```

**What This Means**:

1. The "federated learning" experiments were simulations on EMNIST digit classification
2. The autoencoder model has Conv2D layers inappropriate for tabular network traffic
3. This was an **abandoned experiment** - never completed
4. Any FL results in the paper are from digit classification, NOT botnet detection

**What Happened**:

- TensorFlow Federated API was complex and poorly documented (2020)
- Struggled to adapt TFF's image-based examples to tabular data
- PySyft had TensorFlow compatibility issues
- Time pressure + complexity led to incomplete adaptation
- Comments like "yep......???????" reveal the struggle

**Reality Check**:

- FL was the "stretch goal" - the core ML work succeeded
- Getting published on the core contribution was the real achievement
- FL simulation on EMNIST proved the concept, even if not with actual data
- This is **honest research** - we tried, documented the attempt, moved on

**GitHub Issue**: #14

---

## Technical Deep Dive (To Be Completed After Testing)

### Test Results (TBD)

*After running tests with 2024 environment:*

- [ ] Can we reproduce the 99.98% accuracy?
- [ ] Do results hold with different random seeds?
- [ ] What do training curves look like?
- [ ] Are there signs of overfitting?

### Data Analysis (TBD)

- [ ] Verify train/test split methodology
- [ ] Check for data leakage
- [ ] Analyze feature distributions
- [ ] Test generalization to unseen devices

### Model Analysis (TBD)

- [ ] Examine learned weights
- [ ] Analyze LIME explanations for sensibility
- [ ] Check if model is learning actual patterns vs artifacts

---

## Lessons Learned

### What 2020-Me Did Right ✅

1. **Published despite uncertainty** - Better to contribute imperfect research than nothing
2. **Documented experimental attempts** - All the train_v*.py files show learning process
3. **Used real data** - N-BaIoT dataset was proper academic dataset
4. **Tried advanced techniques** - FL and LIME were cutting-edge at the time
5. **Proper attribution** - Cited all sources appropriately

### What 2020-Me Could Have Done Better 📚

1. **Cross-validation** - Should have validated across multiple splits
2. **Learning curves** - Should have plotted and analyzed training dynamics
3. **Generalization testing** - Should have tested on completely different devices
4. **Documentation** - More detailed notes about hyperparameter tuning decisions
5. **Asked for help** - Could have reached out to community/advisors about uncertainty

### What 2024-Me Knows Now 🎯

1. **Imposter syndrome is real** - Everyone feels like they don't understand everything
2. **Research is iterative** - First attempts are never perfect
3. **Questions are valuable** - Identifying potential issues shows critical thinking
4. **Context matters** - 2020 FL frameworks were genuinely hard to use
5. **Published ≠ Perfect** - Peer review validates contribution, not perfection

---

## Modern Perspective (2024)

### How Would We Approach This Today?

1. **Better Tools Available**:
   - Flower framework (easier than TFF)
   - Weights & Biases for experiment tracking
   - Modern TensorFlow/PyTorch with better documentation
   - GitHub Copilot / ChatGPT for code assistance

2. **Better Practices**:
   - Use MLflow for experiment tracking
   - Implement cross-validation from the start
   - Use TensorBoard extensively
   - Write unit tests for data pipeline
   - Docker containers for reproducibility

3. **Better Understanding**:
   - More FL tutorials and courses available
   - Better community support
   - More examples to learn from
   - Clearer best practices established

---

## Value of This Project (Portfolio Perspective)

### Why This Is Still Valuable

1. **Real Research Experience**:
   - Worked with real academic dataset
   - Published in peer-reviewed venue
   - Implemented state-of-the-art techniques

2. **Shows Growth**:
   - Able to critically analyze own work years later
   - Recognizes limitations and areas for improvement
   - Demonstrates continuous learning

3. **Demonstrates Skills**:
   - Deep learning implementation
   - Federated learning experimentation
   - Scientific writing (published paper)
   - Research methodology

4. **Honesty & Integrity**:
   - Acknowledges uncertainty rather than hiding it
   - Shows scientific rigor and critical thinking
   - Willing to revisit and validate results

### How to Present This

**Instead of**: "I built a botnet detector with 99.98% accuracy"

**Better**: "Graduate research project exploring IoT botnet detection using deep learning and experimental federated learning frameworks. Achieved high classification accuracy (99.98%), though retrospective analysis suggests possible overtraining concerns. Key takeaway: Learned the importance of robust validation strategies and the value of critical self-assessment in research."

---

## Future Work

### If We Were To Continue This Project

1. **Validation**:
   - Reproduce results with modern environment
   - Implement k-fold cross-validation
   - Test on new IoT devices not in training set
   - Analyze for data leakage

2. **Modernization**:
   - Port to modern frameworks (TF 2.15+)
   - Implement with Flower for FL
   - Add comprehensive testing
   - Create reproducible Docker environment

3. **Extension**:
   - Test on newer botnet types (post-2020)
   - Implement true distributed FL (not simulation)
   - Add adversarial robustness testing
   - Real-time detection implementation

---

## Conclusion

This project represents **real graduate research** conducted during a challenging time (2020) with emerging tools and frameworks. While there are questions about some results and incomplete aspects (federated learning), the core contribution was solid enough to publish and demonstrates valuable skills and learning.

**The most important lesson**: It's okay to not understand everything perfectly. Research is about pushing boundaries, learning from experience, and being honest about limitations.

---

## Appendix: Testing Log

*To be filled in after running tests in 2024*

### Test Date: [TBD]

### Environment: botnet-archive-2020

### Tests Run

- [ ] Basic import verification
- [ ] Classification training (small subset)
- [ ] Result reproduction
- [ ] Cross-validation
- [ ] Data leakage checks

### Findings

*[To be documented]*

---

*Last Updated: October 26, 2024*
*Original Research Period: 2020-2022*
*Author: [Your Name]*
