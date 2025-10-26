# TODO for archive-2020-fixed Branch

This branch fixes critical bugs found in the 2020 research code while maintaining the same dependency versions (Python 3.9, TensorFlow 2.10, etc.). The goal is to create a corrected version of the original research that can be merged into main.

## Critical Bugs to Fix

### Priority 1: Data Integrity Issues

- [ ] **Issue #13: Data Leakage in Scaler Fitting**
  - **Files affected**:
    - `anomaly-detection/train_autoencoder.py:29`
    - `anomaly-detection/test_autoencoder.py:40`
  - **Problem**: StandardScaler fit on both training AND validation data
  - **Fix**: Change `scaler.fit(x_train.append(x_opt))` to `scaler.fit(x_train)`
  - **Impact**: Current results likely inflated; need to re-test after fix
  - **Testing**: Re-run anomaly detection and compare to baseline (8.3% FP rate with bug)

- [ ] **Issue #16: Test/Train Data Split Overlap**
  - **Files affected**:
    - `anomaly-detection/train_autoencoder.py`
    - `anomaly-detection/test_autoencoder.py`
  - **Problem**: Both scripts use `random_state=17` causing potential overlap
  - **Fix**: Use different random states OR document if intentional
  - **Testing**: Verify proper train/test separation

### Priority 2: Deprecated Code

- [ ] **Issue #15: Mixed Keras Imports**
  - **Files affected**:
    - `anomaly-detection/train_autoencoder.py`
    - `classification/train_classifier.py`
    - `classification/test_classifier.py`
  - **Problem**: Mixed use of `from keras.` and `from tensorflow.keras.`
  - **Fix**: Standardize to `from tensorflow.keras.` throughout
  - **Impact**: TensorFlow 2.x compatibility

- [ ] **Issue #14: Deprecated pandas.append()**
  - **Files affected**:
    - `anomaly-detection/test_autoencoder.py`
    - `classification/train_classifier.py`
  - **Problem**: `DataFrame.append()` deprecated in pandas 1.4+
  - **Fix**: Replace with `pd.concat()`
  - **Note**: Works in current environment (pandas 1.3.5) but should fix for future compatibility

### Priority 3: Minor Bugs

- [ ] **Bug #20: Type Conversion Error**
  - **File**: `anomaly-detection/train_autoencoder.py:97`
  - **Problem**: Command-line args passed as strings, not ints
  - **Fix**: Add `int()` conversion: `train(int(sys.argv[1]))`
  - **Current workaround**: Call function directly from Python

- [ ] **Bug #21: Unused ModelCheckpoint Callback**
  - **File**: `anomaly-detection/train_autoencoder.py:55`
  - **Problem**: Callback defined but not used in `model.fit()`
  - **Fix**: Change `callbacks=[tensorboard]` to `callbacks=[cp, tensorboard]`
  - **Impact**: Models not saved during training

## Testing Plan

After implementing fixes:

1. **Re-run anomaly detection test**:
   - Dataset: Ecobee_Thermostat (13,113 benign instances)
   - Configuration: Top 5 features, 5 epochs
   - Compare false positive rate to baseline (8.3% with data leakage bug)
   - Document changes in `docs/RETROSPECTIVE.md` appendix

2. **Re-run classification test**:
   - Dataset: Benign, Gafgyt, Mirai (39,339 samples balanced)
   - Configuration: Top 5 features, 25 epochs
   - Compare accuracy to baseline (99.82% without data leakage)
   - Verify confusion matrix patterns remain consistent

3. **Cross-validation** (if time permits):
   - Implement k-fold cross-validation
   - Test with different random seeds
   - Document variance in results

## Non-Goals for This Branch

The following issues are marked for the modernized `main` branch (Issue #17):

- Full TensorFlow 2.15+ migration
- Federated learning implementation with Flower framework
- Docker containerization
- Unit test coverage
- MLflow experiment tracking
- Testing on newer botnet types

## Reference Documents

- **Test baseline results**: `docs/RETROSPECTIVE.md` lines 318-454
- **Bug documentation**: GitHub Issues #13-#16, #20, #21
- **Environment specs**: `environment-archive.yaml`
- **Original development notes**: `docs/archived/DEVELOPMENT_NOTES_2020.md`

## Branch Strategy

```bash
archive-2020-research (untouched) ← Preserve all original flaws
        ↓
archive-2020-fixed (this branch) ← Fix critical bugs, keep dependencies
        ↓ (PR after testing)
main ← Full modernization
```

---

**Last Updated**: October 26, 2024
**Current Status**: Ready to begin Priority 1 fixes
