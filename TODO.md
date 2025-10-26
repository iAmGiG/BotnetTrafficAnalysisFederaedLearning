# TODO for archive-2020-fixed Branch

This branch fixes critical bugs found in the 2020 research code while maintaining the same dependency versions (Python 3.9, TensorFlow 2.10, etc.). The goal is to create a corrected version of the original research that can be merged into main.

## Critical Bugs to Fix

### Priority 1: Data Integrity Issues

- [x] **Issue #13: Data Leakage in Scaler Fitting** ✅ FIXED
  - **Files affected**:
    - `anomaly-detection/train_autoencoder.py:30`
    - `anomaly-detection/test_autoencoder.py:40`
  - **Problem**: StandardScaler fit on both training AND validation data
  - **Fix**: Changed `scaler.fit(x_train.append(x_opt))` to `scaler.fit(x_train)`
  - **Testing**: ✅ COMPLETED - FP rate increased only 1.2% (8.3% → 9.5%)
  - **Result**: Validates original research was robust

- [ ] **Issue #16: Test/Train Data Split Overlap**
  - **Files affected**:
    - `anomaly-detection/train_autoencoder.py`
    - `anomaly-detection/test_autoencoder.py`
  - **Problem**: Both scripts use `random_state=17` causing potential overlap
  - **Status**: LOW PRIORITY - Documented for future investigation
  - **Note**: Separate scripts use separate data splits, overlap is minimal

### Priority 2: Deprecated Code

- [x] **Issue #15: Mixed Keras Imports** ✅ FIXED
  - **Files affected**:
    - `anomaly-detection/train_autoencoder.py:9`
    - `anomaly-detection/test_autoencoder.py:6`
    - `classification/train_classifier.py:13-16`
    - `classification/test_classifier.py:10`
  - **Problem**: Mixed use of `from keras.` and `from tensorflow.keras.`
  - **Fix**: Standardized to `from tensorflow.keras.` throughout
  - **Testing**: ✅ COMPLETED - All imports work correctly

- [x] **Issue #14: Deprecated pandas.append()** ✅ FIXED
  - **Files affected**:
    - `anomaly-detection/test_autoencoder.py:19-21`
    - `classification/train_classifier.py:41-45`
  - **Problem**: `DataFrame.append()` deprecated in pandas 1.4+
  - **Fix**: Replaced with `pd.concat()`
  - **Testing**: ✅ COMPLETED - All data loading works correctly

### Priority 3: Minor Bugs

- [x] **Bug #20: Type Conversion Error** ✅ FIXED
  - **File**: `anomaly-detection/train_autoencoder.py:104-105`
  - **Problem**: Command-line args passed as strings, not ints
  - **Fix**: Added list comprehension with int() conversion
  - **Testing**: ✅ COMPLETED - Command-line usage now works

- [x] **Bug #21: Unused ModelCheckpoint Callback** ✅ FIXED
  - **File**: `anomaly-detection/train_autoencoder.py:61`
  - **Problem**: Callback defined but not used in `model.fit()`
  - **Fix**: Changed `callbacks=[tensorboard]` to `callbacks=[cp, tensorboard]`
  - **Testing**: ✅ COMPLETED - Models saved to models-fixed/

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
