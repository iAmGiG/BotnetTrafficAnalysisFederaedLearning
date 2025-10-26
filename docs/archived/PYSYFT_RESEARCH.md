# PySyft Research Summary

## Historical Context (2020)

### Original Project Status

- **Date**: May-June 2020
- **Primary Framework**: TensorFlow 2.x + TensorFlow Federated (TFF)
- **PySyft Status**: Experimental integration attempted but not completed
- **Key Issue**: "PySyft, as of (May 2020) still does not have a well-defined solution with TF" (from README++.MD)

### What Was Attempted

- Imported `syft` and `syft_tensorflow` in `train_syft.py` (junedeskalt branch)
- Imports were added but never actually used in the code
- Project primarily relied on TensorFlow Federated for simulation
- Python 3.6 environment with PyTorch (see .idea/misc.xml) but main code uses TensorFlow

## Current State (2025)

### PySyft Evolution

- **Current Version**: 0.9.6b6 (Feb 2025, beta)
- **Python Requirement**: >=3.10
- **Architecture Change**: Complete rewrite from TorchHook to "Datasite" architecture
- **Major Breaking Changes**: Old tutorials and code from 2020 era are completely incompatible

### Key Changes

1. **Framework Support**:
   - 2020: PyTorch focus, experimental TensorFlow support via syft-tensorflow
   - 2025: Modern DataSite approach, primarily PyTorch, TensorFlow support status unclear

2. **API Changes**:
   - Complete API redesign
   - Old TorchHook-based approach deprecated
   - New focus on remote data science and privacy preservation

3. **Compatibility**:
   - 2020: PyTorch 1.4.0, Python 3.6-3.7
   - 2025: Modern PyTorch, Python 3.10+

## Modernization Recommendations

### Option 1: TensorFlow Federated (Recommended for Archive)

**Pros**:

- Original codebase already uses TFF
- Mature TensorFlow 2.x support
- Google-backed framework
- Simulation approach well-established

**Cons**:

- TFF can be complex
- Simulation-only (not true distributed in original code)

**Strategy**: Pin to TF 2.10-2.11, TFF 0.40-0.50, Python 3.8-3.9

### Option 2: Modern PySyft (For Modern Recreation)

**Pros**:

- Active development (2025 releases)
- Privacy-preserving capabilities
- Modern architecture

**Cons**:

- Requires complete rewrite (API incompatible with 2020)
- Would need conversion to PyTorch ecosystem
- Steep learning curve for new architecture

**Strategy**: Python 3.10+, PyTorch 2.x, PySyft 0.9.x, complete code rewrite

### Option 3: Flower Framework (Alternative)

**Pros**:

- Modern federated learning framework
- Framework agnostic (works with TensorFlow, PyTorch, JAX)
- Active community
- Better documentation than PySyft

**Cons**:

- Would require refactoring
- Different architecture from original

**Strategy**: Consider for future modernization

## Recommended Path Forward

### Phase 1: Archive Branch

- Preserve original TensorFlow + TFF approach
- Pin dependencies to 2020-era compatible versions:
  - Python 3.8
  - TensorFlow 2.10
  - TensorFlow Federated 0.40.0
  - Pandas <1.4.0
  - No PySyft (it was never actually used)

### Phase 2: Modern Recreation

- **Quick Win**: Update TensorFlow/TFF to modern versions (TF 2.15, TFF 0.60+)
  - Less risky, same ecosystem
  - Fix deprecated API calls
  - Modern Python 3.11

- **Alternative Path**: Convert to Flower framework
  - More modern FL approach
  - Better for production
  - Framework flexibility

- **Skip**: PySyft conversion
  - Too much work for uncertain benefit
  - Original code never actually used it
  - API completely different now

## References

- Original paper: <https://www.sciencedirect.com/science/article/pii/S2666827022000081>
- KSU thesis: <https://digitalcommons.kennesaw.edu/cgi/viewcontent.cgi?article=1044&context=cs_etd>
- PySyft GitHub: <https://github.com/OpenMined/PySyft>
- TensorFlow Federated: <https://www.tensorflow.org/federated>
- Flower: <https://flower.dev/>

## Conclusion

**Original project used TensorFlow Federated, NOT PySyft**. PySyft was attempted but never implemented. Modernization should focus on updating TFF stack or considering Flower framework, not PySyft migration.
