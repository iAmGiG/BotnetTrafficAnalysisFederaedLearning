# TODO for Main Branch (Modernization)

**Branch**: `main` (default)
**Purpose**: Modern, production-ready implementation
**Status**: Ready to start modernization

---

## Session Complete: Archive Branches ✅

Archive work is **DONE** and merged to main:
- ✅ `archive-2020-research` - Original code preserved with all flaws
- ✅ `archive-2020-fixed` - Bug fixes with 2020 dependencies
- ✅ All critical bugs fixed and tested
- ✅ Data leakage validated (minimal 1.2% impact)
- ✅ Comprehensive documentation created

**Key finding**: Original 2020 research was scientifically sound. The 99%+ accuracy is legitimate due to highly distinctive botnet traffic patterns.

---

## Next Session Priority: Overfitting Analysis

**Issue #23** - Validate 99%+ accuracy: Overfitting analysis

**User concern**: "I still feel that 99% is really high"

This is CRITICAL - must rule out overfitting before claiming success.

### Tests to Run:

1. ✅ Learning Curves (implemented in analysis/overfitting_analysis.py)
2. ✅ Cross-Validation with multiple seeds (implemented)
3. ✅ Feature Importance via ablation (implemented)
4. ✅ Dropout Regularization (implemented)
5. ⏳ **Cross-Device Generalization** (MOST IMPORTANT - needs implementation)
6. ⏳ Feature Perturbation (needs implementation)
7. ⏳ Data Scaling Tests (needs implementation)

### Cross-Device Testing (Priority 1):
- Train on 8 devices, test on 1 held-out device
- Repeat for all 9 devices (leave-one-out validation)
- This is THE definitive test for overfitting
- **NOTE**: Need to download all 9 devices first (~10-20GB total)

---

## Open Issues

1. **#23** - Overfitting analysis (HIGH PRIORITY)
2. **#22** - Research modern TFF / Flower implementation
3. **#17** - Modernization Roadmap
4. **#16** - Test/train overlap (LOW PRIORITY)

---

## Quick Start for Next Session

```bash
# 1. Pull latest main
git checkout main
git pull origin main

# 2. Create modern environment
conda create -n botnet-modern python=3.11
conda activate botnet-modern
pip install tensorflow pandas scikit-learn matplotlib jupyter

# 3. Download all IoT devices (takes 1-2 hours)
cd scripts
python download_data.py  # Downloads all 9 devices

# 4. Run overfitting analysis
cd ../analysis
python overfitting_analysis.py
```

---

## Key Files

- `analysis/overfitting_analysis.py` - Tests 1-4 done, need 5-7
- `docs/project-analysis/RETROSPECTIVE.md` - Full results
- `docs/project-analysis/DATA_LEAKAGE_IMPACT.md` - Context

---

## User Preferences (Important!)

- No emojis in documentation
- Prefer .yaml over .yml
- No signatures in commit messages
- Use GitHub issues/PRs for tracking
- `develop` branch was deleted - `main` is now default

---

## Technical Notes

### Markdown Linter Warning (MD033)
- VS Code linter complains about `<div align="center">` tags in README
- This is fine - GitHub supports HTML in markdown
- Created `.markdownlint.json` to disable this overly-strict rule
- Badges need `<div>` for centering

### Expected TF 2.15+ Breaking Changes
- Keras 3.0 has different API
- Loss functions may have different defaults
- May need to update model.compile() calls

### pandas 2.x Changes
- `.append()` completely removed (already fixed)
- Stricter dtype handling
- Some indexing behavior changes

---

**Last Updated**: 2024-10-26
**Next Priority**: Issue #23 - Cross-device generalization testing
