# Environment Installation Note

## TensorFlow Federated 0.40.0 Dependency Issues

TFF 0.40.0 has incompatible dependencies that prevent installation:

- Requires jaxlib==0.3.14 which doesn't exist for Python 3.9 on Windows
- Requires Python >=3.9 but jaxlib 0.3.14 predates Python 3.9 wheel builds
- This is a known TFF ecosystem issue from 2020

## Recommendation

Skip FL testing. Focus on:
1. Classification module (no TFF dependency)
2. Anomaly detection WITHOUT TFF (train_og.py works fine)

The FL code (train_v04.py) was never functional anyway (uses EMNIST, not botnet data).

