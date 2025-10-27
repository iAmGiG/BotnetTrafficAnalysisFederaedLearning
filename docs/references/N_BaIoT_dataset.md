# N-BaIoT Dataset Documentation

## Dataset Title

**N-BaIoT: Data for network based detection of IoT botnet attacks**

---

## Source Information

### Creators

- **Yair Meidan**, Michael Bohadana, Yael Mathov, Yisroel Mirsky, Asaf Shabtai
  *Department of Software and Information Systems Engineering*
  Ben-Gurion University of the Negev, Beer-Sheva, 8410501, Israel

- **Dominik Breitenbacher**, Yuval Elovici
  *iTrust Centre of Cybersecurity*
  Singapore University of Technology and Design, 8 Somapah Rd, Singapore 487372

### Dataset Information

- **Donor**: Yair Meidan (<yairme@bgu.ac.il>)
- **Date**: March 2018
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/00442/)

---

## Citations

### Primary Paper (Dataset Description)

**N-BaIoT—Network-Based Detection of IoT Botnet Attacks Using Deep Autoencoders**

- **Authors**: Y. Meidan, M. Bohadana, Y. Mathov, Y. Mirsky, D. Breitenbacher, A. Shabtai, and Y. Elovici
- **Published**: IEEE Pervasive Computing, Special Issue - Securing the IoT (July/September 2018)
- **Volume**: 17, Issue 3, Pages 12-22
- **DOI**: [10.1109/MPRV.2018.03367731](https://doi.org/10.1109/MPRV.2018.03367731)
- **ArXiv**: [arXiv:1805.03409](https://arxiv.org/abs/1805.03409)
- **IEEE Xplore**: [Link](https://ieeexplore.ieee.org/document/8490192/)

**Abstract**: This paper proposes and empirically evaluates a novel network-based anomaly detection method which extracts behavior snapshots of the network and uses deep autoencoders to detect anomalous network traffic emanating from compromised IoT devices. The method was evaluated by infecting nine commercial IoT devices with Mirai and BASHLITE botnets, demonstrating the ability to accurately and instantly detect attacks.

### Feature Extraction Method (AfterImage/Kitsune)

**Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection**

- **Authors**: Y. Mirsky, T. Doitshman, Y. Elovici & A. Shabtai
- **Published**: Network and Distributed System Security (NDSS) Symposium 2018, San Diego, CA, USA
- **DOI**: [10.14722/ndss.2018.23211](https://doi.org/10.14722/ndss.2018.23211)
- **ArXiv**: [arXiv:1802.09089](https://arxiv.org/abs/1802.09089)
- **GitHub**: [Kitsune-py](https://github.com/ymirsky/Kitsune-py)

**Description**: This paper describes the feature extraction framework (AfterImage) used to convert network packet captures (*.pcap) to CSV files with 115 statistical features. AfterImage efficiently tracks patterns of every network channel using damped incremental statistics.

---

## Dataset Overview

### Purpose

This dataset addresses the lack of public botnet datasets for IoT devices. Unlike prior studies that relied on emulated or simulated data, this dataset enables empirical evaluation with **real traffic data** gathered from nine commercial IoT devices infected by authentic botnets in an isolated network.

### IoT Devices Included

The dataset contains network traffic from **9 commercial IoT devices**:

1. Danmini Doorbell
2. Ecobee Thermostat
3. Ennio Doorbell
4. Philips B120N/10 Baby Monitor
5. Provision PT-737E Security Camera
6. Provision PT-838 Security Camera
7. Samsung SNH 1011 N Webcam
8. SimpleHome XCS7 1002 WHT Security Camera
9. SimpleHome XCS7 1003 WHT Security Camera

### Attack Types

The dataset facilitates examination of **two major IoT botnets**:

- **Mirai**: One of the most widespread IoT botnets
- **BASHLITE (Gafgyt)**: Another common IoT botnet

**Attack Scenarios**: 10 different attack types carried by these 2 botnets

### Data Classification

- **Binary Classification**: Benign vs. Malicious traffic
- **Multi-class Classification**: 10 attack classes + 1 benign class = 11 total classes

---

## Dataset Characteristics

### Number of Instances

Varies for every device and attack combination.

### Number of Attributes

**115 independent features** in each CSV file, plus a class label derived from the filename (e.g., "benign" or "TCP attack").

---

## Feature Description

The features are extracted using the **AfterImage** framework, which creates statistical summaries of network traffic streams.

### Stream Aggregation Types

| Feature Prefix | Description (Paper Name) | Explanation |
|----------------|-------------------------|-------------|
| **H** | Source IP | Statistics summarizing recent traffic from this packet's host (IP address) |
| **MI** | Source MAC-IP | Statistics summarizing recent traffic from this packet's host (IP + MAC address) |
| **HH** | Channel | Statistics summarizing recent traffic from packet's source host to destination host |
| **HH_jit** | Channel Jitter | Statistics summarizing jitter of traffic from source to destination |
| **HpHp** | Socket | Statistics summarizing traffic from source host+port to destination host+port (e.g., 192.168.4.2:1242 → 192.168.4.12:80) |

### Time-Frame (Decay Factor Lambda)

How much recent history is captured in the statistics:

- **L5**: Lambda = 5
- **L3**: Lambda = 3
- **L1**: Lambda = 1
- **L0.1**: Lambda = 0.1
- **L0.01**: Lambda = 0.01

### Statistical Measures

| Statistic | Description |
|-----------|-------------|
| **weight** | Weight of the stream (number of items observed in recent history) |
| **mean** | Average value |
| **std** | Standard deviation |
| **variance** | Variance of the stream |
| **radius** | Root squared sum of two streams' variances |
| **magnitude** | Root squared sum of two streams' means |
| **cov** | Approximated covariance between two streams |
| **pcc** | Approximated Pearson correlation coefficient between two streams |

### Feature Naming Convention

Features follow the pattern: `{StreamType}_{TimeFrame}_{Statistic}`

Example: `HH_L5_mean` = Channel statistics with Lambda=5, mean value

---

## Original Study Results

### Methodology

- Trained and optimized a deep autoencoder on **2/3 of benign data** for each of the 9 IoT devices
- Goal: Capture normal network traffic patterns per device
- Test data comprised: **1/3 benign data + all malicious data**

### Results

- Applied trained autoencoders as anomaly detectors on test sets
- **Achieved 100% True Positive Rate (TPR)** for detecting cyberattacks from all compromised IoT devices
- Demonstrated device-agnostic detection capability

---

## Usage in This Project

This project uses the N-BaIoT dataset for:

1. **Anomaly Detection**: Training autoencoders to detect botnet traffic (benign vs. malicious)
2. **Classification**: Multi-class classification of attack types (Mirai, BASHLITE variants)
3. **Federated Learning**: Experimenting with distributed training across device-specific data
4. **Explainability**: Using LIME to interpret deep learning model decisions

### Data Download

Use the provided script:

```bash
python scripts/download_data.py
```

Downloads from: <https://archive.ics.uci.edu/ml/machine-learning-databases/00442/>

### Fisher Feature Selection

This project includes Fisher score-based feature selection (see `data/fisher/fisher.csv`) to reduce dimensionality from 115 features to top-N most discriminative features (typically 10-100 features).

---

## Related Files in This Repository

- **`demonstrate_structure.csv`**: Sample showing the 115 feature column headers
- **`data/fisher/fisher.csv`**: Fisher scores for feature importance ranking
- **`config/devices.json`**: Configuration for the 9 IoT devices with hyperparameters

---

## License & Usage Terms

Please cite the papers above when using this dataset.

**Recommended Citation**:

```
Y. Meidan, M. Bohadana, Y. Mathov, Y. Mirsky, D. Breitenbacher, A. Shabtai, and Y. Elovici,
"N-BaIoT—Network-Based Detection of IoT Botnet Attacks Using Deep Autoencoders,"
IEEE Pervasive Computing, vol. 17, no. 3, pp. 12-22, July/Sep. 2018.
doi: 10.1109/MPRV.2018.03367731
```

---

*Last Updated*: October 2024
*Original Dataset Release*: March 2018
