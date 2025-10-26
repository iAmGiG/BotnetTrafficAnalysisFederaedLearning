# Reference Materials

This directory contains reference materials and documentation used during the research.

## Files

### `thesis.pdf`

**Description**: Reference thesis document
**Attribution**: This is a reference document, NOT the thesis produced by this project
**Note**: Included for research context and comparison purposes

### `N_BaIoT_dataset_description.txt`

**Description**: Official dataset description for the N-BaIoT dataset
**Source**: UCI Machine Learning Repository
**URL**: <https://archive.ics.uci.edu/ml/machine-learning-databases/00442/>
**Citation**:

```bash
Y. Meidan, M. Bohadana, Y. Mathov, Y. Mirsky, D. Breitenbacher, A. Shabtai, and Y. Elovici
"N-BaIoT: Network-based Detection of IoT Botnet Attacks Using Deep Autoencoders"
IEEE Pervasive Computing, Special Issue - Securing the IoT (July/Sep 2018)
```

**Contents**:

- Dataset creators and source information
- Past usage and citations
- Number of instances and attributes (115 features)
- IoT devices included (9 commercial devices)
- Attack types: Mirai and BASHLITE botnets
- Evaluation methodology and results

## Published Research

This project resulted in a published paper:

**Title**: Detecting, Classifying and Explaining IoT Botnet Attacks Using Deep Learning Methods Based on Network Data

**Published**:

- ScienceDirect: <https://www.sciencedirect.com/science/article/pii/S2666827022000081>
- KSU Digital Commons: <https://digitalcommons.kennesaw.edu/cgi/viewcontent.cgi?article=1044&context=cs_etd>

**Institution**: Kennesaw State University, Department of Computer Science

**Abstract**: The growing adoption of Internet-of-Things devices brings with it the increased participation of said devices in botnet attacks, and as such novel methods for IoT botnet attack detection are needed. This work demonstrates that deep learning models can be used to detect and classify IoT botnet attacks based on network data in a device-agnostic way and that it can be more accurate than some more traditional machine learning methods, especially without feature selection. Furthermore, this work shows that the opaqueness of deep learning models can be mitigated to some degree with the Local Interpretable Model-agnostic Explanations (LIME) technique.

## Related Materials

For information on the federated learning experimental approaches, see:

- `../archived/experimental/README.md` - Experimental FL attempts
- `../../PYSYFT_RESEARCH.md` - PySyft and modernization research
- `../../anomaly-detection/README++.MD` - Development notes and FL details
