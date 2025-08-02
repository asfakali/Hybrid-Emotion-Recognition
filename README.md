# EEG-based Hybrid Emotion Recognition Model with Statistical-Wavelet Features and Modality-Agnostic Loss

This is the official implementation of the paper:

**"EEG-based Hybrid Emotion Recognition Model with Statistical-Wavelet Features and Modality-Agnostic Loss" presented in EAAAI (ex EANN) 2025 26th Engineering Applications and Advances of Artificial Intelligence** 

---

## 🧠 Overview

This paper introduces a **hybrid EEG-based emotion recognition framework** that combines **Discrete Wavelet Transform (DWT)**, **statistical feature extraction**, and **deep learning models (CNN, LSTM, ANN)**. It also introduces a **Modality-Agnostic Consistency Loss (MACL)** to enforce alignment across multi-modal features, improving robustness and generalization across different datasets.

---

## 📊 Key Contributions

- ⚙️ **Multi-modal feature extraction** using:
  - ANN for **statistical features**
  - CNN for **spatial features**
  - LSTM for **temporal features**

- 🌊 **Wavelet Transform** used to retain high-informative, low-frequency EEG components (LL band).

- 🧠 **Multi-head attention mechanism** integrating **self-attention** and **cross-attention** for inter-feature learning.

- 🎯 **Modality-Agnostic Consistency Loss (MACL)** combines:
  - Categorical Cross-Entropy Loss (CCE)
  - Feature Alignment Loss (based on MSE or Cosine distance)

- 📈 **Benchmark results** on three datasets:
  - **SEED** (100% accuracy)
  - **CASE** (93.50% accuracy)
  - **EEG Brainwave** (95.99% accuracy)

---

## 🔧 Methodology
![architecture](https://github.com/asfakali/Hybrid-Emotion-Recognition/blob/main/assets/emotion_eann.png)
### 🧩 1. Discrete Wavelet Transform (DWT)
- Input EEG signal decomposed into sub-bands: LL, LH, HL, HH
- **LL band** used for reduced noise and enhanced feature quality.

### 🔍 2. Feature Extractors
- **ANN**: Processes 8 statistical features (mean, median, std, skewness, etc.)
- **1D CNN**: Captures local spatial features from wavelet-transformed signal.
- **LSTM**: Captures long-term temporal dependencies from CNN output.

### 🧠 3. Attention Mechanism
- **Self-attention** within each modality (ANN, CNN, LSTM).
- **Cross-attention** across modalities to capture inter-modal relationships.
- Outputs are concatenated into a **final feature vector**.

### 🧮 4. Modality-Agnostic Consistency Loss (MACL)
```math
L_{MAC} = \lambda_1 \cdot L_{CE} + \lambda_2 \cdot L_{align}
````

* Encourages feature **coherence** between modalities.
* Helps reduce **redundancy** and improves **generalization**.

![loss](https://github.com/asfakali/Hybrid-Emotion-Recognition/blob/main/assets/loss_eann.png)
---

## 📚 Datasets

* **SEED**
  3 emotion classes: positive, neutral, negative
  15 participants, 3 sessions, 62-channel EEG

* **CASE**
  4 emotions: funny, boring, calming, frightening
  Multimodal (EEG, ECG, EMG, EDA, etc.)

* **EEG Brainwave**
  3 emotion classes: positive, neutral, negative
  Collected via Muse headband (TP9, AF7, AF8, TP10)

---

## 📈 Results Summary

| Dataset       | Accuracy (%) | F1-Score (%) |
| ------------- | ------------ | ------------ |
| **SEED**      | 100.00       | 100.00       |
| **CASE**      | 93.50        | 93.50        |
| **Brainwave** | 95.99        | N/A          |

* t-SNE plots show **well-separated class clusters** across all datasets.
![tsne](https://github.com/asfakali/Hybrid-Emotion-Recognition/blob/main/assets/tsne_eann.png)
---

## 🧪 Ablation Study

| Dataset       | Distance Type | Accuracy (%) | F1-Score (%) |
| ------------- | ------------- | ------------ | ------------ |
| **CASE**      | Cosine        | 93.50        | 93.50        |
|               | MSE           | 81.63        | 81.63        |
|               | No Align Loss | 91.20        | 90.70        |
| **SEED**      | Cosine        | 100.00       | 100.00       |
| **Brainwave** | MSE           | 95.55        | 95.55        |

---

## ✍️ Authors

* **Asfak Ali** (Jadavpur University)
* **Jotiraditya Banerjee** (Jadavpur University)
* **Debam Saha** (Jadavpur University)
* **Akash Dutta** (NIT Durgapur)
* **Friedhelm Schwenker** (Ulm University, Germany)
* **Ram Sarkar** (Jadavpur University)

---

## 📌 Citation

If you use this work in your research, please cite:

```bibtex
@article{ali2025eeg,
  title={EEG-based Hybrid Emotion Recognition Model with Statistical-Wavelet Features and Modality-Agnostic Loss},
  author={Ali, Asfak and Banerjee, Jotiraditya and Saha, Debam and Dutta, Akash and Schwenker, Friedhelm and Sarkar, Ram},
  journal={To appear in EAAAI/EANN 2025},
  year={2025}
}
```
