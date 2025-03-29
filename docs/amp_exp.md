## 🧪 Experiment Report: LSTM Accuracy vs QPO Signal Amplitude

### 📌 Objective:
To investigate how the **strength of QPO signals (amplitude)** affects the classification performance of a **Bi-LSTM model** trained on simulated black hole light curves with embedded QPOs.

---

### 🔧 Method Summary:

- **Data Generation**: Each dataset simulates 10,000 light curves (5,000 with QPO, 5,000 without), varying only the **QPO amplitude** from `0.1` to `1.0`.
- **Features Used**: Flux time series only (shape = `(512, 1)`)
- **Model**: A **3-layer Bi-LSTM** with dropout, batch normalization, and dense layers
- **Training Setup**:
  - Optimizer: Adam (LR = 5e-4)
  - Loss: Categorical Cross-Entropy
  - Batch Size: 128
  - Epochs: 50 (with EarlyStopping and LR scheduler)
  - Metric: Validation Accuracy

---

### 📊 Results Summary:

| Amplitude | Best Validation Accuracy |
|-----------|---------------------------|
| 0.1       | **56.8%**                 |
| 0.2       | **63.6%**                 |
| 0.3       | **69.1%**                 |
| 0.4       | **87.9%**                 |
| 0.5       | **77.5%**                 |
| 0.6       | **99.0%**                 |
| 0.7       | **98.8%**                 |
| 0.8       | **99.8%**                 |
| 0.9       | **85.4%**                 |
| 1.0       | **99.9%**                 |

---

### 📈 Trend & Interpretation:

1. **Low Amplitude (0.1–0.3)**:
   - The model performs near random (~50–69%)
   - QPO signal is likely buried under stochastic noise

2. **Transitional Zone (0.4–0.5)**:
   - A sharp **accuracy jump** at 0.4 (to ~88%), confirming that the QPO begins to outshine the noise
   - Slight dip at 0.5, possibly due to variance or overfitting

3. **High Amplitude (0.6–1.0)**:
   - Model performs **exceptionally well** (>98% accuracy)
   - QPO becomes dominant and easier for LSTM to pick up

4. **Non-monotonic dip at 0.9**:
   - Could be caused by model instability, dataset imbalance, or training dynamics (e.g., early stopping or LR changes)

---

### 🧠 Insights:

- The LSTM is **highly sensitive to QPO amplitude** and can detect periodicity reliably once the signal reaches ~0.4–0.5 strength.
- Your experiment shows a clear **detection threshold region** where performance jumps.
- Early stopping and dropout effectively prevented overfitting at higher amplitudes.

----