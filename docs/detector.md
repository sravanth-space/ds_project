
# 🧠 1. SBI Posterior Inference – `posterior.sample(...)`

### 🔧 What it does:
Infers the **posterior distribution** of the physical parameters:
- `fc` = center frequency of QPO (in Hz)
- `amp` = amplitude of QPO

Given an observed PSD, the posterior returns a **distribution over these parameters**.

---

### 🔢 Input:
```python
x_obs = torch.tensor(Pxx, dtype=torch.float32)  # observed PSD
samples = posterior.sample((500,), x=x_obs)
```

| Parameter        | Meaning                                             |
|------------------|-----------------------------------------------------|
| `x_obs`          | The observed **PSD** from Welch transform (your data) |
| `500`            | Number of posterior samples to draw                |

---

### 📤 Output:
```python
fc_samples = samples[:, 0].numpy()
amp_samples = samples[:, 1].numpy()
```

These are **500 inferred values** of:
- `fc`: Estimated QPO frequency
- `amp`: Estimated amplitude

---

### 💡 Why this approach?
- Uses **Bayesian simulation-based inference** (SBI) → works well when likelihood is intractable.
- Trained using realistic synthetic signals (from your GAN).
- No need to hand-design thresholds → SBI learns from data.

---

# 🔍 2. Lorentzian Fit – `compute_lorentzian_q()`

### 🔧 What it does:
Fits a **Lorentzian curve** to the PSD peak to measure how sharp and centered the QPO is — this gives the **Q-factor**:

```python
Q = f0 / gamma
```

- `f0` = peak frequency
- `gamma` = width of peak (smaller = sharper)

---

### 🔢 Input:
```python
curve        # light curve data (1D)
fs           # sampling frequency (1 Hz for GAN, also used for real)
f_window     # frequency range to search in
```

---

### 📤 Output:
```python
Q-value (float)
```

### 💡 Why Lorentzian?
- QPOs have a **quasi-periodic structure** → not infinite narrow spike.
- Lorentzian fits describe such peaks mathematically.
- Q-factor is a common astrophysical metric of oscillation quality.

---

# ⚖️ 3. Scoring Function – Decision Logic

### 🔧 What it does:
Combines multiple features (`Q`, `amp_mean`, `fc_std`) into one **decision score**:
```python
score = Q / 3 + amp_mean - fc_std
has_qpo = score > 1.4
```

---

### 🔢 Input Features:
| Feature         | From                     | Meaning                                          |
|------------------|--------------------------|--------------------------------------------------|
| `Q`             | From Lorentzian fit       | Sharpness of spectral peak                      |
| `amp_mean`      | From posterior samples    | Mean estimated QPO amplitude                    |
| `fc_std`        | From posterior samples    | Uncertainty in estimated QPO frequency          |

---

### 🧠 Logic Behind This Formula:
- `Q / 3`: Normalize Q to scale ~1
- `+ amp_mean`: Boost if amplitude is high
- `- fc_std`: Penalize if frequency estimation is uncertain

**Score > 1.4** → detected as QPO.  
(You can adjust this threshold based on validation data.)

---

# 📊 4. `detect_qpo_sbi()` – The Full Detector

### 🔧 What it does:
Brings it all together:
1. Compute PSD from input light curve.
2. Run SBI posterior inference to get `fc_samples`, `amp_samples`
3. Fit Lorentzian to get `Q`
4. Combine features into `score`
5. Decide if QPO is detected or not

---

### 🔢 Input:
```python
curve        # the light curve data (1D)
posterior    # trained SBI posterior model
fs           # sampling frequency (usually 1.0)
show_plot    # whether to display PSD
```

---

### 📤 Output:
```python
{
  "fc_mean": ...,   # Posterior mean of QPO frequency
  "fc_std": ...,    # Std. deviation of frequency
  "amp_mean": ...,  # Mean QPO amplitude
  "Q": ...,         # Q-factor from Lorentzian
  "qpo": ...,       # True/False for detection
  "score": ...,     # Confidence score
  "samples": ...    # Full posterior samples
}
```

---

### 💡 Why This Full Pipeline?

| Step                  | Why It's Used                                                                 |
|------------------------|-------------------------------------------------------------------------------|
| **PSD with Welch**     | Well-tested method to estimate frequency content                             |
| **SBI Posterior**      | Captures uncertainty + handles nonlinear mapping from PSD → parameters       |
| **Lorentzian Fit**     | Gives interpretable peak quality metric (Q)                                  |
| **Score Function**     | Combines soft evidence from multiple channels for robust decision-making     |

---

## 🔍 Why Not Just Threshold on Q or amp?

Because:
- Q alone fails when the fit breaks (Band 4)
- amp_mean alone isn’t reliable in noisy cases
- fc_std captures model confidence — very useful!

So a **score-based soft rule** works better.

---

## 🧪 Example: Band 2

```text
Q = 5.52
amp_mean = 0.560
fc_std = 0.251
→ score = 5.52/3 + 0.560 - 0.251 = 2.15 ✅
```

Detected confidently as QPO!

---

## 🧠 Summary (for Viva)

> “We detect QPOs by computing a power spectral density (PSD) of each light curve and using simulation-based inference (SBI) to estimate the underlying physical parameters (frequency and amplitude). We also fit a Lorentzian to the PSD peak to compute the Q-factor. These features are combined into a score function that captures oscillation strength, frequency sharpness, and model uncertainty. A threshold on this score determines whether a QPO is present.”
