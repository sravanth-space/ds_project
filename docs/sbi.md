Certainly! Here's the same explanation formatted as **concise technical documentation notes** — ideal for use in a research notebook, report, or as inline commentary.

---

## 🧠 **SBI Inference Method Notes (SNPE-based)**

### Overview:
This project uses **Simulation-Based Inference (SBI)** with the `sbi` Python library.  
Specifically, we implement:

> **Sequential Neural Posterior Estimation (SNPE)**

---

### ✅ Method Summary:

| Component                     | Description |
|------------------------------|-------------|
| **Inference Type**           | Bayesian |
| **Approach**                 | Likelihood-free posterior estimation |
| **Algorithm Used**           | SNPE (Sequential Neural Posterior Estimation) |
| **Posterior Representation** | Neural density estimator (e.g., MAF, MDN, NSF) |
| **Sampling Strategy**        | Direct Monte Carlo sampling from learned posterior |
| **Simulator**                | GAN-generated light curves (QPO conditioned) |
| **Observation**              | PSD (Power Spectral Density) of input light curve |

---

### 🧪 Sampling Details:

- Posterior is learned from `(θ, x)` pairs, where:
  - `θ` = `[fc, amp]` (latent parameters)
  - `x` = PSD (from GAN or real data)
- After training, the posterior is queried like:

```python
posterior.sample((500,), x=x_obs)
```

- This performs **direct Monte Carlo sampling** from the neural posterior:
  - Not MCMC (e.g., no chains, burn-in, etc.)
  - Sampling is fast, independent, and scalable

---

### ❌ Not Using:

- Markov Chain Monte Carlo (MCMC)
- Approximate Bayesian Computation (ABC)
- Traditional likelihood-based methods

---

### ✅ Advantages of SNPE:

- Efficient inference on **high-dimensional simulators**
- Learns a flexible **parametric posterior** over `θ`
- Does not require likelihood computation
- Supports sequential refinement of the posterior

---

### 🧠 Behind the Scenes (by default):

- Learns `p(θ | x)` using:
  - MAF (Masked Autoregressive Flow)
  - or Mixture Density Network (MDN), depending on config
- Posterior returned is a PyTorch object that supports:
  - `.sample()`
  - `.log_prob()`
  - `.condition_on_observation(x)`

---