1. **Stochastic noise** (to mimic astrophysical variability), and  
2. **Amplitude-modulated (AM) QPO** signal (to represent quasiperiodic oscillations).

---

##  Function Purpose

```python
simulate_black_hole_lightcurve(...)
```
Simulates a light curve with both random variability and a quasi-periodic oscillation (QPO), which is **amplitude-modulated**, a realistic form for black hole systems.

---

## 🧱 Key Concepts Used

| Component             | Meaning                                                                 |
|----------------------|-------------------------------------------------------------------------|
| `fs`                 | Sampling frequency (how many data points per second)                     |
| `fc`                 | Carrier frequency — the central QPO frequency (e.g., ~0.002 Hz)          |
| `fm`                 | Modulating frequency — frequency of the modulation envelope              |
| `modulation_index`   | Determines the strength of modulation (depth of envelope)                |
| `qpo_amplitude`      | How strong the QPO signal is                                             |
| `noise_std`          | Sets the background variability (like real astrophysical data)           |
| `include_qpo`        | Toggle to include/exclude the QPO component                              |

---

## What’s Happening Line-by-Line

### 1. **Time Array**
```python
t = np.arange(0, duration, 1/fs)
```
Creates a uniformly spaced time array from 0 to the desired duration, based on the sampling frequency.

---

### 2. **White Noise**
```python
white_noise = np.random.normal(noise_mean, noise_std, size=len(t))
white_noise = np.exp(white_noise)
```
Generates Gaussian white noise. The `np.exp()` transforms it into **log-normal noise**, mimicking multiplicative (non-Gaussian) astrophysical variability — a common approach in modeling light curves of accreting black holes.

---

### 3. **QPO Signal (if enabled)**
```python
msg = qpo_amplitude * np.cos(2 * np.pi * fm * t)  # Modulator
carrier = qpo_amplitude * np.cos(2 * np.pi * fc * t)  # Carrier
qpo = carrier * (1 + modulation_index * msg / qpo_amplitude)  # AM Signal
```
This uses **Amplitude Modulation (AM)**:
- The **carrier** is the base QPO signal.
- The **modulating signal** adds envelope variations, making it **quasi-periodic** instead of perfectly sinusoidal.
- The multiplication forms the final **AM QPO**.

If `include_qpo=False`, this section is skipped.

---

### 4. **Combine & Normalize**
```python
flux = white_noise + qpo
flux = (flux - np.mean(flux)) / np.std(flux)
```
- Adds QPO signal and noise together.
- Then **normalizes the flux** to zero mean and unit variance.

This makes it suitable for ML models or analysis tools like Lomb-Scargle or FFT.

---

## 🔁 Example Usage

```python
t, flux = simulate_black_hole_lightcurve(
    fs=1, fc=0.01, fm=0.001,
    qpo_amplitude=1.0, duration=1000,
    include_qpo=True
)

plt.plot(t, flux)
plt.title("Simulated Light Curve (With AM QPO)")
plt.xlabel("Time")
plt.ylabel("Flux")
plt.show()
```

---

## 🔬 Why Use AM for QPOs?

QPOs are **not pure sine waves** — their amplitude fluctuates over time. Using **amplitude modulation** realistically mimics:
- Irregular visibility,
- Bursty nature,
- Frequency spread seen in actual astrophysical QPOs.

This is an excellent approach for generating more **physically inspired training data**.

---