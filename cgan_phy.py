# %%
import shutil
import pandas as pd
import csv
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import lognorm
from scipy.signal import welch
from sklearn.model_selection import train_test_split
import tensorflow as tf
from statsmodels.tsa.stattools import acf
from scipy.fft import rfft
from tensorflow.keras.utils import to_categorical
from scipy.signal import find_peaks
import os
import matplotlib.pyplot as plt
import numpy as np
from google.colab import drive
drive.mount('/content/drive')

# %%

# %%
# --- 1. Simulation Function ---


def simulate_black_hole_lightcurve(fs, fc, fm, qpo_amplitude, duration,
                                   noise_mean=0, noise_std=0.5,
                                   include_qpo=True, modulation_index=0.5):
    """
    Simulate a black hole light curve with stochastic noise and an amplitude-modulated QPO signal.

    Parameters:
        fs : int
            Sampling frequency (Hz)
        fc : float
            Carrier frequency (Hz) for QPO
        fm : float
            Modulating frequency (Hz) for QPO
        qpo_amplitude : float
            Amplitude of the carrier signal (QPO)
        duration : float
            Duration of lightcurve (seconds)
        noise_mean : float
            Mean of the Gaussian noise
        noise_std : float
            Standard deviation of the Gaussian noise
        include_qpo : bool
            Whether to include the QPO signal
        modulation_index : float
            Modulation index for AM signal

    Returns:
        t : np.ndarray
            Time array
        flux : np.ndarray
            Normalized flux array
    """
    # Time array
    t = np.arange(0, duration, 1/fs)

    # White noise
    white_noise = np.random.normal(noise_mean, noise_std, size=len(t))
    white_noise = np.exp(white_noise)

    if include_qpo and qpo_amplitude > 0:
        # Modulating signal
        msg = qpo_amplitude * np.cos(2 * np.pi * fm * t)

        # Carrier signal
        carrier = qpo_amplitude * np.cos(2 * np.pi * fc * t)

        # AM QPO signal
        qpo = carrier * (1 + modulation_index * msg / qpo_amplitude)
    else:
        qpo = 0

    # Combine noise and QPO signal
    flux = white_noise + qpo

    # Normalize
    flux = (flux - np.mean(flux)) / np.std(flux)

    return t, flux

# %%


def generate_dataset_with_fc_amp(output_dir, num_samples=5000,
                                 fs=1, duration=512,
                                 modulation_index=0.5,
                                 amp_range=(0.1, 1.0)):
    """
    Generate a dataset of light curves with fc and amp labels for conditional GAN/SBI.

    Stores:
        X: (num_samples, 512, 1) - light curves
        y: (num_samples, 2) - [fc, amp]

    Parameters:
        output_dir : str
            Path to save dataset
        num_samples : int
            Total number of samples
        fs : int
            Sampling frequency
        duration : int
            Light curve length in seconds
        modulation_index : float
            AM index for QPO
        amp_range : tuple
            Amplitude range (min, max)
    """
    os.makedirs(output_dir, exist_ok=True)
    seq_length = int(duration * fs)
    X, y = [], []

    for _ in range(num_samples):
        fc = np.random.uniform(0.01, 0.5)  # Due to Nyquist Frequency (fs/2)
        fm = np.random.uniform(0.005, 0.1)
        amp = np.random.uniform(*amp_range)

        # Generate QPO light curve
        t, flux_qpo = simulate_black_hole_lightcurve(
            fs, fc, fm, amp, duration, include_qpo=True,
            modulation_index=modulation_index
        )
        X.append(flux_qpo[:seq_length].reshape(-1, 1))
        y.append([fc, amp])  # label: [fc, amp]

    # Convert and save
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    np.savez_compressed(os.path.join(output_dir, "data_fc_amp.npz"), X=X, y=y)
    print(
        f" Saved dataset to {output_dir}/data_fc_amp.npz with shape X: {X.shape}, y: {y.shape}")


# %%
output_folder = "qpo_physical_dataset/dataset_fc_amp"

generate_dataset_with_fc_amp(
    output_dir=output_folder,
    num_samples=5000,
    amp_range=(0.6, 1.0)
)

# %%


def load_physical_conditional_dataset(npz_path, batch_size=64, test_size=0.2):
    """
    Load dataset with labels = [fc, amp, is_qpo]
    Returns tf.data.Dataset of (X, [fc, amp, is_qpo])
    """
    data = np.load(npz_path)
    X = data["X"].astype(np.float32)         # shape: (N, 512, 1)
    y = data["y"].astype(np.float32)         # shape: (N, 3)

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Make tf.data datasets
    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)).shuffle(10000).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices(
        (X_val, y_val)).batch(batch_size)

    return train_ds, val_ds

# %%


def build_full_conditional_generator(latent_dim=100, condition_dim=2, seq_length=512):
    """
    Builds a conditional generator for time series data.

    Parameters:
    -----------
    latent_dim : int, default=100
        Dimension of the noise input vector
    condition_dim : int, default=2
        Dimension of the condition vector (fc, amp)
    seq_length : int, default=512
        Length of the output time series

    Returns:
    --------
    tf.keras.Model
        Conditional generator model
    """
    # Input layers
    noise_input = tf.keras.Input(shape=(latent_dim,))
    condition_input = tf.keras.Input(shape=(condition_dim,))  # fc, amp

    # Combine noise and condition vectors
    x = tf.keras.layers.Concatenate()([noise_input, condition_input])

    # Initial dense layers for processing combined input
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.LeakyReLU()(x)  # Non-linear activation

    # Project to sequence length
    x = tf.keras.layers.Dense(seq_length * 32)(x)  # 32 features per time step
    x = tf.keras.layers.Reshape((seq_length, 32))(
        x)  # Reshape to time series format

    # Add noise for regularization
    x = tf.keras.layers.GaussianNoise(0.05)(x)

    # GRU layer for temporal coherence
    x = tf.keras.layers.GRU(64, return_sequences=True)(x)

    # Output layer: one value per time step
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(x)

    return tf.keras.Model([noise_input, condition_input], output)


def build_full_conditional_discriminator(seq_length=512, condition_dim=2):
    """
    Builds a conditional discriminator for time series data.

    Parameters:
    -----------
    seq_length : int, default=512
        Length of the input time series
    condition_dim : int, default=2
        Dimension of the condition vector (fc, amp)

    Returns:
    --------
    tf.keras.Model
        Conditional discriminator model
    """
    # Input layers
    series_input = tf.keras.Input(shape=(seq_length, 1))
    condition_input = tf.keras.Input(shape=(condition_dim,))  # fc, amp

    # Expand conditions to match sequence length
    condition_expanded = tf.keras.layers.RepeatVector(
        seq_length)(condition_input)  # Repeat conditions for each time step

    # Combine time series with conditions
    x = tf.keras.layers.Concatenate()([series_input, condition_expanded])

    # GRU layers for temporal processing
    x = tf.keras.layers.GRU(64, return_sequences=True)(x)
    x = tf.keras.layers.GRU(32)(x)  # Final GRU layer returns single vector

    # Dense layers for classification
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)  # Regularization

    # Output layer: single value for real/fake classification
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model([series_input, condition_input], output)


class ConditionalTimeSeriesGAN(tf.keras.Model):
    """
    Conditional GAN for time series generation with QPOs.
    Implements the training loop for both generator and discriminator.
    """

    def __init__(self, generator, discriminator, latent_dim):
        """
        Initialize the GAN with its components.

        Parameters:
        -----------
        generator : tf.keras.Model
            The generator model
        discriminator : tf.keras.Model
            The discriminator model
        latent_dim : int
            Dimension of the noise input vector
        """
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        # Small learning rate for stable training
        self.gen_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.disc_optimizer = tf.keras.optimizers.Adam(1e-4)

    def compile(self):
        super().compile()

    @tf.function  # Compile function for faster execution
    def train_step(self, data):
        """
        Single training step for both generator and discriminator.

        Parameters:
        -----------
        data : tuple
            Contains (real_series, labels)
        """
        real_series, labels = data
        batch_size = tf.shape(real_series)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])

        # === Train Discriminator ===
        with tf.GradientTape() as disc_tape:
            # Generate fake samples
            fake_series = self.generator([noise, labels], training=True)

            # Get discriminator outputs for real and fake samples
            real_output = self.discriminator(
                [real_series, labels], training=True)
            fake_output = self.discriminator(
                [fake_series, labels], training=True)

            # Calculate discriminator loss
            real_labels = tf.ones((batch_size, 1)) * 0.9  # Label smoothing
            fake_labels = tf.zeros((batch_size, 1))
            disc_loss = self.loss_fn(
                real_labels, real_output) + self.loss_fn(fake_labels, fake_output)

        # Update discriminator weights
        grads_disc = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(
            zip(grads_disc, self.discriminator.trainable_variables))

        # === Train Generator ===
        noise = tf.random.normal([batch_size, self.latent_dim])
        with tf.GradientTape() as gen_tape:
            # Generate new fake samples
            generated_series = self.generator([noise, labels], training=True)
            fake_output = self.discriminator(
                [generated_series, labels], training=True)
            # Generator tries to fool discriminator
            gen_loss = self.loss_fn(tf.ones((batch_size, 1)), fake_output)

        # Update generator weights
        grads_gen = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(
            zip(grads_gen, self.generator.trainable_variables))

        return {"gen_loss": gen_loss, "disc_loss": disc_loss}


# %%


def plot_psd(series, fs=1, label='PSD'):
    f, Pxx = welch(series, fs=fs, nperseg=256)
    plt.semilogy(f, Pxx, label=label)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title("Power Spectral Density")
    plt.legend()


def plot_acf(series, lags=100, label='ACF'):
    autocorr = acf(series, nlags=lags, fft=True)
    plt.plot(autocorr, label=label)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title("ACF")
    plt.legend()


def plot_flux_histogram(series, label='Generated'):
    shape, loc, scale = lognorm.fit(series - np.min(series) + 1e-6)
    x = np.linspace(np.min(series), np.max(series), 100)
    pdf = lognorm.pdf(x, shape, loc, scale)

    plt.hist(series, bins=40, density=True, alpha=0.6, label=f"{label} Flux")
    plt.plot(x, pdf, '--', label=f"LogNorm Fit ({label})")
    plt.xlabel("Flux")
    plt.ylabel("Density")
    plt.title("Flux Histogram with Log-normal Fit")
    plt.legend()


def compute_qpo_score(series):
    from scipy.signal import welch
    from scipy.optimize import curve_fit

    def lorentz(f, A, f0, gamma):
        return A / (1 + ((f - f0)/gamma)**2)

    f, Pxx = welch(series, fs=1, nperseg=256)
    mask = (f > 0.01) & (f < 0.45)
    try:
        p0 = [np.max(Pxx[mask]), f[mask][np.argmax(Pxx[mask])], 0.01]
        popt, _ = curve_fit(lorentz, f[mask], Pxx[mask], p0=p0, maxfev=2000)
        Q = popt[1] / popt[2]
        return float(Q)
    except:
        return 0.0


# %%


class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, generator, val_dataset, latent_dim, num_samples=3, label=[0.3, 0.9], save_dir="gan_outputs", csv_log_path="saved_models/qpo_scores.csv"):
        super().__init__()
        self.generator = generator
        self.val_dataset = val_dataset
        self.latent_dim = latent_dim
        self.num_samples = num_samples
        self.label = tf.convert_to_tensor(
            [label] * num_samples, dtype=tf.float32)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.csv_log_path = csv_log_path
        os.makedirs(save_dir, exist_ok=True)

        #  Create CSV file header
        with open(self.csv_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "sample_q_score"])

    def on_epoch_end(self, epoch, logs=None):
        noise = tf.random.normal([self.num_samples, self.latent_dim])
        generated = self.generator(
            [noise, self.label], training=False).numpy().squeeze()
        real_samples = next(iter(self.val_dataset))[
            0].numpy().squeeze()[:self.num_samples]

        pdf_path = os.path.join(self.save_dir, f"epoch_{epoch+1}.pdf")
        with PdfPages(pdf_path) as pdf:
            for i in range(self.num_samples):
                fig, axes = plt.subplots(1, 3, figsize=(18, 4))

                plt.sca(axes[0])
                plot_psd(generated[i], label='Generated')
                plot_psd(real_samples[i], label='Real')

                plt.sca(axes[1])
                plot_acf(generated[i], label='Generated')
                plot_acf(real_samples[i], label='Real')

                plt.sca(axes[2])
                plot_flux_histogram(generated[i], label='Generated')
                plot_flux_histogram(real_samples[i], label='Real')

                plt.suptitle(f"Sample {i + 1} – Epoch {epoch + 1}")
                plt.tight_layout()

                #  Save this figure as one page in the PDF
                pdf.savefig(fig)
                plt.show(fig)
                plt.close(fig)
        q_score = compute_qpo_score(generated[0])
        print(f"Epoch {epoch+1} – Sample Q: {q_score}")
        print(f"Saved PDF for epoch {epoch + 1} at: {pdf_path}")
        #  Save to CSV

        with open(self.csv_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, q_score])


# %%
os.makedirs("saved_models", exist_ok=True)

# csv_logger = tf.keras.callbacks.CSVLogger("saved_models/conditional_gan_training_log.csv", append=False)

# %%


class ValidationLossLogger(tf.keras.callbacks.Callback):
    def __init__(self, val_dataset, latent_dim, generator, discriminator, save_dir="saved_models", csv_path="saved_models/cgan_phy_log_with_val.csv"):
        super().__init__()
        self.val_dataset = val_dataset
        self.latent_dim = latent_dim
        self.generator = generator
        self.discriminator = discriminator
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.csv_path = csv_path
        self.save_dir = save_dir

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "gen_loss", "disc_loss",
                            "val_gen_loss", "val_disc_loss"])

    def on_epoch_end(self, epoch, logs=None):
        val_gen_loss_total = 0.0
        val_disc_loss_total = 0.0
        batches = 0

        for real_series, labels in self.val_dataset:
            batch_size = tf.shape(real_series)[0]
            noise = tf.random.normal([batch_size, self.latent_dim])
            fake_series = self.generator([noise, labels], training=False)

            # Discriminator loss on validation data
            real_output = self.discriminator(
                [real_series, labels], training=False)
            fake_output = self.discriminator(
                [fake_series, labels], training=False)
            real_labels = tf.ones((batch_size, 1)) * 0.9
            fake_labels = tf.zeros((batch_size, 1))
            disc_loss = self.loss_fn(
                real_labels, real_output) + self.loss_fn(fake_labels, fake_output)

            # Generator loss on validation data
            fake_output = self.discriminator(
                [fake_series, labels], training=False)
            gen_loss = self.loss_fn(tf.ones((batch_size, 1)), fake_output)

            val_gen_loss_total += gen_loss.numpy()
            val_disc_loss_total += disc_loss.numpy()
            batches += 1

        avg_gen = val_gen_loss_total / batches
        avg_disc = val_disc_loss_total / batches

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                logs["gen_loss"],
                logs["disc_loss"],
                avg_gen,
                avg_disc
            ])
        print(
            f" Logged validation losses (gen: {avg_gen:.4f}, disc: {avg_disc:.4f})")
        # Save models on each epoch
        self.generator.save(os.path.join(
            self.save_dir, f"generator_{epoch + 1}.keras"))
        self.discriminator.save(os.path.join(
            self.save_dir, f"discriminator_{epoch + 1}.keras"))


# %%
# Load data
train_ds, val_ds = load_physical_conditional_dataset(
    npz_path="qpo_physical_dataset/dataset_fc_amp/data_fc_amp.npz"
)

# Build models
latent_dim = 100
generator = build_full_conditional_generator(
    latent_dim=latent_dim, condition_dim=2)
discriminator = build_full_conditional_discriminator(condition_dim=2)


gan = ConditionalTimeSeriesGAN(generator, discriminator, latent_dim)
gan.compile()


monitor = GANMonitor(generator, val_dataset=val_ds,
                     # sample QPO
                     latent_dim=latent_dim, label=[0.3, 0.9])

val_logger = ValidationLossLogger(
    val_dataset=val_ds, latent_dim=100, generator=generator, discriminator=discriminator, save_dir="saved_models")

gan.fit(train_ds, epochs=100, callbacks=[monitor, val_logger])

# %%

df = pd.read_csv("saved_models/qpo_scores.csv")
plt.plot(df["epoch"], df["sample_q_score"], marker='o')
plt.title("QPO Score Across Epochs")
plt.xlabel("Epoch")
plt.ylabel("Lorentzian Q (Sharpness of QPO Peak)")
plt.grid(True)
plt.tight_layout()
plt.show()


# %%

# Save generator and discriminator after training
os.makedirs("saved_models", exist_ok=True)
generator.save("saved_models/qpo_cgan_phy_generator.keras")
discriminator.save("saved_models/qpo_cgan_phy_discriminator.keras")
# shutil.copyfile('./conditional_gan_training_log.csv', 'drive/MyDrive/saved_models/cgan_phy_training_log.csv')

# %%
# from tensorflow.keras.models import load_model

# # Load generator and discriminator
# generator = load_model("saved_models/generator_bigru.h5", compile=False)
# discriminator = load_model("saved_models/discriminator_bigru.h5", compile=False)

# generator.summary()  # Optional check

# %%
!zip - r / content/drive/MyDrive/saved_models/gan_outputs.zip / content/gan_outputs/
!zip - r / content/drive/MyDrive/saved_models/saved_models.zip / content/saved_models/
