A **GAN**, or **Generative Adversarial Network**, is a type of machine learning framework developed by **Ian Goodfellow** and his colleagues in 2014. It’s used primarily for **generating new data that is similar to a given dataset**, such as creating realistic images, music, or even text.

### 🧠 **Core Idea**
A GAN consists of **two neural networks** that are trained together in a kind of **game**:

1. **Generator (G)** – This network tries to create fake data that looks real.
2. **Discriminator (D)** – This one tries to tell the difference between real data and the fake data produced by the generator.

These two models are in a **minimax game**:
- The **Generator** tries to fool the **Discriminator**.
- The **Discriminator** tries to correctly classify data as real or fake.

Training continues until the generator gets so good at faking data that the discriminator can no longer tell the difference.

---

### 🔄 How GANs Work (Simple Analogy):

Imagine a **counterfeit artist** (Generator) trying to make fake paintings, and an **art critic** (Discriminator) trying to spot the fakes. Over time, the artist gets better at faking, and the critic gets better at spotting. Eventually, the artist becomes so skilled that the critic can no longer tell real from fake.

---

### 🔍 Applications of GANs:
- **Image generation**: Faces, artworks, scenes (e.g., [ThisPersonDoesNotExist.com](https://thispersondoesnotexist.com))
- **Image-to-image translation**: Converting sketches to photos, night to day
- **Super-resolution**: Enhancing image quality
- **Data augmentation**: Creating synthetic data for training ML models
- **Deepfake technology**: Realistic video and voice generation (also raises ethical concerns)

---

If latent_dim = 100, the generator will take a 100-dimensional random noise vector as input and produce a generated sample (e.g., a signal or image).
