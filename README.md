# A neural network based approach to detecting quasi-periodic oscillations from black holes


## Project Structure
### Main Notebooks
- `dataset_generator_am_signal.ipynb`: AM signal dataset generation
- `cgan_phy.ipynb`: Conditional GAN implementation for physical data
- `hyperparam.ipynb`: Hyperparameter optimization and analysis
- `sbi.ipynb`: Simulation-Based Inference implementation
- `qpo_detector.ipynb`: Main notebook for QPO detection algorithms

### Directories
- `experiments/`: Contains experimental notebooks and results
- `data/`: Data files and datasets
  - `qpo_physical_dataset/`: Physical dataset for training and testing
  - `ltcrv4bands_rej_dt100.dat`: Light curve data from galaxy called REJ1034+396(XMM-Newton) Revolution 3837
- `saved_models/`: Trained model checkpoints
- `utils/`: Utility functions and helper modules
- `papers/`: Related research papers
- `backups/`: Backup files
- `archive-usued-utils/`: Archived utility functions

### Data Files
- `ltcrv4bands_rej_dt100.dat`: Light curve data from galaxy called REJ1034+396(XMM-Newton) Revolution 3837
- Obs. ID 0865010101
- https://nxsa.esac.esa.int/nxsa-web/#search
## Key Features
- Conditional GAN for synthetic data generation
- Hyperparameter optimization
- Simulation-Based Inference for parameter estimation
- QPO Detection

## Installation
1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
The project workflow is organized in Jupyter notebooks following this sequence:

1. Data Generation and Preprocessing:
   - Use `dataset_generator_am_signal.ipynb` to generate synthetic AM signal data
   - Process real astronomical data from `data/ltcrv4bands_rej_dt100.dat`

2. Model Training and Optimization:
   - Use `cgan_phy.ipynb` to train the Conditional GAN model
   - Run `hyperparam.ipynb` for hyperparameter optimization
   - Check `saved_models/` for trained model checkpoints

3. Simulation-Based Inference (SBI):
   - Use `sbi.ipynb` to perform parameter inference
   - The posterior distributions are saved in `sbi_qpo_inference_results.csv`
   - Analyze inference results and parameter estimations

4. QPO Detection:
   - Use `qpo_detector.ipynb` to apply the trained models for QPO detection
   - Utilize the posterior distributions from SBI for improved detection
   - Validate results against real astronomical data

5. Additional Resources:
   - Check `experiments/` for experimental notebooks and results
   - Use `utils/` for helper functions

## Data
### Data Organization
The project data is organized in the `data/` directory:
- `qpo_physical_dataset/`: Contains physical datasets for training and testing
- `ltcrv4bands_rej_dt100.dat`: Light curve data from galaxy REJ1034+396 (XMM-Newton) Revolution 3837

### Data Generation and Processing
- `dataset_generator_am_signal.ipynb` generates synthetic AM signal data
- The synthetic data serves as the initial training set for the GAN
- Parameters are controlled to simulate various QPO characteristics

### Model Pipeline
- The trained GAN generator serves as a simulator for the SBI
- `cgan_phy.ipynb` trains the GAN model
- The GAN generator is then used directly in `sbi.ipynb` for training the SBI neural posterior
- This approach allows efficient parameter inference through the SBI framework

### Results and Analysis
- Final Model and trained weights are saved in `saved_models/`
- GAN QPO results in `gan_qpo_detection_results.csv`