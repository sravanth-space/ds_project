# QPO Detection and Generation Project

## Overview
A neural network based approach to detecting quasi-periodic oscillations from black holes

## Project Structure
### Main Notebooks
- `dataset_generator_am_signal.ipynb`: AM signal dataset generation
- `amp_experiment.ipynb`: Amplitude experiment analysis
- `gan.ipynb`: Base GAN implementation experiments
- `conditional_gan_exp.ipynb`: Conditional GAN experiments
- `cgan_phy.ipynb`: Conditional GAN implementation for physical data
- `cgan_phy_generated_dataset.ipynb`: Generation of datasets using CGAN
- `diagrams.ipynb`: Project diagrams and visualizations
- `cgan_best.ipynb`: Best performing CGAN implementation
- `param.ipynb`: Parameter analysis and optimization
- `plot_hyperparam_runs.ipynb`: Visualization of hyperparameter optimization results
- `sbi.ipynb`: Simulation-Based Inference implementation
- `real_data_from_supervisor.ipynb`: Processing of real astronomical data
- `qpo_detector.ipynb`: Main notebook for QPO detection algorithms

### Directories
- `qpo_physical_dataset/`: Physical dataset for training and testing
- `gan_outputs/`: Directory containing generated samples and model outputs
- `saved_models/`: Trained model checkpoints
- `utils/`: Utility functions and helper modules
- `papers/`: Related research papers
- `backups/`: Backup files
- `archive-usued-utils/`: Archived utility functions

### Data Files
- `ltcrv4bands_rej_dt100.dat`: Light curve data from galaxy called REJ1034+396(XMM-Newton) Revolution 3837
  
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

1. GAN Training and Data Generation:
   - Start with `cgan_phy.ipynb` to train the Conditional GAN
   - Use `cgan_phy_generated_dataset.ipynb` to generate synthetic QPO data
   - Check `cgan_best.ipynb` for the best performing model configuration

2. Simulation-Based Inference (SBI):
   - Use `sbi.ipynb` to perform parameter inference on the generated data
   - The posterior distributions are saved in `sbi_qpo_inference_results.csv`
   - Analyze inference results and parameter estimations

3. QPO Detection:
   - Use `qpo_detector.ipynb` to apply the trained models for QPO detection
   - Utilize the posterior distributions from SBI for improved detection
   - Validate results against real astronomical data using `real_data_from_supervisor.ipynb`

4. Analysis and Visualization:
   - `plot_hyperparam_runs.ipynb` for analyzing model performance
   - `diagrams.ipynb` for visualizing results and creating project figures
   - `param.ipynb` for detailed parameter analysis

5. Additional Experiments:
   - `amp_experiment.ipynb` for amplitude-related studies
   - `dataset_generator_am_signal.ipynb` for AM signal analysis


## Data
### Synthetic Training Data
- `dataset_generator_am_signal.ipynb` generates synthetic AM signal data
- This synthetic data serves as the initial training set for the GAN
- Parameters are controlled to simulate various QPO characteristics

### GAN and SBI Pipeline
- The trained GAN generator serves as a simulator for the SBI
- `cgan_phy.ipynb` trains the GAN model
- The GAN generator is then used directly in `sbi.ipynb` for training the SBI neural posterior
- This approach allows efficient parameter inference through the SBI framework

### Real Astronomical Data
- Real QPO data is stored in `ltcrv4bands_rej_dt100.dat` 
- `real_data_from_supervisor.ipynb` processes and analyzes real QPO data
- Used for model validation and real-world QPO detection

### Results and Analysis
- Model checkpoints and trained weights are saved in `saved_models/`