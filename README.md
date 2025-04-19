# QPO Detection and Generation Project

## Overview
A neural network based approach to detecting quasi-periodic oscillations from black holes

## Project Structure
### Main Notebooks
- `qpo_detector.ipynb`: Main notebook for QPO detection algorithms
- `cgan_phy.ipynb`: Conditional GAN implementation for physical data
- `cgan_phy_generated_dataset.ipynb`: Generation of datasets using CGAN
- `cgan_best.ipynb`: Best performing CGAN implementation
- `sbi.ipynb`: Simulation-Based Inference implementation
- `amp_experiment.ipynb`: Amplitude experiment analysis
- `param.ipynb`: Parameter analysis and optimization
- `plot_hyperparam_runs.ipynb`: Visualization of hyperparameter optimization results
- `diagrams.ipynb`: Project diagrams and visualizations
- `dataset_generator_am_signal.ipynb`: AM signal dataset generation
- `real_data_from_supervisor.ipynb`: Processing of real astronomical data
- `gan.ipynb`: Base GAN implementation
- `conditional_gan_exp.ipynb`: Conditional GAN experiments

### Directories
- `gan_outputs/`: Directory containing generated samples and model outputs
- `qpo_physical_dataset/`: Physical dataset for training and testing
- `utils/`: Utility functions and helper modules
- `saved_models/`: Trained model checkpoints
- `papers/`: Related research papers and documentation
- `backups/`: Backup files
- `bin/`: Binary files and executables
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
The project is organized as Jupyter notebooks:
1. Start with `qpo_detector.ipynb` for QPO detection
2. Use `cgan_phy.ipynb` for generating synthetic data
3. Explore `sbi.ipynb` for parameter inference
4. Use `plot_hyperparam_runs.ipynb` for analyzing optimization results
5. Check `diagrams.ipynb` for project visualizations

## Data
- The project uses astronomical time series data
- Physical dataset is stored in `qpo_physical_dataset/`
- Generated samples are saved in `gan_outputs/`
- Real data processing is documented in `real_data_from_supervisor.ipynb`

## License
See the LICENSE file for details.

## Contact
[Your contact information here]