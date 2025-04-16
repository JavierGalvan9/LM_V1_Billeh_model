# LM_V1-model: Visual Cortical Areas Model

![model_figure](https://github.com/user-attachments/assets/e26d4c85-526d-48b4-903e-60cf2b99518b)

## Overview

This repository contains a computational model of the mouse primary visual cortex (V1) coupled with the lateromedial visual area (LM), built using TensorFlow. The model incorporates lateral geniculate nucleus (LGN) input to process visual stimuli and implements biologically-plausible top-down modulation from LM to V1.

The model is based on the spiking neural network architecture from the Billeh et al. (2020) model, extended to include interactions between V1 and LM areas, with biologically realistic connectivity patterns derived from experimental data.

## Key Features

- **Biologically realistic neural architecture**: based on experimental data about mouse visual cortex
- **Multi-area processing**: models interactions between V1 and LM visual areas
- **Spiking neural network**: implements leaky integrate-and-fire (LIF) neurons with realistic dynamics
- **Visual stimulus processing**: processes various visual stimuli through the LGN
- **Orientation and direction selectivity**: captures key properties of visual cortex neurons
- **Spontaneous activity**: models spontaneous firing rates based on experimental data
- **GPU-accelerated**: designed for efficient training and simulation on GPU hardware

## Requirements

- Python 3.8+
- TensorFlow 2.15+ (GPU version highly recommended)
- NVIDIA CUDA 11.8+ and cuDNN 8.8.0+ (for GPU acceleration)
- Other dependencies as specified in `requirements.txt`

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/LM_V1_Billeh_model.git
   cd LM_V1_Billeh_model
   ```

2. Create and activate a conda environment (recommended):
   ```
   conda create -n tf215 python=3.11
   conda activate tf215
   ```

3. Install the required packages using the requirements.txt file:
   ```
   pip install -r requirements.txt
   ```

   The requirements file includes all necessary dependencies such as:
   - TensorFlow and scientific computing libraries (NumPy, SciPy, Pandas)
   - Visualization tools (Matplotlib, Seaborn)
   - Brain modeling toolkit (BMTK)
   - Performance profiling tools
   - Jupyter notebook support

   Note: Ensure you have compatible CUDA drivers installed for GPU acceleration.

## Model Structure

The model consists of several key components:

- **LGN Input Layer**: Processes visual stimuli and provides input to V1
- **V1 Column**: Primary visual cortex model with realistic layer-specific connectivity
- **LM Column**: Higher-order visual area with feedback connections to V1
- **Inter-area Connectivity**: Realistic connections between V1 and LM

## Running Experiments

The repository includes scripts for running various types of experiments:

### Training the Model

```
python multi_training_single_gpu_split.py --v1_neurons 10000 --lm_neurons 1500 --n_epochs 20 --steps_per_epoch 10 --learning_rate 0.001 --rate_cost 100 --voltage_cost 1 --sync_cost 1 --osi_cost 1
```

Key parameters:
- `--v1_neurons`: Number of neurons in the V1 area
- `--lm_neurons`: Number of neurons in the LM area
- `--n_input`: Number of LGN input filters (default: 17400)
- `--n_epochs`: Number of training epochs
- `--steps_per_epoch`: Training steps per epoch
- `--rate_cost`: Weight for firing rate regularization
- `--voltage_cost`: Weight for voltage regularization
- `--osi_cost`: Weight for orientation selectivity index loss
- `--train_recurrent_v1`: Enable training of V1 recurrent connections
- `--train_recurrent_lm`: Enable training of LM recurrent connections
- `--train_interarea_v1_lm`: Enable training of V1-to-LM connections
- `--train_interarea_lm_v1`: Enable training of LM-to-V1 connections

### Orientation and Direction Selectivity Analysis

```
python osi_dsi_estimator.py --v1_neurons 10000 --lm_neurons 1500 --ckpt_dir path/to/model/checkpoint
```

### Receptive Field Analysis

```
python receptive_field_estimator.py --v1_neurons 10000 --lm_neurons 1500 --ckpt_dir path/to/model/checkpoint
```

### Classification Task

```
python classification_training_testing.py --v1_neurons 10000 --lm_neurons 1500 --n_epochs 20 --steps_per_epoch 50
```

## Analysis Tools

After running experiments, you can analyze the results using the provided Jupyter notebooks:

- `receptive_field_analysis.ipynb`: Analyze receptive fields of model neurons
- `osi_dsi_analysis.ipynb`: Analyze orientation and direction selectivity
- `classification_analysis.ipynb`: Analyze classification performance

## Model Parameters

The model has numerous configurable parameters:

- **Architectural Parameters**: Number of neurons, connectivity patterns, etc.
- **Training Parameters**: Learning rates, regularization weights, etc.
- **Stimulus Parameters**: Types of visual stimuli, stimulus duration, etc.
- **Analysis Parameters**: Parameters for analyzing model outputs

See the help text of individual scripts for more details on available parameters:

```
python multi_training_single_gpu_split.py --help
```

## Data Files

The model requires several data files that define the network architecture and initial weights:

- Network connectivity data (in `GLIF_network` directory)
- LGN model parameters (in `lgn_model` directory)
- Neuropixels data for biological validation (in `Neuropixels_data` directory)

## Example Workflow

Here's an example workflow for using this model:

1. Train a baseline model with default parameters:
   ```
   python multi_training_single_gpu_split.py --v1_neurons 10000 --lm_neurons 1500
   ```

2. Analyze the receptive fields of neurons in the trained model:
   ```
   python receptive_field_estimator.py --ckpt_dir Simulation_results/v1_10000_lm_1500/b_12xyz
   ```

3. Visualize and analyze the results using the Jupyter notebooks.

<!-- ## Citation

If you use this model in your research, please cite:

```
@article{your_paper,
  title={Your Paper Title},
  author={Your Name et al.},
  journal={Journal Name},
  year={2023}
}
``` -->

## References

- Billeh, Y. N., Cai, B., Gratiy, S. L., Dai, K., Iyer, R., Gouwens, N. W., ... & Arkhipov, A. (2020). Systematic integration of structural and functional data into multi-scale models of mouse primary visual cortex. Neuron, 106(3), 388-403.
- [Additional references relevant to your model]

## License

<!-- [Add your license information here]

## Contact

[Add your contact information here] -->
