# Continuous-Time Diffusion Models for Discrete Data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/paulffm/Master-Thesis/blob/master/LICENSE)

This repository serves as an unofficial **PyTorch** reimplementation of the paper [A Continuous Time Framework for Discrete Denoising Models](https://arxiv.org/pdf/2205.14987) (tauLDR) by A. Campbell et al. and first **PyTorch** implementation of the paper [Score-Based Continuous-Time Discrete Diffusion Models](https://openreview.net/pdf?id=BYWWwSY2G5s) (SDDM) by H. Sun et al. As part of my master's thesis, it includes extensions such as the implementation of new **sampling methods** and alternative **loss functions** to enhance the performance and flexibility of both frameworks.

<p align="center">
  <img src="ct_forwardrev_process2.png"  alt="1" width = 438px height = 311px >
</p>

## Overview

This implementation contains various enhancements and extensions aimed at improving sample quality and flexibility. Below are the key areas where enhancements have been made, along with specific additions:

### Sampling Methods

- **Euler Sampling**: Now available in tauLDR and SDDM.
- **PC-Scheme**: Euler sampling can now be utilized with additional corrector steps.
- **Tau-Leaping**: Now available in tauLDR and SDDM.
- **Midpoint Tau-leaping**: A new additional sampler introduced for enhanced performance.

### Loss Functions

- **Log-likelihood loss**: $L_{\text{ll}}(\theta) = T E_{t\sim \mathcal{U}(0, T) \pi_{\text{data}}(x_0) q_{t|0}(x|x_0)}[ - \log p^{\theta}_{0|t}(x_0|x) ]$
led to increased sample quality.
- **CT-ELBO loss**: The Continuous-time ELBO loss is now compatible with SDDM.
- **Combination of Log-likelihood loss and Categorical Ratio Matching loss**: $L_{\text{CRMll}}(\theta) = L_{\text{CRM}} + \lambda L_{\text{ll}}$ with $\lambda$ between 0.001 and 0.01 led to sample quality improvements.
  
### Networks

- I included many more neural networks as initially proposed, i.e. the score-network from the paper [Dirichlet Diffusion Score Model for Biological Sequence Generation](https://arxiv.org/pdf/2305.10699.pdf)

## Installation

Follow these steps to clone the repository and install the dependencies:

### 1. Clone the repository

Clone the repository using the following command:

```sh
git clone https://github.com/paulffm/Continuous-Time-Diffusion-Models-for-Discrete-Data.git
cd Continuous-Time-Diffusion-Models-for-Discrete-Data
```

### 2. Create a virtual environment

Create a virtual environment to install dependencies in isolation:

```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install dependencies

Install the necessary dependencies using pip:

```sh
pip install -r requirements.txt
```

## Usage

This implementation provides example scripts for training SDDM and tauLDR models to generate MNIST/maze/synthetic data. In this script you can simply use my provided configs and start training or retraining your models. You just need to set the correct paths in the beginning of the script, i.e.:

```python
def main():

    script_dir = os.path.dirname(os.path.realpath(__file__))
    save_location = os.path.join(script_dir, f"SavedModels/MNIST/")
    save_location_png = os.path.join(save_location, "PNGs/")
    dataset_location = os.path.join(script_dir, 'lib/datasets')

    train_resume = False
    print(save_location)
    if not train_resume:
        cfg = get_config()
        bookkeeping.save_config(cfg, save_location)

    else:
        model_name = "model_name.pt"
        date = "2024-05-10"
        config_name = "config_001.yaml"
        config_path = os.path.join(save_location, date, config_name)
        cfg = bookkeeping.load_config(config_path)
```

In addition, you need to set a location in the config files where you want to save you trained models:

```python
save_directory = "SavedModels/MNIST/"
```

## Note

More information about the maze dataset and sample quality metrics can be found here.

## Reference

```bibtex
@article{campbell2022continuous,
  title={A continuous time framework for discrete denoising models},
  author={Campbell, Andrew and Benton, Joe and De Bortoli, Valentin and Rainforth, Thomas and Deligiannidis, George and Doucet, Arnaud},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={28266--28279},
  year={2022}
}
@inproceedings{sun2022score,
  title={Score-based Continuous-time Discrete Diffusion Models},
  author={Sun, Haoran and Yu, Lijun and Dai, Bo and Schuurmans, Dale and Dai, Hanjun},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2022}
}
```
