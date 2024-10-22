# xLSTM-Mixer

This is the supplemental code repository for the paper **"xLSTM-Mixer: Multivariate Time Series Forecasting by Mixing via Scalar Memories"**.

## Requirements

The code was tested on an Ubuntu 22.04 machine with CUDA 12.1. The (Python) requirements are defined in `requirements.txt` and `lighting_requirements.txt`. Both can be installed using pip:

```bash
pip install -r requirements.txt
pip install -r lighting_requirements.txt
```

The code expects the benchmark datasets to be located in the (`/`) `/common-ts` directory on a Linux-based system or container. This package is based on PyTorch 2.4 and was tested with Python 3.11.

For the CUDA version of sLSTM, you will need a Compute Capability of 8.0 or higher. Check https://developer.nvidia.com/cuda-gpus for more details (e.g., A100 GPUs are supported).

## Running the Project
The project provides a Lightning-based CLI, which is also used for the experiments in the `scripts` folder. One can get general help for the utilization by running:

```bash
 python -m xlstm_forecasting --help
```

To see specific model or data argument configurations, use:

```bash
 python -m xlstm_forecasting fit --help
```

## Experiments
Predefined scripts with hyperparameters are available in the `scripts` folder.

To quickly check if all requirements are met, one may pass the `--dev` flag to any script, e.g.,

```bash
bash ./scripts/long_term_forecasting/ett/m1.sh --dev
```
This will run a test batch to verify the setup. Full seed runs are organized by dataset in the `scripts` folder.

### Example Commands
Run the following scripts for specific experiments:
```bash
bash ./scripts/long_term_forecasting/weather.sh
```

```bash
bash ./scripts/long_term_forecasting/ett/m1.sh
```
