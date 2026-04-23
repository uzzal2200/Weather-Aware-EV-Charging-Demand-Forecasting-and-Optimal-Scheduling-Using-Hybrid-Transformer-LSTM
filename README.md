# Weather-Aware EV Charging Demand Forecasting and Optimal Scheduling Using Hybrid Transformer–LSTM

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11.2-blue">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red">
  <img src="https://img.shields.io/badge/Pyomo-MILP-green">
  <img src="https://img.shields.io/badge/Status-Reproducible-brightgreen">
</p>

This repository contains the reproducibility code for weather-aware EV charging demand forecasting and optimal charging schedule generation using a Hybrid Transformer-LSTM model and MILP-based optimization.

## Overview

The pipeline performs the following steps in one run:

1. Load and preprocess EV + weather data.
2. Train forecasting models (Transformer-LSTM, LSTM, Random Forest, Linear Regression).
3. Evaluate forecasting performance with six metrics.
4. Run MILP-based charging optimization.
5. Generate all figures and tables used in the study.

## Data Sources

- EV charging data:
  https://catalog.data.gov/dataset/electric-vehicle-ev-charging-data-municipal-lots-and-garages
- Weather data (ASOS):
  https://mesonet.agron.iastate.edu/request/download.phtml?network=NY_ASOS

## Folder Structure

```
.
├── data/
│   └── EV.csv
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── milp_optimizer.py
│   ├── figures.py
│   ├── tables.py
│   ├── pipeline.py
│   └── main.py
├── main.py
├── requirements.txt
└── README.md
```

## Setup

### 1. Create and activate environment

```bash
conda create -n ev python=3.11.2 -y
conda activate ev
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install MILP solver (CBC)

```bash
conda install -c conda-forge coincbc -y
```

### 4. Place dataset

Put `EV.csv` inside the `data/` folder.

## Run

From the project root, run:

```bash
python main.py
```

This command runs everything end-to-end: data loading, training, evaluation, optimization, figures, and tables.

## Output

Generated files are saved in `figures_output/`:

- Figures: `Fig1_Framework.png` to `Fig10_Weather.png`
- Tables: `Table1_Features.csv`, `Table2_Performance.csv`, `Table3_Optimization.csv`

## Configuration

Adjust settings from `src/config.py`:

- Data split and sequence window (`LOOKBACK`, `TRAIN_RATIO`, `VAL_RATIO`)
- Training (`EPOCHS`, `BATCH_SIZE`, `LR`, `PATIENCE`)
- Model architecture (`D_MODEL`, `N_HEADS`, `D_FF`, `N_LAYERS`, `LSTM_HIDDEN`)
- Optimization (`N_EVS`, `P_MAX_KW`, `CHARGE_RATE_KW`, MILP weights)

## Troubleshooting

- If CBC is unavailable, optimization falls back to a heuristic schedule.
- If you face import issues, run from the project root and use `python main.py`.
- GPU is used automatically when CUDA is available; otherwise CPU is used.

## Author

MD Uzzal Mia and Sajib Debnath
