# Predictive Maintenance with Machine Learning on the NASA C-MAPSS Dataset

Individual Independent Study, Spring 2026.

This repository contains the full implementation, results, and write-up
for an independent study on machine-learning-based predictive maintenance
of industrial assets, using the publicly released NASA C-MAPSS turbofan
engine degradation dataset (Saxena & Goebel, 2008) as a stand-in for the
proprietary industrial data originally targeted.

## Headline result

| Subset | n train engines | Best model | Test RMSE (cycles) |
|--------|----------------:|:----------:|-------------------:|
| FD001  | 100             | LSTM       | **15.27** |
| FD002  | 260             | LSTM       | **14.74** |
| FD003  | 100             | LightGBM   | **14.47** |
| FD004  | 249             | LSTM       | **15.50** |

All four numbers are competitive with published benchmarks on the same
splits (Zheng et al. 2017; Li et al. 2018; Ramasso & Saxena 2014).

## Layout

```
project/
├── data/CMAPSSData/            raw NASA files (downloaded via git clone)
├── src/
│   ├── data_loader.py          load CMAPSS, compute RUL, PHM08 score
│   ├── features.py             regime detection, rolling stats, sequencing
│   ├── eda.py                  EDA + the 8 dataset figures
│   ├── baselines.py            Mean / Ridge / RF / XGBoost / LightGBM
│   ├── lstm_model.py           PyTorch LSTM
│   ├── result_figures.py       result-section figures
│   ├── api.py                  Flask REST API for serving predictions
│   ├── dashboard.py            Streamlit dashboard (optional)
│   └── dashboard_mockup.py     static mock-up of the dashboard
├── models/                     saved LightGBM / XGBoost / LSTM artefacts
├── results/                    CSV result tables + LSTM predictions/history
├── figures/                    all PDF figures used in the paper
└── paper/                      LaTeX source for the final write-up
```

## Reproducing the results

```bash
# Data: ~45 MB, downloaded via a public mirror of the NASA archive
mkdir -p data && cd data
git clone --depth=1 https://github.com/mapr-demos/predictive-maintenance.git
mkdir CMAPSSData && mv predictive-maintenance/notebooks/jupyter/Dataset/CMAPSSData/*.txt CMAPSSData/

# Exploration + figures (≈ 30 s)
python src/eda.py

# Tabular baselines on all four subsets (≈ 4 min CPU)
python src/baselines.py

# Deep model: LSTM on each subset (≈ 3-5 min per subset, CPU)
python src/lstm_model.py

# Result figures + dashboard mock-up
python src/result_figures.py
python src/dashboard_mockup.py

# Deployment API (Flask)
python src/api.py --subset FD001 --port 5000
```
