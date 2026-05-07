# Battery Degradation Modeling with LSTM: Rollout Stability and Uncertainty Analysis

Predicting lithium-ion battery **State of Health (SOH)** and **Remaining Useful Life (RUL)** using LSTM-based sequence modeling, with a focus on long-horizon prediction stability and uncertainty quantification.

## Motivation

Accurate SOH/RUL estimation is critical for predictive maintenance, battery safety, and lifecycle optimization in large-scale Battery Management Systems (BMS). Long-horizon prediction is inherently challenging: autoregressive models suffer from **compounding error accumulation**, where small per-step errors grow over extended rollouts.

This project investigates that degradation dynamics directly — building on findings from [rnn-lstm-sequence-prediction](https://github.com/UNICODEY/rnn-lstm-sequence-prediction) and applying them to a real industrial scenario.

---

## Results

| Model | MAE | RMSE |
|-------|-----|------|
| Persistence | 0.0036 | 0.0056 |
| Linear Regression | 0.0030 | 0.0056 |
| MLP | 0.0873 | 0.0882 |
| **LSTM (ours)** | **0.0079** | **0.0086** |

**RUL prediction error: 6 cycles out of 161 total (3.7%)**

> Note: Simple baselines outperform LSTM at short horizons. LSTM's advantage emerges at extended prediction horizons — see Rollout Analysis below.

---

## Key Findings

### 1. Rollout Stability: LSTM vs Persistence

| Horizon | LSTM MAE | Persistence MAE |
|---------|----------|-----------------|
| 1 cycle | 0.0068 | 0.0056 |
| 10 cycles | 0.0075 | 0.0135 |
| 30 cycles | 0.0080 | 0.0291 |

At short horizons, naive baselines are competitive. But as prediction horizon grows, persistence error increases linearly while **LSTM error stays flat** — a 3.6× advantage at horizon 30. This directly validates the compounding error hypothesis from autoregressive sequence modeling.

### 2. Multivariate Features: A Meaningful Negative Result

Adding voltage, current, and temperature features (6-dimensional input) increased MAE from **0.0079 → 0.0274**. With only 110 training samples, statistical feature aggregation (mean/min per cycle) loses critical temporal detail. **More features ≠ better model** — feature engineering quality matters more than feature quantity.

### 3. Uncertainty Estimation

Monte Carlo Dropout provides per-prediction confidence intervals. Uncertainty is higher in early cycles (where degradation is less predictable) and decreases as the battery approaches EOL (where the degradation curve becomes more regular). This is consistent with real BMS requirements: knowing *how confident* a prediction is matters as much as the prediction itself.

---

## Method

**Dataset:** NASA Li-ion Battery Aging Dataset (B0005)
- 168 discharge cycles, capacity fade from 2 Ah → 1.4 Ah (EOL at 30% fade)

**Pipeline:**
1. Extract per-cycle discharge capacity → compute SOH
2. Build sliding window dataset (window = 30 cycles)
3. Train 2-layer LSTM (hidden=64, dropout=0.2)
4. Evaluate against persistence, linear regression, and MLP baselines
5. Rollout horizon analysis: measure MAE at horizons 1–30
6. Multivariate experiment: add voltage/current/temperature features
7. Uncertainty estimation via Monte Carlo Dropout (100 forward passes)

---

## Visualizations

| | |
|---|---|
| Capacity Degradation | SOH Prediction vs Ground Truth |
| Rollout Error Accumulation | Uncertainty Band (MC Dropout) |

---

## Usage

1. Download the [NASA Battery Dataset](https://data.nasa.gov/dataset/Li-ion-Battery-Aging-Datasets/uj5r-zjdb)
2. Upload `5.+Battery+Data+Set.zip` to Google Drive
3. Open `battery_soh_prediction.ipynb` in Google Colab
4. Run all cells sequentially

**Requirements:**
```
torch
scipy
numpy
matplotlib
scikit-learn
```

---

## Related Work

This project is part of a broader investigation into long-horizon time series stability:

- [RNN/LSTM Sequence Prediction Stability](https://github.com/UNICODEY/rnn-lstm-sequence-prediction) — compounding error analysis, teacher forcing vs free rollout, latent space prediction (RSSM-inspired architecture)
