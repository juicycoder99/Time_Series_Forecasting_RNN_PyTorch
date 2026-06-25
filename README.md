# Time-Series Forecasting with RNNs (PyTorch)

Comparing vanilla RNN, LSTM, and GRU architectures for one-step-ahead temperature forecasting on a decade-long daily series.

Recurrent neural networks for one-step-ahead forecasting of the **Daily Minimum Temperatures**
series (Melbourne, 1981–1990). A vanilla RNN, an LSTM, and a GRU are implemented in PyTorch
(GPU-accelerated) and compared. The full implementation is in
[`rnn_timeseries_forecasting.ipynb`](rnn_timeseries_forecasting.ipynb).

## Workflow

1. Load and visualise the 3,650-point daily temperature series.
2. Chronological 80/20 split, MinMax scaling (fit on train), and 30-day sliding windows.
3. A configurable recurrent model (RNN / LSTM / GRU) with a linear output head.
4. Train each variant on the GPU and evaluate with RMSE and MAE in °C.
5. Plot training-loss curves, an RMSE comparison, and the best model's forecast vs the actual series.

## Results (RTX 3080)

| Model | Test RMSE (°C) | Test MAE (°C) |
|-------|---------------:|--------------:|
| Vanilla RNN | 2.234 | 1.759 |
| LSTM | 2.234 | 1.762 |
| GRU | 2.287 | 1.800 |

The three architectures perform almost identically because one-step-ahead prediction of a strongly
seasonal signal needs only short-range memory, so the LSTM/GRU gating advantage barely shows.

## Running it

```bash
pip install torch numpy pandas matplotlib scikit-learn   # CUDA build of torch for GPU
jupyter notebook rnn_timeseries_forecasting.ipynb
```

## Files

| File | Description |
|------|-------------|
| `rnn_timeseries_forecasting.ipynb` | Full implementation and analysis (RNN / LSTM / GRU comparison) |
| `daily-min-temperatures.csv` | Dataset (Melbourne daily minimum temperatures) |
| `PROJECT_BRIEF.pdf` | Project brief (goals, objectives, outcomes) |
