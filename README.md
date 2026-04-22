# Task 2 — Multi-step Forecasting: Strategy vs Model

## Goal
Compare forecast drift for horizon **H = 100** using three inference strategies:

1. **One-step model** (predicts 1 step ahead), rolled out recursively with **stride = 1**.
2. **K-step model** (K = 20, predicts 20 steps ahead), rolled out with **stride = 20** (block rollout).
3. **K-step model** rolled out with **stride = 1**, using **only the first predicted step** each time.

You will:
- implement windowing for horizon=1 and horizon=K
- train both models
- implement the three rollout strategies
- compare drift visually and via MAE/RMSE on the horizon

## Structure
```
src/
  multistep_forecast.py
  multistep_forecast_solution.py   # mentor-only

tests/
  test_multistep_forecast.py
```

## How to run
```bash
pip install -r requirements.txt
python -m unittest -q
```

## Notes
- **No shuffle** in time series splits.
- The tests validate *logic* (windowing/splits/rollouts) and include a small training smoke test.
- Use `python src/multistep_forecast.py` to run the demo plot (after implementing functions).
