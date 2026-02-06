# StockandCrypto

End-to-end BTC forecasting MVP aligned with `Project_plan.md`:
- Direction probability (`P(up) / P(down)`)
- Start window prediction (`W0/W1/W2/W3`, where `W0 = no_start`)
- Magnitude interval prediction (`q10 / q50 / q90`)
- Purged walk-forward evaluation (`gap = max_horizon`)
- Streamlit dashboard for hourly/daily outputs

## Project Structure

```text
CryptoForecast/
├── configs/config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── features/
│   ├── labels/
│   ├── models/
│   ├── evaluation/
│   └── utils/
└── dashboard/app.py
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline Commands

```bash
python -m src.ingestion.update_data --config configs/config.yaml
python -m src.features.build_features --config configs/config.yaml
python -m src.labels.build_labels --config configs/config.yaml
python -m src.models.train --config configs/config.yaml
python -m src.models.predict --config configs/config.yaml
python -m src.evaluation.walk_forward --config configs/config.yaml
python -m src.evaluation.backtest --config configs/config.yaml
python -m streamlit run dashboard/app.py
```

Or run all in one shot:

```bash
powershell -ExecutionPolicy Bypass -File scripts/run_pipeline.ps1
```

## Notes

- Data is stored in UTC (`timestamp_utc`) and displayed with market timezone (`timestamp_market`).
- If Binance API fetch fails, ingestion automatically falls back to synthetic data so the pipeline remains runnable.
- Model artifacts are versioned in `data/models/<timestamp>_<branch>/`.
- Reproducibility: `seed=42`, config snapshot saved with each model version.
