param(
  [string]$Config = "configs/config.yaml"
)

$ErrorActionPreference = "Stop"

python -m src.ingestion.update_data --config $Config
python -m src.features.build_features --config $Config
python -m src.labels.build_labels --config $Config
python -m src.models.train --config $Config
python -m src.models.predict --config $Config
python -m src.evaluation.walk_forward --config $Config
python -m src.evaluation.backtest --config $Config

Write-Host "Pipeline complete." -ForegroundColor Green
