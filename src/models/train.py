from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src.models.factory import (
    build_logistic_binary,
    build_logistic_multiclass,
    build_mvp_classifier,
    build_quantile_regressor,
)
from src.utils.common import set_global_seed, utc_now_timestamp
from src.utils.config import load_config, save_yaml
from src.utils.io import ensure_dir, save_json, save_model, write_csv
from src.utils.metrics import (
    classification_metrics,
    interval_metrics,
    multiclass_metrics,
    pinball_loss,
    regression_metrics,
    rows_from_metric_dict,
)


def _load_branch_dataset(processed_dir: Path, branch_name: str) -> pd.DataFrame:
    feat_path = processed_dir / f"features_{branch_name}.csv"
    label_path = processed_dir / f"labels_{branch_name}.csv"
    if not feat_path.exists() or not label_path.exists():
        raise FileNotFoundError(
            f"Missing features/labels for branch={branch_name}. Run feature and label pipelines first."
        )
    feats = pd.read_csv(feat_path)
    labels = pd.read_csv(label_path)
    df = feats.merge(labels, on="timestamp_utc", how="inner")
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    return df


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    ignore_prefix = ("y_",)
    ignore_cols = {"timestamp_utc", "timestamp_market", "market_tz"}
    candidates = []
    for c in df.columns:
        if c in ignore_cols:
            continue
        if c.startswith(ignore_prefix):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            candidates.append(c)
    return candidates


def _time_split(df: pd.DataFrame, train_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if n < 20:
        raise ValueError("Dataset too small. Need at least 20 rows.")
    split = int(n * train_ratio)
    split = max(10, min(split, n - 5))
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def _prepare_xy(
    train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_cols: List[str], target_col: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    tr = train_df.dropna(subset=[target_col]).copy()
    va = valid_df.dropna(subset=[target_col]).copy()
    X_train = tr[feature_cols].copy()
    y_train = tr[target_col].copy()
    X_valid = va[feature_cols].copy()
    y_valid = va[target_col].copy()
    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med)
    X_valid = X_valid.fillna(med)
    return X_train, y_train, X_valid, y_valid, med


def run_train(config_path: str) -> None:
    cfg = load_config(config_path)
    set_global_seed(int(cfg.get("project", {}).get("seed", 42)))

    paths = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    models_cfg = cfg.get("models", {})
    processed_dir = Path(paths.get("processed_data_dir", "data/processed"))
    models_root = ensure_dir(paths.get("models_dir", "data/models"))

    train_ratio = float(models_cfg.get("train_ratio", 0.8))
    quantiles: List[float] = [float(x) for x in models_cfg.get("quantiles", [0.1, 0.5, 0.9])]
    calibrate_cfg = models_cfg.get("calibration", {})
    do_calibrate = bool(calibrate_cfg.get("enabled", True))
    cal_method = str(calibrate_cfg.get("method", "sigmoid"))
    cal_cv = int(calibrate_cfg.get("cv", 3))
    seed = int(cfg.get("project", {}).get("seed", 42))

    symbol = data_cfg.get("symbol", "BTCUSDT").lower()
    branches = data_cfg.get("branches", {})
    ts = utc_now_timestamp()

    latest_versions: Dict[str, str] = {}
    overall_rows: List[Dict] = []

    for branch_name, branch_cfg in branches.items():
        df = _load_branch_dataset(processed_dir, branch_name)
        feature_cols = _select_feature_columns(df)
        train_df, valid_df = _time_split(df, train_ratio=train_ratio)

        version_dir = ensure_dir(models_root / f"{ts}_{branch_name}")
        latest_versions[branch_name] = str(version_dir)
        medians_to_save: Dict[str, float] = {}
        branch_rows: List[Dict] = []

        # 1) Start-window classification
        target_start = "y_start_window"
        X_train, y_train, X_valid, y_valid, med = _prepare_xy(
            train_df, valid_df, feature_cols, target_start
        )
        y_train_cls = y_train.astype(int)
        y_valid_cls = y_valid.astype(int)
        medians_to_save.update({k: float(v) for k, v in med.to_dict().items()})

        # Baseline: most frequent class in train
        most_freq = int(y_train_cls.value_counts().index[0])
        pred_base = np.full(len(y_valid_cls), most_freq, dtype=int)
        n_class = max(2, int(y_train_cls.max()) + 1)
        proba_base = np.zeros((len(y_valid_cls), n_class))
        for i, p in enumerate(pred_base):
            p = int(min(max(p, 0), n_class - 1))
            proba_base[i, p] = 1.0
        m_base = multiclass_metrics(y_valid_cls.to_numpy(), pred_base, proba_base)
        branch_rows.extend(
            rows_from_metric_dict(
                m_base,
                {
                    "branch": branch_name,
                    "task": "start_window",
                    "model": "baseline_most_freq",
                    "horizon": "max",
                    "split": "holdout",
                },
            )
        )

        start_model, backend = build_mvp_classifier(cfg, n_classes=n_class, seed=seed)
        start_model.fit(X_train, y_train_cls)
        pred_mvp = start_model.predict(X_valid)
        proba_mvp = start_model.predict_proba(X_valid)
        m_mvp = multiclass_metrics(y_valid_cls.to_numpy(), pred_mvp, proba_mvp)
        branch_rows.extend(
            rows_from_metric_dict(
                m_mvp,
                {
                    "branch": branch_name,
                    "task": "start_window",
                    "model": f"mvp_{backend}",
                    "horizon": "max",
                    "split": "holdout",
                },
            )
        )
        save_model(start_model, version_dir / "start_window_mvp.pkl")

        # 2) Direction binary + 3) Quantile regressors per horizon
        horizons: List[int] = [int(h) for h in branch_cfg.get("horizons", [1, 2, 4])]
        for h in horizons:
            # Direction
            target_dir = f"y_dir_h{h}"
            if target_dir in df.columns:
                X_train, y_train, X_valid, y_valid, med = _prepare_xy(
                    train_df, valid_df, feature_cols, target_dir
                )
                y_train_bin = y_train.astype(int)
                y_valid_bin = y_valid.astype(int)
                medians_to_save.update({k: float(v) for k, v in med.to_dict().items()})

                # Baseline: previous bar direction from return_1.
                prev_ret = valid_df.loc[y_valid.index, "return_1"].fillna(0.0)
                pred_base = (prev_ret > 0).astype(int).to_numpy()
                proba_base = pred_base.astype(float)
                m_base = classification_metrics(
                    y_valid_bin.to_numpy(),
                    pred_base,
                    proba_base,
                )
                branch_rows.extend(
                    rows_from_metric_dict(
                        m_base,
                        {
                            "branch": branch_name,
                            "task": "direction",
                            "model": "baseline_prev_bar",
                            "horizon": str(h),
                            "split": "holdout",
                        },
                    )
                )

                # Logistic baseline
                logistic = build_logistic_binary(seed=seed)
                logistic.fit(X_train, y_train_bin)
                pred_lr = logistic.predict(X_valid)
                proba_lr = logistic.predict_proba(X_valid)[:, 1]
                m_lr = classification_metrics(y_valid_bin.to_numpy(), pred_lr, proba_lr)
                branch_rows.extend(
                    rows_from_metric_dict(
                        m_lr,
                        {
                            "branch": branch_name,
                            "task": "direction",
                            "model": "baseline_logistic",
                            "horizon": str(h),
                            "split": "holdout",
                        },
                    )
                )
                save_model(logistic, version_dir / f"direction_h{h}_baseline_logistic.pkl")

                mvp_dir, backend = build_mvp_classifier(cfg, n_classes=2, seed=seed)
                mvp_dir.fit(X_train, y_train_bin)

                if do_calibrate:
                    calibrated = CalibratedClassifierCV(
                        estimator=mvp_dir,
                        method=cal_method,
                        cv=cal_cv,
                    )
                    calibrated.fit(X_train, y_train_bin)
                    dir_model = calibrated
                    model_name = f"mvp_{backend}_calibrated_{cal_method}"
                else:
                    dir_model = mvp_dir
                    model_name = f"mvp_{backend}"

                pred_mvp = dir_model.predict(X_valid)
                proba_mvp = dir_model.predict_proba(X_valid)[:, 1]
                m_mvp = classification_metrics(y_valid_bin.to_numpy(), pred_mvp, proba_mvp)
                branch_rows.extend(
                    rows_from_metric_dict(
                        m_mvp,
                        {
                            "branch": branch_name,
                            "task": "direction",
                            "model": model_name,
                            "horizon": str(h),
                            "split": "holdout",
                        },
                    )
                )
                save_model(dir_model, version_dir / f"direction_h{h}_mvp.pkl")

            # Quantile regressors
            target_ret = f"y_ret_h{h}"
            if target_ret in df.columns:
                X_train, y_train, X_valid, y_valid, med = _prepare_xy(
                    train_df, valid_df, feature_cols, target_ret
                )
                y_train_reg = y_train.astype(float)
                y_valid_reg = y_valid.astype(float)
                medians_to_save.update({k: float(v) for k, v in med.to_dict().items()})

                # Baseline return=0
                pred_zero = np.zeros(len(y_valid_reg), dtype=float)
                m_zero = regression_metrics(y_valid_reg.to_numpy(), pred_zero)
                branch_rows.extend(
                    rows_from_metric_dict(
                        m_zero,
                        {
                            "branch": branch_name,
                            "task": "magnitude",
                            "model": "baseline_zero",
                            "horizon": str(h),
                            "split": "holdout",
                        },
                    )
                )

                q_preds: Dict[float, np.ndarray] = {}
                quant_backend = ""
                for q in quantiles:
                    reg, quant_backend = build_quantile_regressor(cfg, quantile=q, seed=seed)
                    reg.fit(X_train, y_train_reg)
                    yq = reg.predict(X_valid)
                    q_preds[q] = yq
                    save_model(reg, version_dir / f"ret_h{h}_q{q:.1f}_mvp.pkl")
                    branch_rows.append(
                        {
                            "branch": branch_name,
                            "task": "magnitude_quantile",
                            "model": f"mvp_{quant_backend}",
                            "horizon": str(h),
                            "split": "holdout",
                            "metric": f"pinball_q{q:.1f}",
                            "value": pinball_loss(y_valid_reg.to_numpy(), yq, q),
                        }
                    )

                if 0.5 in q_preds:
                    reg_m = regression_metrics(y_valid_reg.to_numpy(), q_preds[0.5])
                    branch_rows.extend(
                        rows_from_metric_dict(
                            reg_m,
                            {
                                "branch": branch_name,
                                "task": "magnitude",
                                "model": f"mvp_{quant_backend}_q50",
                                "horizon": str(h),
                                "split": "holdout",
                            },
                        )
                    )

                if 0.1 in q_preds and 0.9 in q_preds:
                    im = interval_metrics(y_valid_reg.to_numpy(), q_preds[0.1], q_preds[0.9])
                    branch_rows.extend(
                        rows_from_metric_dict(
                            im,
                            {
                                "branch": branch_name,
                                "task": "interval",
                                "model": f"mvp_{quant_backend}",
                                "horizon": str(h),
                                "split": "holdout",
                            },
                        )
                    )

        # Persist artifacts for branch
        save_json(
            {
                "symbol": symbol,
                "branch": branch_name,
                "feature_columns": feature_cols,
                "feature_medians": medians_to_save,
                "quantiles": quantiles,
                "seed": seed,
            },
            version_dir / "artifact_meta.json",
        )
        save_yaml(cfg, version_dir / "config_snapshot.yaml")
        branch_metrics = pd.DataFrame(branch_rows)
        write_csv(branch_metrics, version_dir / "metrics.csv")
        print(f"[OK] Saved model artifacts -> {version_dir}")
        overall_rows.extend(branch_rows)

    # Save global pointers
    save_json(latest_versions, models_root / "latest_versions.json")
    if overall_rows:
        write_csv(pd.DataFrame(overall_rows), processed_dir / "metrics_train_holdout.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline + MVP models.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_train(args.config)


if __name__ == "__main__":
    main()
