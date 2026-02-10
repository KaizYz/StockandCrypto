from __future__ import annotations

import os
from typing import Any, Dict, Tuple

from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _try_lightgbm():
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        return LGBMClassifier, LGBMRegressor
    except Exception:
        return None, None


def _try_xgboost():
    try:
        from xgboost import XGBClassifier, XGBRegressor

        return XGBClassifier, XGBRegressor
    except Exception:
        return None, None


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}


def _gpu_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    models_cfg = cfg.get("models", {})
    gpu_cfg = models_cfg.get("gpu", {}) if isinstance(models_cfg, dict) else {}
    enabled = _env_flag("FORECAST_USE_GPU", bool(gpu_cfg.get("enabled", False)))
    fallback = _env_flag(
        "FORECAST_GPU_FALLBACK_CPU",
        bool(gpu_cfg.get("fallback_to_cpu", True)),
    )
    return {
        "enabled": enabled,
        "fallback_to_cpu": fallback,
        "lightgbm_device_type": str(gpu_cfg.get("lightgbm_device_type", "gpu")),
        "xgboost_device": str(gpu_cfg.get("xgboost_device", "cuda")),
        "xgboost_tree_method": str(gpu_cfg.get("xgboost_tree_method", "hist")),
    }


def build_logistic_binary(seed: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    random_state=seed,
                    class_weight="balanced",
                    max_iter=2000,
                ),
            ),
        ]
    )


def build_logistic_multiclass(seed: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    random_state=seed,
                    class_weight="balanced",
                    max_iter=3000,
                    multi_class="auto",
                ),
            ),
        ]
    )


def build_mvp_classifier(
    cfg: Dict[str, Any],
    *,
    n_classes: int,
    seed: int,
) -> Tuple[Any, str]:
    backend = cfg.get("models", {}).get("mvp_backend", "lightgbm").lower()
    model_cfg = cfg.get("models", {}).get("classifier", {})
    gpu_cfg = _gpu_settings(cfg)

    LGBMClassifier, _ = _try_lightgbm()
    if backend == "lightgbm" and LGBMClassifier is not None:
        params = {
            "n_estimators": model_cfg.get("n_estimators", 500),
            "learning_rate": model_cfg.get("learning_rate", 0.03),
            "num_leaves": model_cfg.get("num_leaves", 63),
            "max_depth": model_cfg.get("max_depth", -1),
            "subsample": model_cfg.get("subsample", 0.8),
            "colsample_bytree": model_cfg.get("colsample_bytree", 0.8),
            "reg_alpha": model_cfg.get("reg_alpha", 0.0),
            "reg_lambda": model_cfg.get("reg_lambda", 0.0),
            "random_state": seed,
            "verbosity": -1,
        }
        if n_classes > 2:
            params["objective"] = "multiclass"
            params["num_class"] = n_classes
            params["class_weight"] = "balanced"
        else:
            params["objective"] = "binary"
            params["class_weight"] = "balanced"
        if gpu_cfg["enabled"]:
            params["device_type"] = gpu_cfg["lightgbm_device_type"]
        return LGBMClassifier(**params), "lightgbm"

    XGBClassifier, _ = _try_xgboost()
    if backend == "xgboost" and XGBClassifier is not None:
        obj = "multi:softprob" if n_classes > 2 else "binary:logistic"
        xgb_max_depth = int(model_cfg.get("max_depth", 6))
        if xgb_max_depth <= 0:
            xgb_max_depth = 6
        params = {
            "objective": obj,
            "n_estimators": model_cfg.get("n_estimators", 500),
            "learning_rate": model_cfg.get("learning_rate", 0.03),
            "max_depth": xgb_max_depth,
            "subsample": model_cfg.get("subsample", 0.8),
            "colsample_bytree": model_cfg.get("colsample_bytree", 0.8),
            "reg_alpha": model_cfg.get("reg_alpha", 0.0),
            "reg_lambda": model_cfg.get("reg_lambda", 1.0),
            "eval_metric": "mlogloss" if n_classes > 2 else "logloss",
            "random_state": seed,
        }
        if gpu_cfg["enabled"]:
            params["tree_method"] = gpu_cfg["xgboost_tree_method"]
            params["device"] = gpu_cfg["xgboost_device"]
        xgb = XGBClassifier(**params)
        return xgb, "xgboost"

    # Pure sklearn fallback
    fallback = HistGradientBoostingClassifier(
        max_depth=model_cfg.get("max_depth", 6),
        learning_rate=model_cfg.get("learning_rate", 0.05),
        max_iter=model_cfg.get("n_estimators", 300),
        random_state=seed,
    )
    return fallback, "sklearn_hgb"


def build_quantile_regressor(
    cfg: Dict[str, Any], *, quantile: float, seed: int
) -> Tuple[Any, str]:
    backend = cfg.get("models", {}).get("mvp_backend", "lightgbm").lower()
    model_cfg = cfg.get("models", {}).get("regressor", {})
    gpu_cfg = _gpu_settings(cfg)

    _, LGBMRegressor = _try_lightgbm()
    if backend == "lightgbm" and LGBMRegressor is not None:
        params = {
            "objective": "quantile",
            "alpha": quantile,
            "n_estimators": model_cfg.get("n_estimators", 500),
            "learning_rate": model_cfg.get("learning_rate", 0.03),
            "num_leaves": model_cfg.get("num_leaves", 63),
            "max_depth": model_cfg.get("max_depth", -1),
            "subsample": model_cfg.get("subsample", 0.8),
            "colsample_bytree": model_cfg.get("colsample_bytree", 0.8),
            "reg_alpha": model_cfg.get("reg_alpha", 0.0),
            "reg_lambda": model_cfg.get("reg_lambda", 0.0),
            "random_state": seed,
            "verbosity": -1,
        }
        if gpu_cfg["enabled"]:
            params["device_type"] = gpu_cfg["lightgbm_device_type"]
        reg = LGBMRegressor(**params)
        return reg, "lightgbm_quantile"

    _, XGBRegressor = _try_xgboost()
    if backend == "xgboost" and XGBRegressor is not None:
        xgb_max_depth = int(model_cfg.get("max_depth", 6))
        if xgb_max_depth <= 0:
            xgb_max_depth = 6
        params = {
            "objective": "reg:quantileerror",
            "quantile_alpha": float(quantile),
            "n_estimators": model_cfg.get("n_estimators", 500),
            "learning_rate": model_cfg.get("learning_rate", 0.03),
            "max_depth": xgb_max_depth,
            "subsample": model_cfg.get("subsample", 0.8),
            "colsample_bytree": model_cfg.get("colsample_bytree", 0.8),
            "reg_alpha": model_cfg.get("reg_alpha", 0.0),
            "reg_lambda": model_cfg.get("reg_lambda", 1.0),
            "random_state": seed,
        }
        if gpu_cfg["enabled"]:
            params["tree_method"] = gpu_cfg["xgboost_tree_method"]
            params["device"] = gpu_cfg["xgboost_device"]
        return XGBRegressor(**params), "xgboost_quantile"

    # sklearn fallback supports quantile loss directly.
    reg = GradientBoostingRegressor(
        loss="quantile",
        alpha=quantile,
        n_estimators=model_cfg.get("n_estimators", 300),
        learning_rate=model_cfg.get("learning_rate", 0.05),
        max_depth=model_cfg.get("max_depth", 3),
        random_state=seed,
    )
    return reg, "sklearn_gbr_quantile"
