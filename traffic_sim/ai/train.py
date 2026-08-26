"""Train the XGBoost outcome models.

Two regressors, one per delivery mode, both predicting the achievable
delivery rate a group would get. XGBoost is the only model used in this
project.

Run:  python -m traffic_sim.ai.train
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config.config import SimConfig
from traffic_sim.ai.dataset import Dataset, generate_dataset
from traffic_sim.ai.feature_engineering import FEATURE_NAMES
from traffic_sim.ai.predict import BROADCAST_MODEL_FILE, UNICAST_MODEL_FILE

DEFAULT_DATASET_PATH = "data/training/outcomes.csv"


def train_models(
    dataset: Dataset,
    model_dir: str | Path,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Fit both regressors, report accuracy, and save them."""
    import numpy as np
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split

    x, y_unicast, y_broadcast = dataset.xy()
    x_arr = np.array(x, dtype=float)

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"n_rows": len(dataset), "features": FEATURE_NAMES, "modes": {}}

    for label, y_values, filename in (
        ("unicast", y_unicast, UNICAST_MODEL_FILE),
        ("broadcast", y_broadcast, BROADCAST_MODEL_FILE),
    ):
        y_arr = np.array(y_values, dtype=float)
        x_train, x_test, y_train, y_test = train_test_split(
            x_arr, y_arr, test_size=test_fraction, random_state=random_state
        )

        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
        )
        model.fit(x_train, y_train)

        predictions = model.predict(x_test)
        importances = dict(
            sorted(
                zip(FEATURE_NAMES, (float(v) for v in model.feature_importances_)),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        report["modes"][label] = {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "r2": float(r2_score(y_test, predictions)),
            "top_features": dict(list(importances.items())[:5]),
        }

        model.save_model(str(model_dir / filename))

    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the XGBoost outcome models.")
    parser.add_argument("--ticks", type=int, default=40, help="Ticks per scenario/seed run.")
    parser.add_argument("--seeds", type=int, default=6, help="How many seeds to sweep.")
    parser.add_argument("--model-dir", type=str, default=None, help="Defaults to config.ai_model_path.")
    parser.add_argument("--dataset-out", type=str, default=DEFAULT_DATASET_PATH)
    args = parser.parse_args(argv)

    config = SimConfig.default()
    model_dir = args.model_dir or config.ai_model_path

    print(f"Generating dataset ({args.seeds} seeds x 2 scenarios x {args.ticks} ticks)...")
    dataset = generate_dataset(config, seeds=tuple(range(1, args.seeds + 1)), ticks=args.ticks)
    dataset.to_csv(args.dataset_out)
    print(f"  {len(dataset)} rows -> {args.dataset_out}")

    print("Training XGBoost outcome models...")
    report = train_models(dataset, model_dir)

    for mode, stats in report["modes"].items():
        print(f"  {mode:9s} MAE={stats['mae']:.4f} Mbps   R2={stats['r2']:.4f}")
        top = ", ".join(f"{name}={value:.3f}" for name, value in stats["top_features"].items())
        print(f"            top features: {top}")
    print(f"Models saved to {model_dir}/")


if __name__ == "__main__":
    main()
