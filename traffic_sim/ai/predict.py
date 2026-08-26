"""Inference: estimate each mode's outcome from telemetry alone.

Two XGBoost regressors — one per delivery mode — predict the achievable
delivery rate a group would get. XGBoost is the only ML model in this
project; there is no second model to compare against.

Why regression and not a UNICAST/BROADCAST classifier: a classifier
would bury the reasoning inside the model. Predicting *outcomes* keeps
the three questions separate and inspectable — what will happen (here),
which outcome we prefer (`ai.utility`), and what we are allowed to do
about it (`ai.agent`).

If the model files are missing, or xgboost is not installed, the
predictor falls back to computing outcomes analytically via
`ai.outcomes`. The simulator therefore never breaks because of a missing
artifact; it just stops being predictive and becomes an oracle. Which
path was used is reported in `PredictionResult.source` so results are
never silently mislabelled as model-driven.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from config.config import SimConfig
from traffic_sim.ai.feature_engineering import GroupFeatures
from traffic_sim.ai.outcomes import ModeOutcome, OutcomePair, evaluate_modes

UNICAST_MODEL_FILE = "xgb_unicast_rate.json"
BROADCAST_MODEL_FILE = "xgb_broadcast_rate.json"


@dataclass
class PredictionResult:
    outcomes: OutcomePair
    source: str  # "xgboost" or "analytical"
    inference_latency_ms: float


class OutcomePredictor:
    """Loads the trained models once and predicts per group."""

    def __init__(self, config: SimConfig, model_dir: str | Path | None = None):
        self._config = config
        self._model_dir = Path(model_dir if model_dir is not None else config.ai_model_path)
        self._unicast_model = None
        self._broadcast_model = None
        self._load_models()

    @property
    def uses_model(self) -> bool:
        return self._unicast_model is not None and self._broadcast_model is not None

    def _load_models(self) -> None:
        unicast_path = self._model_dir / UNICAST_MODEL_FILE
        broadcast_path = self._model_dir / BROADCAST_MODEL_FILE
        if not (unicast_path.exists() and broadcast_path.exists()):
            return
        try:
            import xgboost as xgb
        except ImportError:
            return
        try:
            unicast = xgb.XGBRegressor()
            unicast.load_model(str(unicast_path))
            broadcast = xgb.XGBRegressor()
            broadcast.load_model(str(broadcast_path))
        except Exception:
            # A corrupt or incompatible artifact must not take the
            # simulator down; fall back to the analytical path.
            return
        self._unicast_model = unicast
        self._broadcast_model = broadcast

    def predict(self, features: GroupFeatures, dt: float | None = None) -> PredictionResult:
        started = time.perf_counter()

        if not self.uses_model:
            outcomes = evaluate_modes(features, self._config, dt)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return PredictionResult(
                outcomes=outcomes, source="analytical", inference_latency_ms=elapsed_ms
            )

        import numpy as np

        row = np.array([features.vector()], dtype=float)
        unicast_rate = float(self._unicast_model.predict(row)[0])
        broadcast_rate = float(self._broadcast_model.predict(row)[0])

        # A predicted rate is still bounded by physics: never negative,
        # never more than the stream's own bitrate.
        bitrate = features.segment_bitrate_mbps
        unicast_rate = max(0.0, min(unicast_rate, bitrate))
        broadcast_rate = max(0.0, min(broadcast_rate, bitrate))

        outcomes = OutcomePair(
            unicast=self._build_outcome(unicast_rate, features, dt),
            broadcast=self._build_outcome(broadcast_rate, features, dt),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return PredictionResult(outcomes=outcomes, source="xgboost", inference_latency_ms=elapsed_ms)

    def _build_outcome(self, rate_mbps: float, features: GroupFeatures, dt: float | None) -> ModeOutcome:
        from traffic_sim.network.throughput import delivered_seconds_of_content

        cfg = self._config
        step = dt if dt is not None else cfg.simulation_timestep_s
        delivered = delivered_seconds_of_content(rate_mbps, features.segment_bitrate_mbps, step)
        consumed = cfg.playback_consumption_rate * step
        projected = max(0.0, features.buffer_min_s + delivered - consumed)
        return ModeOutcome(
            achievable_mbps=rate_mbps,
            delivered_seconds=delivered,
            projected_min_buffer_s=projected,
            would_stall=projected <= 0.0,
        )
