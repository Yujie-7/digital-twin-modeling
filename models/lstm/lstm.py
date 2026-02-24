#!/usr/bin/env python3
"""
Steam Flow Digital Twin (LSTM + Online Periodic Fine-Tuning)

This script:
1) Loads time-series data from a CSV file
2) Runs an LSTM-based steam flow digital twin for one-step-ahead prediction
3) Optionally performs periodic online fine-tuning using a sliding update window
4) Saves per-sample predictions (CSV) and summary metrics (CSV)

Notes:
- update_period = 0 means "no online update" (pure inference).
- Online update uses the most recent `update_window` samples and trains `update_epochs`.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================
# Environment knobs (optional)
# ============================
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")


# ============================
# Logging
# ============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ============================
# Metrics
# ============================
def compute_mae_rmse_mape(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    eps: float = 1e-12,
) -> Tuple[float, float, float]:
    """
    Compute MAE, RMSE, MAPE. MAPE uses eps to avoid division by zero.
    """
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)

    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

    denom = np.maximum(np.abs(yt), eps)
    mape = float(np.mean(np.abs((yt - yp) / denom)) * 100.0)
    return mae, rmse, mape


# ============================
# Config
# ============================
@dataclass(frozen=True)
class ExperimentConfig:
    # Inputs
    model_path: str = "models/lstm/lstm"
    data_csv_path: str = "data/test/filename"

    # Output
    pred_dir: str = "results/predictions/lstm"
    metrics_out_csv: str = "results/evaluation metrics/lstm_steady-state_error.csv"

    # Period scan
    period_min: int = 0
    period_max: int = 50  # inclusive

    # Update settings (fixed in your original script)
    update_window: int = 50
    update_epochs: int = 3
    lr: float = 5e-3

    # Device
    device: str = "cpu"

    # Data columns (must match what the model expects)
    col_time_ms: str = "birth_mills"        # t
    col_setpoint: str = "lop_stm_fl_sv"     # r
    col_target: str = "lop_stm_fl_inv"      # y


# ============================
# Model
# ============================
class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ============================
# LSTM Digital Twin
# ============================
class ContinuousDigitalTwinLSTM:
    """
    LSTM-based digital twin with optional periodic online fine-tuning.
    """

    def __init__(
        self,
        model_path: str,
        update_window: int = 20,
        update_epochs: int = 3,
        lr: float = 5e-4,
        update_period: int = 20,   # 0 = no online updates
        device: str = "cpu",
    ) -> None:
        self.device = device

        # -------- Load checkpoint --------
        ckpt = torch.load(model_path, map_location=device, weights_only=False)

        self.scaler = ckpt["scaler_1"]
        self.inputs: List[str] = list(ckpt["inputs_1"])
        self.outputs: List[str] = list(ckpt["outputs_1"])
        self.time_steps: int = int(ckpt["time_steps"])

        if self.scaler.n_features_in_ != (len(self.inputs) + len(self.outputs)):
            raise ValueError("Scaler feature dimension mismatch with inputs+outputs.")

        # -------- Build model --------
        self.model = LSTMModel(
            input_dim=len(self.inputs),
            hidden_dim=64,
            output_dim=len(self.outputs),
        ).to(device)

        self.model.load_state_dict(ckpt["stage1_model_state"])
        self.model.eval()

        # -------- Online update config --------
        self.update_window = int(update_window)
        self.update_epochs = int(update_epochs)
        self.lr = float(lr)
        self.update_period = int(update_period)

        self.criterion = nn.MSELoss()
        self.reset_optimizer()

    def reset_optimizer(self) -> None:
        """
        Reset optimizer for fair comparisons across different update_period runs.
        """
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-5,
        )

    def online_update(self, x_win: torch.Tensor, y_win: torch.Tensor) -> None:
        """
        Online fine-tuning on a window of recent samples.
        """
        self.model.train()
        for _ in range(self.update_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.criterion(self.model(x_win), y_win)
            loss.backward()
            self.optimizer.step()
        self.model.eval()

    def run_once(
        self,
        df: pd.DataFrame,
        col_time_ms: str,
        col_setpoint: str,
        col_target: str,
        save_pred_path: Optional[str] = None,
    ) -> Dict[str, np.ndarray | float]:
        """
        Run one pass on the given dataframe, return predictions and static metrics.
        """
        # ---- Basic checks ----
        if col_setpoint not in df.columns:
            raise ValueError(f"CSV missing column: {col_setpoint}")
        if col_target not in df.columns:
            raise ValueError(f"CSV missing column: {col_target}")
        for c in (self.inputs + self.outputs):
            if c not in df.columns:
                raise ValueError(f"CSV missing model-required column: {c}")

        # ---- Build scaled sequences ----
        df_full = df[self.inputs + self.outputs].copy()
        data_scaled = self.scaler.transform(df_full.to_numpy(dtype=float))

        X_list, Y_list = [], []
        for i in range(len(data_scaled) - self.time_steps):
            X_list.append(data_scaled[i : i + self.time_steps, : len(self.inputs)])
            Y_list.append(data_scaled[i + self.time_steps, len(self.inputs) :])

        if len(X_list) == 0:
            raise ValueError("Not enough samples to build sequences for the given time_steps.")

        X = torch.tensor(np.asarray(X_list), dtype=torch.float32, device=self.device)
        Y = torch.tensor(np.asarray(Y_list), dtype=torch.float32, device=self.device)

        # ---- Prepare true/pred in raw space ----
        idx_y = self.outputs.index(col_target)

        # Ground truth: directly from raw dataframe (most reliable)
        y_true_all = df[col_target].to_numpy(dtype=float)
        y_true = y_true_all[self.time_steps : self.time_steps + len(X)]

        r_all = df[col_setpoint].to_numpy(dtype=float)
        r_series = r_all[self.time_steps : self.time_steps + len(X)]

        # Time axis
        if col_time_ms in df.columns:
            t_all = df[col_time_ms].to_numpy(dtype=float) / 1000.0
            t_series = t_all[self.time_steps : self.time_steps + len(X)]
        

        y_pred = np.zeros(len(X), dtype=float)

        # ---- Inference + periodic update ----
        do_update = (self.update_period is not None) and (self.update_period > 0)

        for k in range(len(X)):
            with torch.no_grad():
                y_pred_scaled = self.model(X[k : k + 1]).detach().cpu().numpy()  # (1, out_dim)

            # inverse_transform needs full feature vector: [inputs, outputs]
            x_last = X[k : k + 1, -1, :].detach().cpu().numpy()  # (1, in_dim)
            full_scaled = np.concatenate([x_last, y_pred_scaled], axis=1)  # (1, in+out)

            y_pred_raw_all = self.scaler.inverse_transform(full_scaled)[:, len(self.inputs) :]  # (1, out_dim)
            y_pred[k] = float(y_pred_raw_all[0, idx_y])

            # Periodic online update
            if (
                do_update
                and self.update_window > 0
                and k >= self.update_window
                and (k % self.update_period == 0)
            ):
                x_win = X[k - self.update_window : k]
                y_win = Y[k - self.update_window : k]
                self.online_update(x_win, y_win)

        mae, rmse, mape = compute_mae_rmse_mape(y_true, y_pred)

        if save_pred_path is not None:
            os.makedirs(os.path.dirname(save_pred_path) or ".", exist_ok=True)
            pd.DataFrame(
                {
                    "t": t_series,
                    "r": r_series,
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            ).to_csv(save_pred_path, index=False, encoding="utf-8-sig")

        return {
            "t": t_series,
            "r": r_series,
            "true": y_true,
            "pred": y_pred,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
        }


# ============================
# Experiment runner
# ============================
def load_dataframe(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def run_experiment(cfg: ExperimentConfig) -> pd.DataFrame:
    df = load_dataframe(cfg.data_csv_path)
    os.makedirs(cfg.pred_dir, exist_ok=True)

    rows: List[Dict[str, float]] = []

    for upd_period in range(cfg.period_min, cfg.period_max + 1):
        logger.info("Running update_period=%d", upd_period)

        twin = ContinuousDigitalTwinLSTM(
            model_path=cfg.model_path,
            update_window=cfg.update_window,
            update_epochs=cfg.update_epochs,
            lr=cfg.lr,
            update_period=upd_period,
            device=cfg.device,
        )
        twin.reset_optimizer()  # ensure fairness

        save_pred_path = os.path.join(cfg.pred_dir, f"lstm_pred_period_{upd_period:02d}.csv")
        out = twin.run_once(
            df=df,
            col_time_ms=cfg.col_time_ms,
            col_setpoint=cfg.col_setpoint,
            col_target=cfg.col_target,
            save_pred_path=save_pred_path,
        )

        rows.append(
            {
                "update_period": float(upd_period),
                "update_window": float(cfg.update_window),
                "update_epochs": float(cfg.update_epochs),
                "lr": float(cfg.lr),
                "MAE": float(out["mae"]),
                "RMSE": float(out["rmse"]),
                "MAPE": float(out["mape"]),
            }
        )

        logger.info(
            "[update_period=%02d] MAE=%.3f | RMSE=%.3f | MAPE=%.3f",
            upd_period,
            out["mae"],
            out["rmse"],
            out["mape"],
        )

    out_df = pd.DataFrame(rows).sort_values("update_period")
    out_df.to_csv(cfg.metrics_out_csv, index=False, encoding="utf-8-sig")
    logger.info("Saved steady-state metrics: %s", cfg.metrics_out_csv)
    return out_df


def main() -> None:
    cfg = ExperimentConfig(
        # If you want to test only period=0 quickly:
        # period_min=0, period_max=0,
        # data_csv_path="../test_data/11.csv",
    )

    df_metrics = run_experiment(cfg)
    logger.info("Done. Head of metrics:\n%s", df_metrics.head())


if __name__ == "__main__":
    main()