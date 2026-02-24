"""
Steam Flow Digital Twin (Physics + PI + Periodic Calibration)

This script:
1) Loads time-series data from a CSV file
2) Runs a physics-based steam flow digital twin with PI valve control
3) Optionally performs periodic parameter calibration (Cv_max) using a sliding window
4) Saves per-sample predictions (CSV) and summary metrics (CSV)

"""

from __future__ import annotations

import os
import math
import logging
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from iapws import IAPWS97


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
    Compute MAE, RMSE, and MAPE.

    Notes:
      - MAPE can explode when y_true contains zeros; we use eps to avoid division by zero.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

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
    # Input
    data_csv_path: str = "data/test/filename"

    # Output
    pred_dir: str = "results/predictions/physics-based"
    metrics_out_csv: str = "results/evaluation metrics/physics_steady-state_error.csv"

    # Period scan: inclusive range [period_min, period_max]
    period_min: int = 0
    period_max: int = 50

    # Twin parameters
    Cv_max_init: float = 90.0
    R: float = 50.0
    tau_v: float = 70.0
    Kp_open: float = 0.1
    I_rep_per_min: float = 5.0

    # Calibration parameters
    calib_window: int = 50
    min_samples: int = 50
    learning_rate: float = 0.01

    # Data columns
    col_time_ms: str = "birth_mills"
    col_setpoint: str = "lop_stm_fl_sv"
    col_flow: str = "lop_stm_fl_inv"
    col_valve: str = "lop_stm_vop_inv"
    col_p1: str = "upp_hetg_pr_inv"
    col_p2: str = "lop_hetg_pr_inv"


# ============================
# Digital Twin Model
# ============================
class SteamFlowDigitalTwin:
    """
    Physics-based steam flow digital twin with PI valve control and periodic calibration on Cv_max.

    Core states:
      - xv_sim: simulated valve opening (%)
      - Q_pred: predicted flow
      - I_state: PI integrator state

    Calibration:
      - Maintains a sliding buffer of Cv observations (cv_obs_buffer)
      - Every `update_period` steps, update Cv_max toward median(buffer)
    """

    def __init__(
        self,
        Cv_max_init: float = 90.0,
        R: float = 50.0,
        tau_v: float = 70.0,
        Kp_open: float = 0.1,
        I_rep_per_min: float = 5.0,
        Ts: float = 6.0,
        update_period: int = 50,
        calib_window: int = 60,
        min_samples: int = 30,
        learning_rate: float = 0.01,
    ) -> None:
        # Physics / valve params
        self.Cv_max: float = float(Cv_max_init)
        self.R: float = float(R)
        self.tau_v: float = float(tau_v)
        self.Ts: float = float(Ts)

        # PI controller params
        self.Kp: float = float(Kp_open)
        self.Ti: float = 60.0 / float(I_rep_per_min)

        # Calibration config
        self.update_period: int = int(update_period)  # <=0 means disable
        self.step_counter: int = 0
        self.calib_window: int = int(calib_window)
        self.min_samples: int = int(min_samples)
        self.learning_rate: float = float(learning_rate)

        self.cv_obs_buffer: Deque[float] = deque(maxlen=self.calib_window)

        # Internal states
        self.xv_sim: float = 0.0
        self.Q_pred: float = 0.0
        self.I_state: float = 0.0

        # For output/analysis
        self.cv_history: List[float] = []

    @staticmethod
    def _get_rho_sat(P_abs_MPa: float) -> float:
        """Saturated vapor density (kg/m^3) at absolute pressure P (MPa)."""
        try:
            return float(IAPWS97(P=P_abs_MPa, x=1).rho)
        except Exception:
            # Fallback to a reasonable constant if IAPWS fails
            return 1.25

    def align_state(self, r_init: float, Q_act_init: float, xv_act_init: float) -> None:
        """
        Initialize internal states to align with the first sample.
        """
        self.xv_sim = float(xv_act_init)
        self.Q_pred = float(Q_act_init)

        error_init = float(r_init) - float(Q_act_init)
        self.I_state = float(xv_act_init) - self.Kp * error_init

        self.cv_history = [self.Cv_max]

    def step_predict(self, r_k: float, p1_g: float, p2_g: float) -> Tuple[float, float]:
        """
        One-step simulation:
          1) PI control to generate valve command
          2) First-order valve dynamics
          3) Physics-based flow calculation
        """
        # PI
        e = float(r_k) - self.Q_pred
        P_term = self.Kp * e

        delta_I = self.Kp * (self.Ts / self.Ti) * e
        temp_I_state = self.I_state + delta_I
        raw_x_cmd = P_term + temp_I_state

        # Anti-windup: only integrate when not saturated
        if 0.0 < raw_x_cmd < 100.0:
            self.I_state = temp_I_state

        x_cmd = float(np.clip(raw_x_cmd, 0.0, 100.0))

        # Valve first-order lag
        alpha = self.Ts / max(self.tau_v, 1e-6)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        self.xv_sim = self.xv_sim + alpha * (x_cmd - self.xv_sim)

        # Physics flow prediction
        self.Q_pred = self.flow_physics(p1_g, p2_g, self.xv_sim)
        return self.Q_pred, self.xv_sim

    def flow_physics(self, p1_g: float, p2_g: float, opening: float) -> float:
        """
        ISA-like control valve mass flow approximation.

        Inputs are gauge pressures in MPa. Internally converted to absolute MPa.
        """
        P_atm = 1.013  
        p1 = float(p1_g) + P_atm
        p2 = float(p2_g) + P_atm

        Fy, xT = 0.90, 0.72

        if p1 <= p2:
            return 0.0

        theta = float(np.clip(opening / 100.0, 0.0, 1.0))
        char_factor = self.R ** (theta - 1.0)

        rho = self._get_rho_sat(p2)
        x = (p1 - p2) / p1

        Cv_now = self.Cv_max * char_factor

        if x < Fy * xT:
            term = 1.0 - x / (3.0 * Fy * xT)
            Q = 2.73 * Cv_now * term * math.sqrt((p1 - p2) * 1000.0 * rho)
        else:
            Q = 0.66 * 2.73 * Cv_now * math.sqrt(Fy * xT * p1 * 1000.0 * rho)

        return float(Q)

    def update_parameter_periodic(
        self,
        Q_act: float,
        p1_g: float,
        p2_g: float,
        xv_act: float,
    ) -> None:
        """
        Periodic calibration for Cv_max using a sliding historical window.

        - Accumulate Cv observations when conditions are valid
        - Every `update_period` steps, if >= min_samples, update:
            Cv_max <- Cv_max + lr * (median(buffer) - Cv_max)
        """
        self.cv_history.append(self.Cv_max)

        if self.update_period <= 0:
            return

        self.step_counter += 1

        # Validity gate (keep your original gate)
        if float(xv_act) > 5.0:
            P_atm = 1.013
            p1 = float(p1_g) + P_atm
            p2 = float(p2_g) + P_atm

            if p1 > p2:
                rho = self._get_rho_sat(p2)

                theta = float(np.clip(float(xv_act) / 100.0, 0.0, 1.0))
                char_factor = self.R ** (theta - 1.0)
                x = (p1 - p2) / p1

                Fy, xT = 0.90, 0.72
                if x < Fy * xT:
                    pf = (
                        2.73
                        * char_factor
                        * (1.0 - x / (3.0 * Fy * xT))
                        * math.sqrt((p1 - p2) * 1000.0 * rho)
                    )
                else:
                    pf = 0.66 * 2.73 * char_factor * math.sqrt(Fy * xT * p1 * 1000.0 * rho)

                if pf > 0.1 and np.isfinite(pf):
                    cv_obs = float(Q_act) / float(pf)
                    if np.isfinite(cv_obs) and (0.0 < cv_obs < 1e6):
                        self.cv_obs_buffer.append(cv_obs)

        # Periodic trigger
        if (self.step_counter % self.update_period) == 0:
            if len(self.cv_obs_buffer) >= self.min_samples:
                med_cv = float(np.median(np.asarray(self.cv_obs_buffer)))
                self.Cv_max = self.Cv_max + self.learning_rate * (med_cv - self.Cv_max)


# ============================
# I/O Helpers
# ============================
def load_data(cfg: ExperimentConfig) -> Dict[str, np.ndarray]:
    df = pd.read_csv(cfg.data_csv_path)

    # Time in seconds
    t = df[cfg.col_time_ms].to_numpy(dtype=float) / 1000.0

    data = {
        "t": t,
        "r": df[cfg.col_setpoint].to_numpy(dtype=float),
        "Q": df[cfg.col_flow].to_numpy(dtype=float),
        "xv": df[cfg.col_valve].to_numpy(dtype=float),
        "p1": df[cfg.col_p1].to_numpy(dtype=float),
        "p2": df[cfg.col_p2].to_numpy(dtype=float),
    }
    return data


def save_pointwise_predictions(
    out_dir: str,
    update_period: int,
    t: np.ndarray,
    r: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    y_true: np.ndarray,
    xv_act: np.ndarray,
    y_pred: np.ndarray,  # length N-1 aligned to Q_true[1:]
    xv_sim: np.ndarray,         # length N-1
    Cv_hist: np.ndarray,        # length N-1
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    df_out = pd.DataFrame(
        {
            "t": t[1:],
            "r": r[1:],
            "p1": p1[1:],
            "p2": p2[1:],
            "y_true": y_true[1:],
            "y_pred": y_pred,
            "xv_act": xv_act[1:],
            "xv_sim": xv_sim,
            "Cv_max": Cv_hist,
            "update_period": int(update_period),
        }
    )

    path = os.path.join(out_dir, f"physics_pred_period_{update_period:02d}.csv")
    df_out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ============================
# Experiment Runner
# ============================
def run_single_period(
    cfg: ExperimentConfig,
    update_period: int,
    data: Dict[str, np.ndarray],
) -> Dict[str, float]:
    t = data["t"]
    r = data["r"]
    y_true = data["Q"]
    xv_true = data["xv"]
    p1 = data["p1"]
    p2 = data["p2"]

    Ts = float(np.median(np.diff(t)))
    N = len(t)

    twin = SteamFlowDigitalTwin(
        Cv_max_init=cfg.Cv_max_init,
        R=cfg.R,
        tau_v=cfg.tau_v,
        Kp_open=cfg.Kp_open,
        I_rep_per_min=cfg.I_rep_per_min,
        Ts=Ts,
        update_period=update_period,
        calib_window=cfg.calib_window,
        min_samples=cfg.min_samples,
        learning_rate=cfg.learning_rate,
    )
    twin.align_state(r[0], y_true[0], xv_true[0])

    # Store per-step results: [Q_pred, xv_sim, Cv_max]
    results = np.zeros((N, 3), dtype=float)

    for k in range(N):
        q_p, xv_p = twin.step_predict(r[k], p1[k], p2[k])
        if update_period > 0:
            twin.update_parameter_periodic(y_true[k], p1[k], p2[k], xv_true[k])

        results[k, 0] = q_p
        results[k, 1] = xv_p
        results[k, 2] = twin.Cv_max

    # Align 1-step prediction: pred[k] corresponds to Q_actual[k+1]
    y_pred = results[:-1, 0]
    xv_sim = results[:-1, 1]
    Cv_hist = results[:-1, 2]

    mae, rmse, mape = compute_mae_rmse_mape(y_true[1:], y_pred)

    # Save pointwise file
    pred_path = save_pointwise_predictions(
        out_dir=cfg.pred_dir,
        update_period=update_period,
        t=t,
        r=r,
        p1=p1,
        p2=p2,
        y_true=y_true,
        xv_act=xv_true,
        y_pred=y_pred,
        xv_sim=xv_sim,
        Cv_hist=Cv_hist,
    )
    logger.info("Saved pointwise pred: %s", pred_path)

    return {
        "update_period": float(update_period),
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Cv_end": float(twin.Cv_max),
    }


def run_experiment(cfg: ExperimentConfig) -> pd.DataFrame:
    data = load_data(cfg)

    rows: List[Dict[str, float]] = []
    for period in range(cfg.period_min, cfg.period_max + 1):
        logger.info("Running update_period=%d", period)
        row = run_single_period(cfg, period, data)
        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values("update_period")
    out_df.to_csv(cfg.metrics_out_csv, index=False, encoding="utf-8-sig")
    logger.info("Saved metrics summary: %s", cfg.metrics_out_csv)

    return out_df


def main() -> None:
    cfg = ExperimentConfig(
        # You can override defaults here if needed
        # data_csv_path="../test_data/20250310.csv",
        # period_min=0,
        # period_max=50,
    )
    df_metrics = run_experiment(cfg)
    logger.info("Done. Head of metrics:\n%s", df_metrics.head())


if __name__ == "__main__":
    main()