"""
Steam Flow Digital Twin (ARX  + RLS Calibration)

This script:
1) Loads time-series data from a CSV file
2) Runs an ARX-based digital twin using predicted output history
3) Applies observer-aligned correction using measured outputs
4) Performs optional online parameter calibration via gated recursive least squares (RLS)
5) Saves per-sample predictions (CSV) and summary metrics (CSV)
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


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
    pred_dir: str = "results/predictions/system-identification"
    metrics_out_csv: str = "results/evaluation metrics/system-identification_steady-state_error.csv"

    # Scan calib period: inclusive range [period_min, period_max]
    period_min: int = 0
    period_max: int = 50

    # Data columns
    col_time_ms: str = "birth_mills"
    col_setpoint: str = "lop_stm_fl_sv"     # r
    col_u2: str = "lop_hetg_pr_inv"         # u2
    col_y: str = "lop_stm_fl_inv"           # y_true

    # Warmup
    warmup_min: int = 10  


# ============================
# ARX Digital Twin
# ============================
class DigitalTwinARX:
    """
    Digital Twin ARX model:
    - predict_one_step(): free-run prediction using predicted y-history
    - correct_with_measurement(): observer-aligned output using measured y-history
    - rls_update(): gated recursive least squares update
    """

    def __init__(
        self,
        params: Dict,
        scalers: Dict[str, float],
        lambda_base: float = 0.995,
        P0: float = 1.0,
        step_th_phys: float = 100.0,
        hold_steps: int = 2,
        err_gate_z: float = 3.5,
        phi_norm_gate: float = 80.0,
        theta_clip: float = 8.0,
        z_clip: float = 12.0,
    ) -> None:
        self.na: int = int(params["na"])
        self.nb1: int = int(params["nb1"])
        self.nb2: int = int(params["nb2"])
        self.Lz: int = int(params["Lz"])

        self.beta_Z = np.asarray(params["beta_Z"], dtype=float).copy()
        self.s = dict(scalers)

        max_len = max(self.na, self.nb1, self.nb2, self.Lz) + 1
        self.hist_rz = np.zeros(max_len, dtype=float)
        self.hist_u2z = np.zeros(max_len, dtype=float)
        self.hist_u1hatz = np.zeros(max_len, dtype=float)
        self.hist_ymeasz = np.zeros(max_len, dtype=float)
        self.hist_ypredz = np.zeros(max_len, dtype=float)

        a = np.asarray(params["a"], dtype=float).copy()
        B1 = np.asarray(params["B1"], dtype=float).copy()
        B2 = np.asarray(params["B2"], dtype=float).copy()
        c0 = float(params["c0"])

        # theta = [-a, B1, B2, c0]
        self.theta = np.concatenate([-a, B1, B2, [c0]]).astype(float)

        self.P = np.eye(len(self.theta), dtype=float) * float(P0)
        self.lambda_base = float(lambda_base)

        # gating / safety
        self.step_th_phys = float(step_th_phys)
        self.hold_steps = int(hold_steps)
        self.err_gate_z = float(err_gate_z)
        self.phi_norm_gate = float(phi_norm_gate)
        self.theta_clip = float(theta_clip)
        self.z_clip = float(z_clip)

        self.freeze_cnt: int = 0
        self.prev_r_raw: Optional[float] = None

        self._last_phi: Optional[np.ndarray] = None
        self._last_yhat: Optional[float] = None

    @staticmethod
    def _shift_and_update(buf: np.ndarray, new_val: float) -> None:
        buf[1:] = buf[:-1]
        buf[0] = float(new_val)

    def _detect_step_and_freeze(self, r_raw: float) -> None:
        if self.prev_r_raw is None:
            self.prev_r_raw = float(r_raw)
            return

        if self.freeze_cnt == 0:
            if abs(float(r_raw) - self.prev_r_raw) >= self.step_th_phys:
                self.freeze_cnt = self.hold_steps

        self.prev_r_raw = float(r_raw)

    def _stable_check(self, theta: np.ndarray) -> bool:
        """
        Check AR stability by eigenvalues of companion form built from y-lags.
        """
        na = self.na
        if na <= 0:
            return True

        theta_y = theta[:na]
        if na == 1:
            return abs(theta_y[0]) < 1.0

        A = np.zeros((na, na), dtype=float)
        A[0, :] = theta_y
        A[1:, :-1] = np.eye(na - 1)
        eigs = np.linalg.eigvals(A)
        return float(np.max(np.abs(eigs))) < 1.0

    def _phi_from_hist(self, y_hist: np.ndarray, u1hatz_k: float, u2z_k: float) -> np.ndarray:
        """
        Build ARX regressor phi for one-step prediction in z-space.
        Layout matches your original implementation.
        """
        return np.concatenate(
            [
                y_hist[: self.na],
                [u1hatz_k],
                self.hist_u1hatz[: self.nb1 - 1],
                [u2z_k],
                self.hist_u2z[: self.nb2 - 1],
                [1.0],
            ]
        )

    def predict_one_step(self, r_raw: float, u2_raw: float) -> float:
        """
        Free-run one-step prediction:
        - build u1hat from Z features
        - build phi using predicted y-history
        - update histories (rz, u2z, u1hatz, ypredz)
        - return yhat in raw scale
        """
        rz_k = (float(r_raw) - self.s["r_mu"]) / self.s["r_sd"]
        u2z_k = (float(u2_raw) - self.s["u2_mu"]) / self.s["u2_sd"]

        z_feat = np.concatenate([[rz_k], self.hist_rz[: self.Lz], self.hist_u2z[: self.Lz]])
        u1hatz_k = float(np.dot(z_feat, self.beta_Z))

        phi_pred = self._phi_from_hist(self.hist_ypredz, u1hatz_k, u2z_k)
        ypredz_k1 = float(np.dot(phi_pred, self.theta))
        ypredz_k1 = float(np.clip(ypredz_k1, -self.z_clip, self.z_clip))

        self._shift_and_update(self.hist_rz, rz_k)
        self._shift_and_update(self.hist_u2z, u2z_k)
        self._shift_and_update(self.hist_u1hatz, u1hatz_k)
        self._shift_and_update(self.hist_ypredz, ypredz_k1)

        return ypredz_k1 * self.s["y_sd"] + self.s["y_mu"]

    def correct_with_measurement(self, r_raw: float, u2_raw: float, y_true_raw: float) -> float:
        """
        Observer-aligned correction step:
        - uses measured y-history for phi
        - stores last phi and yhat for potential RLS update
        - updates measured y-history buffer
        - returns yhat in raw scale (aligned output)
        """
        self._detect_step_and_freeze(r_raw)

        rz_k = (float(r_raw) - self.s["r_mu"]) / self.s["r_sd"]
        u2z_k = (float(u2_raw) - self.s["u2_mu"]) / self.s["u2_sd"]
        y_meas_z = (float(y_true_raw) - self.s["y_mu"]) / self.s["y_sd"]

        z_feat = np.concatenate([[rz_k], self.hist_rz[: self.Lz], self.hist_u2z[: self.Lz]])
        u1hatz_k = float(np.dot(z_feat, self.beta_Z))

        phi_meas = self._phi_from_hist(self.hist_ymeasz, u1hatz_k, u2z_k)
        yhat_meas = float(np.dot(phi_meas, self.theta))
        yhat_meas = float(np.clip(yhat_meas, -self.z_clip, self.z_clip))

        self._last_phi = phi_meas.copy()
        self._last_yhat = float(yhat_meas)

        self._shift_and_update(self.hist_ymeasz, y_meas_z)
        return yhat_meas * self.s["y_sd"] + self.s["y_mu"]

    def rls_update(self, y_true_raw: float) -> None:
        """
        Gated RLS update using the most recent phi_meas and yhat_meas from correction step.
        """
        if self._last_phi is None or self._last_yhat is None:
            return

        if self.freeze_cnt > 0:
            self.freeze_cnt -= 1
            return

        y_meas_z = (float(y_true_raw) - self.s["y_mu"]) / self.s["y_sd"]
        phi = self._last_phi.reshape(-1, 1)
        yhat = float(self._last_yhat)

        err = float(y_meas_z - yhat)

        if abs(err) > self.err_gate_z:
            return
        if float(np.linalg.norm(phi)) > self.phi_norm_gate:
            return

        theta_old = self.theta.copy()
        P_old = self.P.copy()

        lam = self.lambda_base
        den = lam + float((phi.T @ self.P @ phi)[0, 0])
        K = (self.P @ phi) / den

        theta_new = self.theta + (K.flatten() * err)
        theta_new = np.clip(theta_new, -self.theta_clip, self.theta_clip)

        P_new = (self.P - K @ phi.T @ self.P) / lam

        if not self._stable_check(theta_new):
            self.theta = theta_old
            self.P = P_old
            return

        self.theta = theta_new
        self.P = P_new

    def anchor_pred_chain_to_meas(self, y_true_raw: float) -> None:
        """
        During warmup, you anchored the predicted chain (hist_ypredz) to the measured y.
        Keep as an explicit helper for clarity.
        """
        y_meas_z = (float(y_true_raw) - self.s["y_mu"]) / self.s["y_sd"]
        self._shift_and_update(self.hist_ypredz, y_meas_z)

    def compute_warmup_length(self, warmup_min: int = 10) -> int:
        """
        Match your original warmup logic:
        warmN = max(na+2, Lz+2, warmup_min)
        """
        return int(max(self.na + 2, self.Lz + 2, warmup_min))


# ============================
# I/O Helpers
# ============================
def load_data(cfg: ExperimentConfig) -> Dict[str, np.ndarray]:
    df = pd.read_csv(cfg.data_csv_path)

    t = df[cfg.col_time_ms].to_numpy(dtype=float) / 1000.0
    r = df[cfg.col_setpoint].to_numpy(dtype=float)
    u2 = df[cfg.col_u2].to_numpy(dtype=float)
    y = df[cfg.col_y].to_numpy(dtype=float)

    return {"t": t, "r": r, "u2": u2, "y": y}


def save_pointwise_predictions(
    out_dir: str,
    calib_period: int,
    t: np.ndarray,
    r: np.ndarray,
    u2: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    warmN: int,
) -> Tuple[str, str]:
    """
    Save two CSVs:
      1) full: contains NaNs for non-valid prediction indices
      2) valid: filtered to valid prediction points only
    """
    os.makedirs(out_dir, exist_ok=True)

    df_out = pd.DataFrame(
        {
            "t": t,
            "r": r,
            "u2": u2,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    df_out["is_valid_pred"] = ~np.isnan(df_out["y_pred"])
    df_out["warmup_end_idx"] = int(warmN)

    full_path = os.path.join(out_dir, f"system-identification_pred_period_{calib_period:02d}.csv")

    df_out.to_csv(full_path, index=False, encoding="utf-8-sig")

    return full_path


# ============================
# Experiment Runner
# ============================
def run_single_period(
    cfg: ExperimentConfig,
    calib_period: int,
    model_params: Dict,
    scalers: Dict[str, float],
    data: Dict[str, np.ndarray],
) -> Dict[str, float]:
    t = data["t"]
    r_stream = data["r"]
    u2_stream = data["u2"]
    y_true_stream = data["y"]

    dt_engine = DigitalTwinARX(
        model_params,
        scalers,
        lambda_base=0.995,
        P0=1.0,
        step_th_phys=100.0,
        hold_steps=0,          # keep your original
        err_gate_z=3.5,
        phi_norm_gate=80.0,
        theta_clip=8.0,
        z_clip=12.0,
    )

    warmN = dt_engine.compute_warmup_length(cfg.warmup_min)
    warmN = int(min(warmN, len(r_stream) - 2))
    if warmN < 0:
        warmN = 0

    # -------- Warmup --------
    for k in range(warmN):
        _ = dt_engine.predict_one_step(r_stream[k], u2_stream[k])
        _ = dt_engine.correct_with_measurement(r_stream[k], u2_stream[k], y_true_stream[k])
        dt_engine.anchor_pred_chain_to_meas(y_true_stream[k])

    # -------- 1-step prediction array (aligned to k+1) --------
    y_pred = np.full(len(y_true_stream), np.nan, dtype=float)

    for k in range(warmN, len(r_stream) - 1):
        yhat_k1 = dt_engine.predict_one_step(r_stream[k], u2_stream[k])
        y_pred[k + 1] = float(yhat_k1)

        _ = dt_engine.correct_with_measurement(r_stream[k], u2_stream[k], y_true_stream[k])

        if calib_period > 0 and (k % calib_period == 0):
            dt_engine.rls_update(y_true_stream[k])

    # -------- Metrics on valid points only --------
    mask = ~np.isnan(y_pred)
    mae, rmse, mape = compute_mae_rmse_mape(y_true_stream[mask], y_pred[mask])

    # -------- Save pointwise --------
    full_path = save_pointwise_predictions(
        out_dir=cfg.pred_dir,
        calib_period=calib_period,
        t=t,
        r=r_stream,
        u2=u2_stream,
        y_true=y_true_stream,
        y_pred=y_pred,
        warmN=warmN,
    )
    logger.info("Saved pointwise preds: %s", full_path)

    return {
        "calib_period": float(calib_period),
        "warmN": float(warmN),
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
    }


def run_experiment(
    cfg: ExperimentConfig,
    model_params: Dict,
    scalers: Dict[str, float],
) -> pd.DataFrame:
    data = load_data(cfg)

    rows: List[Dict[str, float]] = []
    for calib_period in range(cfg.period_min, cfg.period_max + 1):
        logger.info("Running update_period=%d", calib_period)
        row = run_single_period(cfg, calib_period, model_params, scalers, data)
        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values("calib_period")
    out_df.to_csv(cfg.metrics_out_csv, index=False, encoding="utf-8-sig")
    logger.info("Saved steady-state metrics: %s", cfg.metrics_out_csv)

    return out_df


# ============================
# Main
# ============================
def main() -> None:
    # Model definition (keep your original numbers)
    model_params = {
        "na": 2,
        "nb1": 2,
        "nb2": 2,
        "Lz": 3,
        "a": np.array([-0.692289, -0.089080]),
        "B1": np.array([0.326436, -0.123956]),
        "B2": np.array([1.168503, -1.133168]),
        "c0": -0.005059,
        "beta_Z": np.array([0.8683, -0.1314, -0.0039, 0.2870, 1.1610, -1.4785, 0.2002]),
    }

    scalers = {
        "u1_mu": 0.407482,
        "u1_sd": 0.014054,
        "u2_mu": 1.182368,
        "u2_sd": 0.145389,
        "y_mu": 3107.194627,
        "y_sd": 182.336967,
        "r_mu": 3106.506470,
        "r_sd": 175.136056,
    }

    cfg = ExperimentConfig(
        # Override defaults here if needed
        # data_csv_path="../test_data/20250310.csv",
        # period_min=0,
        # period_max=50,
    )

    df_metrics = run_experiment(cfg, model_params, scalers)
    logger.info("Done. Head of metrics:\n%s", df_metrics.head())


if __name__ == "__main__":
    main()