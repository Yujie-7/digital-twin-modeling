#!/usr/bin/env python3
"""
Steam Flow Digital Twin (Closed-Loop MISO-ARX Plant Identification via 2SLS)

This script:
1) Loads time-series data from a CSV file (steam flow loop)
2) Standardizes signals (z-score)
3) Identifies a closed-loop plant model using two-stage least squares (2SLS)
   - Stage 1: instrument-variable regression to estimate u1_hat (valve opening proxy)
   - Stage 2: ARX regression of y on [u1_hat, u2] (plant-only, excluding r)
4) Grid-searches input delays (d1, d2) to maximize R² (in z-space)
5) Prints the best model coefficients (A, B1, B2, bias)
6) Optionally saves identification summary

"""

from __future__ import annotations

import json
import time
import os
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error


# ============================
# Logging
# ============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ============================
# Config
# ============================
@dataclass(frozen=True)
class IdentifyConfig:
    csv_path: str = "/Users/zoe/Desktop/pre-flow.csv"

    col_time: str = "birth_mills"
    col_u1: str = "lop_stm_vop_inv"     # valve opening
    col_u2: str = "lop_hetg_pr_inv"     # pressure
    col_y: str = "lop_stm_fl_inv"       # measured flow
    col_r: str = "lop_stm_fl_sv"        # setpoint


    # ARX orders
    na: int = 2
    nb1: int = 2
    nb2: int = 2

    # Instrument design
    Lz: int = 3                         # number of lags used in instrument vector

    # Delay grid search
    d_min: int = 0
    d_max: int = 5                      # inclusive

    # Numerical stability
    cond_th: float = 1e6
    ridge: float = 1e-3

    # Output
    save_dir: str = "ident_results"
    save_summary: bool = True


# ============================
# Utils
# ============================
def to_seconds_and_sort(df: pd.DataFrame, col_time: str) -> Tuple[pd.DataFrame, np.ndarray, float]:
    """
    Sort by time and convert ms to seconds if needed.
    Returns (sorted_df, t_seconds, Ts).
    """
    t = df[col_time].astype(float).to_numpy()
    if np.mean(np.diff(t)) > 100:  # heuristic: likely ms
        t = t / 1000.0

    sort_idx = np.argsort(t)
    df = df.iloc[sort_idx].reset_index(drop=True)
    t = t[sort_idx]

    Ts = float(np.median(np.diff(t)))
    return df, t, Ts


def zscore(x: np.ndarray, eps: float = 1e-9) -> Tuple[np.ndarray, float, float]:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < eps:
        raise ValueError("Standard deviation is ~0 (insufficient excitation).")
    return (x - mu) / sd, mu, sd


def zrestore(z: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return z * sd + mu


def lag(sig: np.ndarray, L: int) -> np.ndarray:
    """
    Build lag matrix: [sig(k-1), ..., sig(k-L)].
    Returns shape (N, L) with NaNs at the beginning.
    """
    N = len(sig)
    mats = []
    for l in range(1, L + 1):
        v = np.concatenate([np.full(l, np.nan), sig[:-l]])
        mats.append(v)
    return np.column_stack(mats) if mats else np.zeros((N, 0), dtype=float)


def lstsq_stable(Phi: np.ndarray, Y: np.ndarray, cond_th: float = 1e6, ridge: float = 1e-3) -> np.ndarray:
    """
    Least squares with light ridge when Phi is ill-conditioned.
    """
    c = float(np.linalg.cond(Phi))
    if c < cond_th:
        return np.linalg.lstsq(Phi, Y, rcond=None)[0]

    m = Phi.shape[1]
    return np.linalg.solve(Phi.T @ Phi + ridge * np.eye(m), Phi.T @ Y)


def build_arx_regression(
    yz: np.ndarray,
    inputs: List[np.ndarray],
    na: int,
    delays: List[int],
    nb_list: List[int],
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Build ARX regression:
      y(k) = -a1 y(k-1) - ... - a_na y(k-na)
             + sum_i sum_j b_ij u_i(k-d_i-j) + c0

    Returns (Phi, Y, max_lag).
    """
    N = len(yz)
    max_lag = max([na] + [delays[i] + nb_list[i] - 1 for i in range(len(inputs))])

    rows, targets = [], []
    for k in range(max_lag, N):
        row = [-yz[k - i - 1] for i in range(na)]
        for i, sig in enumerate(inputs):
            d, nb = delays[i], nb_list[i]
            row += [sig[k - d - j] for j in range(nb)]
        row += [1.0]  # bias
        rows.append(row)
        targets.append(yz[k])

    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float), int(max_lag)


# ============================
# 2SLS identification
# ============================
def identify_plant_2sls(
    yz: np.ndarray,
    u1z: np.ndarray,
    u2z: np.ndarray,
    rz: Optional[np.ndarray],
    na: int,
    nb1: int,
    nb2: int,
    d1: int,
    d2: int,
    Lz: int,
    y_mu: float,
    y_sd: float,
    cond_th: float,
    ridge: float,
) -> Dict:
    """
    Stage 1:
      u1_hat = Z * beta_Z
      Z = [r, lag(r, Lz), lag(u2, Lz)]  (if r exists) else [lag(u2, Lz)]

    Stage 2:
      y = ARX(y, [u1_hat, u2])  (plant-only model; excludes r)
    """
    # ---- Stage 1: instruments ----
    Z_blocks = []
    if rz is not None:
        Z_blocks.append(rz.reshape(-1, 1))
        Z_blocks.append(lag(rz, Lz))
    Z_blocks.append(lag(u2z, Lz))

    Z_full = np.hstack(Z_blocks) if Z_blocks else np.zeros((len(yz), 0), dtype=float)
    Z_mask = ~np.any(np.isnan(Z_full), axis=1)

    yz_c = yz[Z_mask]
    u1z_c = u1z[Z_mask]
    u2z_c = u2z[Z_mask]
    Z = Z_full[Z_mask, :]

    beta_Z = lstsq_stable(Z, u1z_c, cond_th=cond_th, ridge=ridge)
    u1_hat = Z @ beta_Z

    # ---- Stage 2: plant ARX ----
    inputs = [u1_hat, u2z_c]
    Phi, Y, max_lag = build_arx_regression(
        yz=yz_c,
        inputs=inputs,
        na=na,
        delays=[d1, d2],
        nb_list=[nb1, nb2],
    )

    theta = lstsq_stable(Phi, Y, cond_th=cond_th, ridge=ridge)
    yhat = Phi @ theta

    R2 = float(r2_score(Y, yhat))
    MAE = float(mean_absolute_error(Y, yhat))

    y_raw = zrestore(yz_c, y_mu, y_sd)[max_lag:]
    yhat_raw = zrestore(yhat, y_mu, y_sd)

    return {
        "theta": theta,
        "beta_Z": beta_Z,
        "R2": R2,
        "MAE_z": MAE,
        "max_lag": max_lag,
        "mask": Z_mask,
        "y_raw": y_raw,
        "yhat_raw": yhat_raw,
        "delays": {"d1": int(d1), "d2": int(d2)},
        "orders": {"na": int(na), "nb1": int(nb1), "nb2": int(nb2), "Lz": int(Lz)},
    }


def unpack_theta(theta: np.ndarray, na: int, nb1: int, nb2: int) -> Dict[str, np.ndarray | float]:
    a = theta[:na]
    offset = na
    B1 = theta[offset : offset + nb1]
    offset += nb1
    B2 = theta[offset : offset + nb2]
    bias = float(theta[-1])
    return {"a": a, "B1": B1, "B2": B2, "bias": bias}


# ============================
# Main
# ============================
def main() -> None:
    cfg = IdentifyConfig()
    os.makedirs(cfg.save_dir, exist_ok=True)

    # ---- Load ----
    df = pd.read_csv(cfg.csv_path)

    if cfg.col_time not in df.columns:
        raise ValueError(f"CSV missing time column: {cfg.col_time}")
    for c in [cfg.col_u1, cfg.col_u2, cfg.col_y]:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")

    df, t, Ts = to_seconds_and_sort(df, cfg.col_time)

    # ---- Extract signals ----
    u1 = df[cfg.col_u1].to_numpy(dtype=float) /100
    u2 = df[cfg.col_u2].to_numpy(dtype=float)
    y = df[cfg.col_y].to_numpy(dtype=float)
    r = df[cfg.col_r].to_numpy(dtype=float) if cfg.col_r in df.columns else None

    mask = ~np.isnan(u1) & ~np.isnan(u2) & ~np.isnan(y)
    if r is not None:
        mask &= ~np.isnan(r)

    u1, u2, y = u1[mask], u2[mask], y[mask]
    if r is not None:
        r = r[mask]

    # ---- Z-score ----
    u1_z, u1_mu, u1_sd = zscore(u1)
    u2_z, u2_mu, u2_sd = zscore(u2)
    y_z, y_mu, y_sd = zscore(y)
    r_z = None
    if r is not None:
        r_z, r_mu, r_sd = zscore(r)

    logger.info("Z-score params: u1(mu=%.6f, sd=%.6f) u2(mu=%.6f, sd=%.6f) y(mu=%.6f, sd=%.6f)",
                u1_mu, u1_sd, u2_mu, u2_sd, y_mu, y_sd)
    if r_z is not None:
        logger.info("Z-score params: r(mu=%.6f, sd=%.6f)", r_mu, r_sd)

    # ---- Grid search delays ----
    best: Optional[Dict] = None

    for d1 in range(cfg.d_min, cfg.d_max + 1):
        for d2 in range(cfg.d_min, cfg.d_max + 1):
            try:
                res = identify_plant_2sls(
                    yz=y_z,
                    u1z=u1_z,
                    u2z=u2_z,
                    rz=r_z,
                    na=cfg.na,
                    nb1=cfg.nb1,
                    nb2=cfg.nb2,
                    d1=d1,
                    d2=d2,
                    Lz=cfg.Lz,
                    y_mu=y_mu,
                    y_sd=y_sd,
                    cond_th=cfg.cond_th,
                    ridge=cfg.ridge,
                )
                if (best is None) or (res["R2"] > best["R2"]):
                    best = res
            except Exception:
                # keep searching; you can log debug if needed
                continue

    

    if best is None:
        raise RuntimeError("No feasible model found in the delay grid search.")

    coef = unpack_theta(best["theta"], cfg.na, cfg.nb1, cfg.nb2)

    logger.info(
        "Best model: R2=%.4f | MAE(z)=%.4f | delays=%s | orders=%s",
        best["R2"],
        best["MAE_z"],
        best["delays"],
        best["orders"],
    )

    logger.info("Plant ARX polynomials:")
    logger.info("A(q^-1) = 1 + %s", np.array2string(coef["a"], precision=6))
    logger.info("B1(u1_hat)(q^-1) = %s", np.array2string(coef["B1"], precision=6))
    logger.info("B2(u2)(q^-1) = %s", np.array2string(coef["B2"], precision=6))
    logger.info("Bias = %.6f", coef["bias"])
    logger.info("beta_Z (instrument model coefficients) = %s",np.array2string(best["beta_Z"], precision=6)
)
    # ---- Save summary ----
    if cfg.save_summary:
        summary = {
            "csv_path": cfg.csv_path,
            "Ts_sec": Ts,
            "orders": best["orders"],
            "delays": best["delays"],
            "R2_z": best["R2"],
            "MAE_z": best["MAE_z"],
            "theta": best["theta"].tolist(),
            "beta_Z": best["beta_Z"].tolist(),
            "zscore": {
                "u1_mu": u1_mu, "u1_sd": u1_sd,
                "u2_mu": u2_mu, "u2_sd": u2_sd,
                "y_mu": y_mu, "y_sd": y_sd,
                "r_mu": r_mu if r_z is not None else None,
                "r_sd": r_sd if r_z is not None else None,
            },
        }
        save_path = os.path.join(cfg.save_dir, "plant_arx_2sls_best_summary.csv")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved summary csv: %s", os.path.abspath(save_path))



if __name__ == "__main__":
    main()