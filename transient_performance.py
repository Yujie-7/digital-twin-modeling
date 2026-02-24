"""
Steam Flow Digital Twin - Dynamic Metrics (Overshoot/Undershoot, Peak/Trough time, Ts)

This script:
1) Loads per-sample prediction CSV (columns: r, y_true, and multiple model preds; optional: t)
2) Detects step changes using setpoint r (MAD threshold on diff)
3) For each step segment, computes:
   - steady-state value (tail mean)
   - Overshoot (%) + Peak time (s) for rising steps
   - Undershoot (%) + Trough time (s) for falling steps
   - Settling time Ts (s): first entry into ±band% and stays for hold_sec
4) Plots step windows with annotations
5) Saves dynamic metrics to CSV and figure to PNG
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# Logging
# ============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EPS = 1e-12


# ============================
# Config
# ============================
@dataclass(frozen=True)
class ExperimentConfig:
    # Input
    pred_csv_path: str = "data/test/filename"

    # Columns
    col_time: str = "t"   # if no time column, set None
    col_setpoint: str = "r"
    col_y_true: str = "y_true"

    # Multi-model prediction columns
    pred_cols: Dict[str, str] = None  

    # Output
    metrics_out_csv: str = "results/evaluation metrics/summary_transient_error.csv"
    fig_out_png: str = "results/figures/transient_compare.png"

    # Plot window around each step (seconds)
    pre_sec: float = 100.0
    post_sec: float = 200.0

    # Steady-state / settling-time parameters
    steady_tail_sec: float = 180.0
    band_ratio: float = 0.05
    hold_sec: float = 60.0

    # Step detection parameters (MAD-based threshold on diff)
    thr_k: float = 3.0
    min_gap_sec: float = 20.0

    # Plot style
    figsize: Tuple[int, int] = (16, 7)
    lw_actual: float = 1.8
    lw_pred: float = 3.0
    styles: Dict[str, Tuple[str, str]] = None  # (color, linestyle)

    def __post_init__(self):
        object.__setattr__(
            self,
            "pred_cols",
            self.pred_cols
            or {
                "LSTM": "lstm_y_pred",
                "Identification": "sys_y_pred",
                "Physics-based": "phy_y_pred",
            },
        )
        object.__setattr__(
            self,
            "styles",
            self.styles
            or {
                "Actual": ("gray", "--"),
                "LSTM": ("#7391C7", "-"),
                "Identification": ("#F0B043", "-"),
                "Physics-based": ("#859E69", "-"),
            },
        )


# ============================
# Helpers
# ============================
def robust_dt_from_time(t: Sequence[float]) -> float:
    t = np.asarray(t, dtype=float)
    t_plot = t - t[0]
    d = np.diff(t_plot)
    d = d[np.isfinite(d)]
    d = d[d > 0]
    if len(d) == 0:
        raise ValueError("Cannot infer dt: time differences are non-positive or empty.")
    return float(np.median(d))


def detect_steps(sig: np.ndarray, dt: float, cfg: ExperimentConfig) -> Tuple[List[int], List[int], float]:
    sig = np.asarray(sig, dtype=float)
    d = np.diff(sig)

    mad = np.median(np.abs(d - np.median(d))) + EPS
    sigma = 1.4826 * mad
    thr = max(cfg.thr_k * sigma, 1e-6)

    cand = np.where(np.abs(d) > thr)[0] + 1
    min_gap_N = max(1, int(cfg.min_gap_sec / dt))

    step_points: List[int] = []
    step_dirs: List[int] = []
    for idx in cand:
        if (not step_points) or (idx - step_points[-1] >= min_gap_N):
            k0 = int(idx - 1)
            step_points.append(k0)
            step_dirs.append(1 if (sig[idx] - sig[idx - 1]) > 0 else -1)

    return step_points, step_dirs, thr


def compute_segment_metrics(
    arr: np.ndarray,
    t_plot: np.ndarray,
    k0: int,
    k1: int,
    direction: int,
    dt: float,
    cfg: ExperimentConfig,
) -> Optional[Dict[str, float]]:
    arr = np.asarray(arr, dtype=float)
    Q_seg = arr[k0:k1]
    t_seg = t_plot[k0:k1]

    steady_tail_N = max(5, int(cfg.steady_tail_sec / dt))
    hold_N = max(2, int(cfg.hold_sec / dt))

    if len(Q_seg) < max(10, steady_tail_N + 2):
        return None

    steady = float(np.mean(Q_seg[-steady_tail_N:]))
    low_band = steady * (1.0 - cfg.band_ratio)
    high_band = steady * (1.0 + cfg.band_ratio)

    t0 = float(t_plot[k0])

    # Overshoot/Undershoot + Peak/Trough time
    overshoot_pct = np.nan
    undershoot_pct = np.nan
    peak_time_s = np.nan
    trough_time_s = np.nan

    if direction > 0:
        i_peak = int(np.argmax(Q_seg))
        peak_val = float(Q_seg[i_peak])
        peak_t = float(t_seg[i_peak])

        overshoot_pct = max(0.0, (peak_val - steady) / (abs(steady) + EPS) * 100.0)
        peak_time_s = max(0.0, peak_t - t0)

        extremum_type = "peak"
        extremum_value = peak_val
        extremum_time = peak_t
    else:
        i_trough = int(np.argmin(Q_seg))
        trough_val = float(Q_seg[i_trough])
        trough_t = float(t_seg[i_trough])

        undershoot_pct = max(0.0, (steady - trough_val) / (abs(steady) + EPS) * 100.0)
        trough_time_s = max(0.0, trough_t - t0)

        extremum_type = "trough"
        extremum_value = trough_val
        extremum_time = trough_t

    # Settling time Ts: first time enters band and stays hold_N
    inside = (Q_seg >= low_band) & (Q_seg <= high_band)
    count = 0
    first_idx: Optional[int] = None
    for i, ok in enumerate(inside):
        if ok:
            if count == 0:
                first_idx = i
            count += 1
        else:
            count = 0
            first_idx = None
        if count >= hold_N:
            break

    if first_idx is None:
        Ts = np.nan
        Ts_abs_time = np.nan
    else:
        Ts_abs_time = float(t_seg[first_idx])
        Ts = max(0.0, Ts_abs_time - t0)

    return {
        "steady_value": steady,
        "low_band": low_band,
        "high_band": high_band,

        "overshoot_percent": float(overshoot_pct) if np.isfinite(overshoot_pct) else np.nan,
        "undershoot_percent": float(undershoot_pct) if np.isfinite(undershoot_pct) else np.nan,
        "peak_time_s": float(peak_time_s) if np.isfinite(peak_time_s) else np.nan,
        "trough_time_s": float(trough_time_s) if np.isfinite(trough_time_s) else np.nan,

        "settling_time_Ts_s": float(Ts),
        "settle_abs_time": float(Ts_abs_time),

        "extremum_type": extremum_type,
        "extremum_value": extremum_value,
        "extremum_time": extremum_time,
    }


def load_data(cfg: ExperimentConfig) -> Dict[str, np.ndarray]:
    df = pd.read_csv(cfg.pred_csv_path)

    # required columns
    for col in [cfg.col_setpoint, cfg.col_y_true]:
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}' in {cfg.pred_csv_path}")

    # model columns
    for model_name, col in cfg.pred_cols.items():
        if col not in df.columns:
            raise KeyError(f"Missing prediction column '{col}' for model '{model_name}' in {cfg.pred_csv_path}")

    r = df[cfg.col_setpoint].to_numpy(dtype=float)
    y_true = df[cfg.col_y_true].to_numpy(dtype=float)

    preds: Dict[str, np.ndarray] = {m: df[c].to_numpy(dtype=float) for m, c in cfg.pred_cols.items()}

    # time

    t = df[cfg.col_time].to_numpy(dtype=float)
    
    return {"t": t, "r": r, "y_true": y_true, **preds}


def save_metrics(cfg: ExperimentConfig, records: List[Dict[str, float]]) -> str:
    out_df = pd.DataFrame(records).sort_values(["step_id", "model"]).reset_index(drop=True)
    out_df.to_csv(cfg.metrics_out_csv, index=False, encoding="utf-8-sig")
    logger.info("Saved dynamic metrics: %s", cfg.metrics_out_csv)
    return cfg.metrics_out_csv


# ============================
# Plotting
# ============================
def plot_steps(
    cfg: ExperimentConfig,
    t_plot: np.ndarray,
    step_points: List[int],
    step_dirs: List[int],
    signals: Dict[str, np.ndarray],
    records_by_step_model: Dict[Tuple[int, str], Dict[str, float]],
    dt: float,
) -> None:
    pre_N = int(cfg.pre_sec / dt)
    post_N = int(cfg.post_sec / dt)

    plt.figure(figsize=cfg.figsize)

    def window_ylim(s: int, e: int) -> Tuple[float, float]:
        y_min = min(np.min(arr[s:e]) for arr in signals.values())
        y_max = max(np.max(arr[s:e]) for arr in signals.values())
        return float(y_min), float(y_max)
    
    os_offset_ratio = {"Actual": 0.18, "LSTM": 0.21, "Identification": 0.24, "Physics-based": 0.27}
    ts_offset_ratio = {"Actual": 0.06, "LSTM": 0.03, "Identification": 0.00, "Physics-based": -0.03}

    for si, (k0, direction) in enumerate(zip(step_points, step_dirs), start=1):
        k1 = step_points[si] if si < len(step_points) else len(t_plot)

        s = max(0, k0 - pre_N)
        e = min(len(t_plot), k0 + post_N)

        t_win = t_plot[s:e]
        y_min, y_max = window_ylim(s, e)
        y_span = max(EPS, y_max - y_min)

        t_start = float(t_plot[k0])

        # curves (legend only once)
        for name, arr in signals.items():
            color, style = cfg.styles[name]
            lw = cfg.lw_actual if name == "Actual" else cfg.lw_pred
            plt.plot(t_win, arr[s:e], linestyle=style, color=color, linewidth=lw,
                     label=name if si == 1 else None)

        # step line
        plt.axvline(t_plot[k0], color="k", linestyle="--", alpha=0.45)

        # annotations
        for name in signals.keys():
            key = (si, name)
            if key not in records_by_step_model:
                continue
            m = records_by_step_model[key]
            color, _ = cfg.styles[name]

            # steady lines in window
            x0 = max(t_plot[k0], t_plot[s])
            x1 = min(t_plot[k1 - 1], t_plot[e - 1])
            if x1 > x0:
                plt.hlines(m["steady_value"], x0, x1, colors=color, linestyles="--", linewidth=1.1, alpha=0.7)
                plt.hlines([m["low_band"], m["high_band"]], x0, x1, colors=color, linestyles=":", linewidth=0.8, alpha=0.45)

            # extremum mark + overshoot/undershoot
            ex_t = m["extremum_time"]
            ex_v = m["extremum_value"]
            if t_plot[s] <= ex_t <= t_plot[e - 1]:
                plt.scatter(ex_t, ex_v, s=18, color=color, zorder=5)

                if direction > 0:
                    val = m["overshoot_percent"]
                    if np.isfinite(val) and val > 0:
                        plt.annotate("", xy=(ex_t, ex_v), xytext=(ex_t, m["steady_value"]),
                                     arrowprops=dict(arrowstyle="<->", color=color, lw=1.2))
                        plt.text(t_start+80, y_max - os_offset_ratio.get(name, 0.0) * y_span, f"{name} Overshoot={val:.1f}%",
                                 color=color, fontsize=12, ha="center", va="bottom", fontweight="bold")
                else:
                    val = m["undershoot_percent"]
                    if np.isfinite(val) and val > 0:
                        plt.annotate("", xy=(ex_t, m["steady_value"]), xytext=(ex_t, ex_v),
                                     arrowprops=dict(arrowstyle="<->", color=color, lw=1.2))
                        plt.text(t_start+80, y_max - os_offset_ratio.get(name, 0.0) * y_span, f"{name} Undershoot={val:.1f}%",
                                 color=color, fontsize=12, ha="center", va="top", fontweight="bold")

            # Ts arrow
            if np.isfinite(m["settling_time_Ts_s"]) and np.isfinite(m["settle_abs_time"]):
                t_end = float(m["settle_abs_time"])
                if t_plot[s] <= t_end <= t_plot[e - 1]:
                    plt.axvline(t_end, color=color, linestyle="--", alpha=0.35)
                    plt.annotate("", xy=(t_end, y_min + ts_offset_ratio.get(name, 0.0) * y_span), xytext=(t_start, y_min + ts_offset_ratio.get(name, 0.0) * y_span),
                                 arrowprops=dict(arrowstyle="<->", color=color, lw=1.1))
                    plt.text(t_start+70 ,y_min + ts_offset_ratio.get(name, 0.0) * y_span,
                             f"{name} Ts={m['settling_time_Ts_s']:.1f}s",
                             color=color, fontsize=12, ha="center", fontweight="bold")

    plt.xlabel("Time (s)", fontsize=12, fontweight="bold")
    plt.ylabel("Steam Flow (kg/h)", fontsize=12, fontweight="bold")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=12, loc="best")
    plt.tight_layout()
    plt.savefig(cfg.fig_out_png, dpi=300)
    plt.show()
    logger.info("Saved figure: %s", cfg.fig_out_png)


# ============================
# Runner
# ============================
def run_dynamic_metrics(cfg: ExperimentConfig) -> pd.DataFrame:
    data = load_data(cfg)

    t = data["t"]
    r = data["r"]
    y_true = data["y_true"]

    t_plot = t - t[0]
    dt = robust_dt_from_time(t)

    # step detection uses setpoint r
    step_points, step_dirs, thr = detect_steps(r, dt, cfg)
    logger.info("[Step detection] signal=setpoint(%s), thr=%.6g, steps=%d", cfg.col_setpoint, thr, len(step_points))

    # signals to evaluate
    signals: Dict[str, np.ndarray] = {"Actual": y_true}
    for model_name in cfg.pred_cols.keys():
        signals[model_name] = data[model_name]

    # compute metrics
    records: List[Dict[str, float]] = []
    rec_map: Dict[Tuple[int, str], Dict[str, float]] = {}

    for si, (k0, direction) in enumerate(zip(step_points, step_dirs), start=1):
        k1 = step_points[si] if si < len(step_points) else len(t_plot)

        for model_name, arr in signals.items():
            m = compute_segment_metrics(arr, t_plot, k0, k1, direction, dt, cfg)
            if m is None:
                continue

            row = {
                "step_id": si,
                "step_index": int(k0),
                "step_time": float(t_plot[k0]),
                "direction": "rise" if direction > 0 else "fall",
                "model": model_name,
                **m,
            }
            records.append(row)
            rec_map[(si, model_name)] = row

    save_metrics(cfg, records)
    plot_steps(cfg, t_plot, step_points, step_dirs, signals, rec_map, dt)

    return pd.DataFrame(records).sort_values(["step_id", "model"]).reset_index(drop=True)


# ============================
# Main
# ============================
def main() -> None:
    cfg = ExperimentConfig(
        # If no t column:
        # col_time=None,
        # sample_period_sec=1.0,
    )

    df_metrics = run_dynamic_metrics(cfg)
    logger.info("Done. Head of metrics:\n%s", df_metrics.head())


if __name__ == "__main__":
    main()