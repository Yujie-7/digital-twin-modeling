#!/usr/bin/env python3
"""
Steam Flow Digital Twin (LSTM Training)

This script:
1) Loads multiple CSV files for training
2) Fits a MinMaxScaler on ALL concatenated training rows (inputs + outputs)
3) Builds sequences *within each file* to avoid cross-file windowing
4) Trains an LSTM for one-step-ahead prediction
5) Evaluates on a test CSV (scaler is NOT refit)
6) Saves model checkpoint (state_dict + scaler + metadata) to a .pth file

"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error


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
class TrainConfig:
    # Data
    train_files: List[str]
    test_file: str

    inputs: List[str]
    outputs: List[str]

    # Model
    time_steps: int = 10
    hidden_dim: int = 64

    # Training
    epochs: int = 300
    lr: float = 5e-4
    batch_size: int = 64
    val_ratio: float = 0.0  # 0 disables validation split (sequence-level shuffle split)

    # Runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0

    # Output
    save_path: str = "lstm.pth"


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
# Helpers
# ============================
def create_sequences(
    data_2d: np.ndarray,
    input_len: int,
    output_len: int,
    time_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    data_2d: shape (T, input_len + output_len) in *scaled* space
    returns:
      X: (N, time_steps, input_len)
      y: (N, output_len)
    """
    X, y = [], []
    T = len(data_2d)
    for i in range(T - time_steps):
        X.append(data_2d[i : i + time_steps, :input_len])
        y.append(data_2d[i + time_steps, input_len : input_len + output_len])
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Put per-file preprocessing here to ensure consistency across all files.
    """
    # Example hook (keep if you plan to normalize/clip/filter):
    if "lop_stm_vop_inv" in df.columns:
        df["lop_stm_vop_inv"] = df["lop_stm_vop_inv"].astype(float)
    return df


def load_clean_df(fp: str, cols: List[str]) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df = preprocess_df(df)
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in file: {fp}")
    df = df[cols].dropna().reset_index(drop=True)
    return df


def fit_scaler_on_training_rows(
    dfs: List[pd.DataFrame],
    feature_cols: List[str],
) -> MinMaxScaler:
    """
    Fit MinMaxScaler on concatenated training rows (preserve feature names).
    """
    all_train = pd.concat(dfs, axis=0, ignore_index=True)
    scaler = MinMaxScaler()
    scaler.fit(all_train[feature_cols])  # DataFrame -> keeps feature names
    return scaler


def build_sequences_from_files_no_cross_window(
    file_list: List[str],
    inputs: List[str],
    outputs: List[str],
    time_steps: int,
    scaler: Optional[MinMaxScaler] = None,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    
    feature_cols = inputs + outputs
    dfs = [load_clean_df(fp, feature_cols) for fp in file_list]

    if scaler is None:
        scaler = fit_scaler_on_training_rows(dfs, feature_cols)

    X_all, y_all = [], []
    for d in dfs:
        data_scaled = scaler.transform(d[feature_cols])  # DataFrame -> no feature-name warning
        X, y = create_sequences(data_scaled, len(inputs), len(outputs), time_steps)
        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)

    if not X_all:
        raise ValueError("No sequences created. Check time_steps or data length in training files.")

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    return X_all, y_all, scaler


def split_train_val(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Random split at the sequence level.
    """
    if val_ratio <= 0:
        return X, y, None, None

    rng = np.random.default_rng(seed)
    n = len(X)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_val = int(n * val_ratio)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int,
    lr: float,
    device: str,
    log_every: int = 20,
) -> nn.Module:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        model.train()
        train_losses = []

        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        if (ep % log_every == 0) or (ep in (1, epochs)):
            msg = f"Epoch {ep:3d}/{epochs} | train_loss={np.mean(train_losses):.6f}"
            if val_loader is not None:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for Xv, yv in val_loader:
                        Xv = Xv.to(device)
                        yv = yv.to(device)
                        pv = model(Xv)
                        lv = criterion(pv, yv)
                        val_losses.append(float(lv.item()))
                msg += f" | val_loss={np.mean(val_losses):.6f}"
            logger.info(msg)

    t1 = time.perf_counter()
    logger.info("Training time: %.3f s (%d epochs)", (t1 - t0), epochs)
    return model


def evaluate_single_output(
    model: nn.Module,
    X_test_t: torch.Tensor,
    y_test_np: np.ndarray,
    scaler: MinMaxScaler,
    inputs: List[str],
    outputs: List[str],
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate and return:
      y_true_rescaled: (N, output_len)
      y_pred_rescaled: (N, output_len)
      rmse: (output_len,)
      mae:  (output_len,)

    Note:
    inverse_transform needs full vector = inputs + outputs,
    so we concatenate last input step + output.
    """
    model.eval()
    model.to(device)

    with torch.no_grad():
        y_pred_np = model(X_test_t.to(device)).cpu().numpy()

    X_last = X_test_t[:, -1, :].cpu().numpy()  # (N, input_len)

    y_true_full = np.concatenate([X_last, y_test_np], axis=1)
    y_pred_full = np.concatenate([X_last, y_pred_np], axis=1)

    # Use DataFrame to avoid feature-name warnings
    feat_cols = inputs + outputs
    y_true_full_df = pd.DataFrame(y_true_full, columns=feat_cols)
    y_pred_full_df = pd.DataFrame(y_pred_full, columns=feat_cols)

    y_true_rescaled = scaler.inverse_transform(y_true_full_df)[:, len(inputs) :]
    y_pred_rescaled = scaler.inverse_transform(y_pred_full_df)[:, len(inputs) :]

    rmse = np.sqrt(mean_squared_error(y_true_rescaled, y_pred_rescaled, multioutput="raw_values"))
    mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled, multioutput="raw_values")
    mape= mean_absolute_percentage_error(y_true_rescaled, y_pred_rescaled, multioutput="raw_values")
    return  rmse, mae, mape


def save_checkpoint(
    save_path: str,
    model: nn.Module,
    cfg: TrainConfig,
    scaler: MinMaxScaler,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "model_config": {
            "input_dim": len(cfg.inputs),
            "hidden_dim": cfg.hidden_dim,
            "output_dim": len(cfg.outputs),
        },
        "scaler": scaler,
        "inputs": cfg.inputs,
        "outputs": cfg.outputs,
        "time_steps": cfg.time_steps,
        "train_files": cfg.train_files,
        "test_file": cfg.test_file,
        "train_hparams": {
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "val_ratio": cfg.val_ratio,
        },
    }
    torch.save(payload, save_path)
    logger.info("Saved checkpoint: %s", os.path.abspath(save_path))


# ============================
# Main
# ============================
def main() -> None:
    cfg = TrainConfig(
        train_files=[
            "/Users/zoe/Desktop/pre-flow.csv",
        ],
        test_file="../test_data/20250310.csv",
        inputs=["lop_stm_fl_sv", "lop_hetg_pr_inv", "lop_stm_vop_inv"],
        outputs=["lop_stm_fl_inv"],
        time_steps=10,
        hidden_dim=64,
        epochs=300,
        lr=5e-4,
        batch_size=64,
        val_ratio=0.0,
        save_path="lstm.pth",
    )

    logger.info("Device: %s", cfg.device)
    logger.info("Inputs: %s", cfg.inputs)
    logger.info("Outputs: %s", cfg.outputs)
    logger.info("time_steps: %d", cfg.time_steps)

    # ---- Build TRAIN sequences ----
    X_train, y_train, scaler = build_sequences_from_files_no_cross_window(
        cfg.train_files, cfg.inputs, cfg.outputs, cfg.time_steps, scaler=None
    )
    logger.info("Train sequences: X=%s y=%s", X_train.shape, y_train.shape)

    # ---- Train/Val split ----
    X_tr, y_tr, X_val, y_val = split_train_val(X_train, y_train, cfg.val_ratio, seed=0)

    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                             torch.tensor(y_tr, dtype=torch.float32))
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.startswith("cuda")),
    )

    val_loader = None
    if X_val is not None:
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                               torch.tensor(y_val, dtype=torch.float32))
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(cfg.device.startswith("cuda")),
        )
        logger.info("Val sequences: X=%s y=%s", X_val.shape, y_val.shape)

    # ---- Build TEST sequences (single file; scaler NOT refit) ----
    feature_cols = cfg.inputs + cfg.outputs
    df_test = load_clean_df(cfg.test_file, feature_cols)
    test_scaled = scaler.transform(df_test[feature_cols])  # DataFrame -> no warning
    X_test, y_test = create_sequences(test_scaled, len(cfg.inputs), len(cfg.outputs), cfg.time_steps)
    logger.info("Test sequences: X=%s y=%s", X_test.shape, y_test.shape)

    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # ---- Train ----
    model = LSTMModel(input_dim=len(cfg.inputs), hidden_dim=cfg.hidden_dim, output_dim=len(cfg.outputs))
    model = train_model(model, train_loader, val_loader, cfg.epochs, cfg.lr, cfg.device)

    # ---- Evaluate ----
    rmse, mae, mape = evaluate_single_output(
        model=model,
        X_test_t=X_test_t,
        y_test_np=y_test,
        scaler=scaler,
        inputs=cfg.inputs,
        outputs=cfg.outputs,
        device=cfg.device,
    )

    logger.info("Test performance:")
    for i, name in enumerate(cfg.outputs):
        logger.info("  %s | RMSE=%.4f | MAE=%.4f| MAPE=%.4f" , name, float(rmse[i]), float(mae[i]), float(mape[i]))

    # ---- Save checkpoint ----
    save_checkpoint(cfg.save_path, model, cfg, scaler)


if __name__ == "__main__":
    main()