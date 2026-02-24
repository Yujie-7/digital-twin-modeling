# Digital Twin Modeling and Evaluation (Steam Flow Control)

This repository provides the implementation of three representative digital twin (DT) modeling approaches for an industrial steam flow control process, together with a unified evaluation framework for **steady-state** and **transient** metrics.

**Included DT models**
- **Physics-based model**
- **Data-driven model (LSTM)**
- **Hybrid model (System Identification)**

**Evaluation**
- Steady-state Error: MAE / RMSE / MSE 
- Transient Error: Settling Time (Ts), Peak Time (Tp), Overshoot (OS)

---

## 📁 Repository Structure

```
.
├── data/
│   ├── train/                      # Training datasets
│   └── test/                       # Testing datasets
│
├── model/
│   ├── lstm/                       # LSTM model code
│   ├── system-identification/      # System-identification model code
│   └── physics-based/              # Physics-based model code
│
├── results/
│   ├── predictions/                # Saved predictions of each model
│   ├── evaluation metrics/         # Saved evaluation results 
│   └── figures/                    # Saved figures
│
├── transient_performance.py        # transient performance analysis scripts
│
└── README.md
```
---

## 📊 Data

**Note:** The original industrial dataset used in this study is confidential and cannot be publicly released.

### 2.1 Training and Testing Data

- `data/train/`  
  Used for model training and parameter estimation.

- `data/test/`  
  Used for evaluating model performance under unseen operating conditions.

### 2.2 Data Description

Each dataset typically includes the following variables:

- `t`: time index  
- `r(t)`: steam flow setpoint  
- `y(t)`: measured steam flow  
- Additional process variables (e.g., valve opening, upstream/downstream pressure)

---

## 🚀 Installation

```
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run Each Model

Run each model script to generate predictions and steady-state metrics:

```
python model/lstm/lstm.py

python model/system-identification/system-identification.py

python model/physics-based/physics-based.py
```

Each model will:

- Generate prediction results  
- Compute steady-state metrics

Outputs will be saved to:

```
results/predictions/
results/evaluation_metrics/
```

---

### Transient Performance Evaluation

You need to **aggregate valid prediction results from all three models** into the same file

Run the unified transient analysis:

```
python transient_performance.py
```

This script will:

- Compute transient metrics

Outputs will be saved to:

```
results/evaluation_metrics/summary_transient_metrics.csv
results/figures/
```
---