# Federated Learning Under the Lens of Task Arithmetic

**Course:** Advanced Machine Learning — Politecnico di Torino  
**Authors:** XI FAN (s328208) · Marco Roman (s343781) · Davide Randino (s346257) · Adrien Trahan (s350155)

---

## Overview

This project investigates whether **sensitivity-aware sparse fine-tuning** (rooted in the [TaLoS framework](https://arxiv.org/abs/2408.xxxxx)) can alleviate client drift and preserve pre-trained representations in Federated Learning.  
We use a **DINO ViT-S/16** backbone on **CIFAR-100** and systematically compare:

- Centralized training baselines (standard + sparse)
- FedAvg under IID and non-IID data distributions
- Task Arithmetic (SparseSGDM) variants across sparsity levels, calibration rounds, and mask strategies

---

## Repository Structure

```
.
├── datapreprocessing.py          # Dataset loading, IID/non-IID partitioning
├── centralizedmodel.py           # Model definition, centralized training pipeline
├── centralizedtaskarithmetic.py  # Centralized sparse fine-tuning (Task Arithmetic baseline)
├── taskarithmetic.py             # SparseSGDM optimizer, Fisher sensitivity, mask calibration
├── fed_iid.py                    # FedAvg — IID setting
├── fed_non_iid.py                # FedAvg — Non-IID setting (Nc × J grid)
├── fed_iid_taskarithmetic.py     # FedAvg + Task Arithmetic — IID
├── fed_non_iid_taskarithmetic.py # FedAvg + Task Arithmetic — Non-IID
├── test_partition.py             # Visualize and validate data partitioning
└── test.py                       # Plot frozen-backbone FedAvg convergence curves
```

---

## Requirements

```bash
pip install torch torchvision numpy matplotlib
```

> **GPU recommended.** All experiments were developed for Google Colab (free tier).  
> The code automatically detects CUDA and falls back to CPU.

DINO weights are fetched automatically via `torch.hub` on first run (requires internet access).

---

## Quickstart

### 1. Verify data partitioning

```bash
python test_partition.py
```

Outputs:
- Console report of IID / non-IID label distributions per client
- `partition_comparison.png` — per-client sample counts and class distributions
- `class_coverage_heatmap.png` — class coverage heatmap for Nc ∈ {1, 2, 5, 10, 20}

---

### 2. Centralized baseline

```bash
python centralizedmodel.py
```

Trains a DINO ViT-S/16 + linear head on the full CIFAR-100 training set.

**Key hyperparameters** (editable in `__main__`):

| Parameter | Value |
|---|---|
| Epochs | 40 |
| Learning rate | 0.001 |
| Batch size | 128 |
| Momentum | 0.9 |
| Weight decay | 5e-4 |
| Warmup epochs | 5 |
| Backbone unfreeze | epoch 20 |

**Outputs:**
- `best_centralized_model.pth` — best checkpoint by validation accuracy
- `centralizedmodel_training_history.json` — full loss/accuracy history
- `centralized_model_training_curves.png` — training curves plot

---

### 3. Centralized Task Arithmetic baseline

```bash
python centralizedtaskarithmetic.py
```

Trains the same model but with gradient masking via Fisher Information sensitivity.

```python
# Edit strategy and sparsity in __main__:
run_centralized_task_arithmetic(strategy='least_sensitive', sparsity_ratio=0.1)
```

Available strategies: `least_sensitive` · `most_sensitive` · `low_magnitude` · `high_magnitude` · `random`

---

### 4. FedAvg — IID

```bash
python fed_iid.py
```

Runs FedAvg with **K=100 clients, C=0.1, J=4, 200 rounds** on an IID partition.

**Outputs:**
- `fedavg_training_history.json`
- Checkpoints every 10 rounds: `fedavg_checkpoint_r{N}.pth`

---

### 5. FedAvg — Non-IID grid

```bash
python fed_non_iid.py
```

Runs the full **Nc × J** grid required by the project specification:

| Nc (classes/client) | J (local steps) | Rounds |
|---|---|---|
| 1, 5, 10, 50 | 4 | 200 |
| 1, 5, 10, 50 | 8 | 100 |
| 1, 5, 10, 50 | 16 | 50 |

> **Note:** This runs 12 experiments sequentially. On Colab free tier each experiment takes ~30–60 min. Use checkpointing to resume interrupted runs.

**Outputs per experiment:**
- `non_iid_history_Nc{Nc}_J{J}.json`
- Checkpoints in `checkpoint/fed_non_iid/J_{J}_Nc_{Nc}/`

To run a single configuration:

```python
# In fed_non_iid.py, replace __main__ with:
results = run_fed_non_iid_experiment(Nc_value=10, J_value=8)
```

---

### 6. FedAvg + Task Arithmetic — IID

```bash
python fed_iid_taskarithmetic.py
```

Each client computes its own Fisher mask locally before each round.

**Key settings** (edit inside `run_federated_task_arithmetic_iid()`):

```python
K       = 100          # total clients
C       = 0.1          # participation fraction
ROUNDS  = 50
J       = 4            # local steps
STRATEGY = 'least_sensitive'
SPARSITY = 0.1         # fraction of parameters to update
```

---

### 7. FedAvg + Task Arithmetic — Non-IID

```bash
python fed_non_iid_taskarithmetic.py
```

Same as above but with non-IID data partitioning.

**Key settings** (edit inside `run_federated_task_arithmetic_non_iid()`):

```python
K        = 100
C        = 0.1
Nc       = 10          # classes per client (integer, e.g. 1, 5, 10, 50)
ROUNDS   = 100
J        = 8
STRATEGY = 'least_sensitive'
SPARSITY = 0.05
```

---

## Reproducing the Paper Figures

### Figure 4 — Final accuracy vs. sparsity level

After running experiments with `fed_non_iid_taskarithmetic.py` for all sparsity × Rcal combinations, use:

```bash
python plot_figures.py   # (if available) or adapt test.py
```

Replace the placeholder data arrays in `plot_figures.py` with values from your saved JSON logs.

### Figure 5 — Accuracy trajectories by calibration rounds

Same script, using per-round accuracy arrays loaded from the JSON history files.

---

## Checkpointing and Resuming on Colab

All training scripts save checkpoints periodically. To resume a run:

```python
# Load checkpoint and continue from round R:
global_model.load_state_dict(torch.load('fedavg_checkpoint_r100.pth'))
# then start the loop from r = 100
```

We recommend saving checkpoints to **Google Drive** on Colab:

```python
save_dir = '/content/drive/MyDrive/AML_checkpoints/...'
```

---

## Experiment Summary

| Script | Setting | Rounds | Key output |
|---|---|---|---|
| `centralizedmodel.py` | Centralized | 40 epochs | `best_centralized_model.pth` |
| `centralizedtaskarithmetic.py` | Centralized + TA | 30 epochs | val accuracy |
| `fed_iid.py` | FedAvg IID | 200 | `fedavg_training_history.json` |
| `fed_non_iid.py` | FedAvg Non-IID (12 configs) | 50–200 | per-config JSON + checkpoints |
| `fed_iid_taskarithmetic.py` | FedAvg IID + TA | 50 | test accuracy per round |
| `fed_non_iid_taskarithmetic.py` | FedAvg Non-IID + TA | 100 | test accuracy per round |

---

## Notes

- `datapreprocessing.py` applies **with-replacement sampling** for non-IID partitions; clients may share some images. This is a deliberate simplification — see the report for discussion.
- Test transforms currently use `Resize(256)` without a center crop. For strict 224×224 consistency, add `transforms.CenterCrop(224)` to `test_transforms` in `datapreprocessing.py`.
- `fed_non_iid.py` scales rounds as `ROUNDS = 800 // J_value` to keep total gradient steps constant across J configurations.
