# TRuHEAL-IoMT

**Code implementation by [Ishan Gupta](https://github.com/IshanG-ai)** and (https://github.com/vaishalimeena) Vaishali Meena

PyTorch implementation of TRuHEAL, from *"Proactive Trust-Driven Intrusion Detection in IoMT-Based
Smart Healthcare Ecosystem"* (V. Meena, G. Indra, A. K. Das, Y. Park), evaluated on CICIoMT2024.

Pipeline: AETL (online EWMA normalization + sliding-window moments) -> Type-3 fuzzy trust
(z-slice, Karnik-Mendel) -> relational WGAN-GP reputation -> adaptive trust-reputation fusion ->
Zero-Trust decision (Allow / Challenge / Deny).

## Repository structure
| Path | Purpose |
|---|---|
| `truheal/` | config, data pipeline, model, training and evaluation |
| `scripts/run_main.py` | binary / 6-class / 19-class training and evaluation |
| `scripts/run_ablations.py` | component, trust-variant, robustness and reputation-learner ablations |
| `scripts/run_synthetic.py` | noise-perturbed replay evaluation |
| `scripts/run_ns3.py`, `ns3/iomt.cc` | ns-3.40 simulation and classification demo |

## Setup
```bash
pip install -r requirements.txt
export IOMT_DATA_DIR=/path/to/CICIoMT2024
export TRUHEAL_WORK_DIR=./work
```
Dataset: CICIoMT2024 (Canadian Institute for Cybersecurity), not redistributed here.
GPU recommended (developed on a Tesla T4). Seed is fixed to 42.

## Run
```bash
python scripts/run_main.py
python scripts/run_ablations.py
python scripts/run_synthetic.py
python scripts/run_ns3.py --build
```
Outputs are written to `work/results/`.

## Configuration
| Component | Setting |
|---|---|
| T3FL z-slices | 4 |
| Input features | 46 |
| Fuzzy rules | 27 |
| WGAN-GP epochs / batch size | 100 / 128 |
| Gradient penalty (lambda_GP) | 10 |
| Optimizer / learning rate | Adam / 1e-3 |
| Dropout | 0.2 |
| Generator | 100-d latent -> 128 -> 256 -> 512 -> d (ReLU, Tanh output) |
| Critic | 512 -> 256 -> 128 -> 1 (LeakyReLU 0.2) |
| Window / EWMA alpha | 32 / 0.3 |
| tau_min | 0.30 |
| ZTA weights (b1, b2, b3) | 0.4, 0.35, 0.25 |
| Seed | 42 |

All values are set in `truheal/config.py`.

## Results
Data: 71 labelled files, 269,708 AETL windows (window 32), train 219,275 / test 50,433
(official file-based split). 10 of the 19 classes are present in this data copy.

| Task | Accuracy | Macro-F1 | ROC-AUC | MCC |
|---|---|---|---|---|
| Binary | 0.9986 | 0.9850 | 0.9994 | 0.9703 |
| 6-class | 0.9428 | 0.9040 | 0.9963 | 0.8973 |
| 19-class (10 present) | 0.9339 | 0.8978 | 0.9956 | 0.9149 |

Full metric tables are produced in `work/results/`.

## Notes
- Trust and reputation are computed per AETL window, not per entity pair with peer aggregation
  (CICIoMT2024 has no entity-interaction graph).
- The 19-to-6 class mapping is inferred. Metrics are macro-averaged over classes present in the test split.
- With default settings the reputation score is close to 1.0 for all classes, so the Zero-Trust
  layer allows nearly all traffic. Set `REP_ANCHOR_WEIGHT` (e.g. 1.0) in `config.py` to supervise reputation.
- `run_synthetic.py` is a noise-perturbed replay of test windows, not independent data.
- `run_ns3.py` is an integration demo. The simulator produces TCP/UDP traffic only, so
  ICMP/ARP/recon/MQTT scenario labels are nominal and accuracy is low (domain shift).

## Citation
BibTeX will be added once the paper is published.

## License
MIT
