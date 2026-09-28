# TRuHEAL-IoMT

**Code implementation by [Vaishali Meena](https://github.com/vaishalimeena) and [Ishan Gupta](https://github.com/IshanG-ai)** (@IshanG-ai) 


PyTorch implementation of TRuHEAL, from *"Proactive Trust-Driven Intrusion Detection in IoMT-Based
Smart Healthcare Ecosystem"* (V. Meena, G. Indra, A. K. Das, Y. Park), evaluated on CICIoMT2024.

Pipeline: AETL (EWMA + window moments) -> Type-3 fuzzy trust (z-slice, Karnik-Mendel) ->
WGAN-GP reputation -> adaptive fusion -> Zero-Trust decision (Allow / Challenge / Deny).

## Authorship
- **Code, experiments, ablations, synthetic evaluation and ns-3 pipeline:** Vaishali Meena and Ishan Gupta
- **Method and paper:** V. Meena, G. Indra, A. K. Das, Y. Park

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

## Results
Data: 71 labelled files, 269,708 AETL windows (window 32), train 219,275 / test 50,433
(official file-based split). 10 of the 19 classes are present in this data copy.

| Task | Accuracy | Macro-F1 | ROC-AUC | MCC |
|---|---|---|---|---|
| Binary | 0.9986 | 0.9850 | 0.9994 | 0.9703 |
| 6-class | 0.9428 | 0.9040 | 0.9963 | 0.8973 |
| 19-class (10 present) | 0.9339 | 0.8978 | 0.9956 | 0.9149 |

Component ablation (binary, 8 epochs):

| Variant | Accuracy | F1 | FPR |
|---|---|---|---|
| AETL only | 0.9971 | 0.9985 | 0.0000 |
| AETL + T3FL | 0.9970 | 0.9985 | 0.0000 |
| AETL + WGAN-GP | 0.9971 | 0.9985 | 0.0000 |
| AETL + Fusion (no trust) | 0.9970 | 0.9984 | 0.0009 |
| Full TRuHEAL | 0.9971 | 0.9985 | 0.0009 |

Reputation learner (binary, accuracy): Isolation Forest 0.265, AE 0.046, VAE 0.047,
Vanilla GAN 0.995, WGAN 0.995, WGAN-GP 0.997.

Full metric tables are produced in `work/results/`.

## Deviations from the paper
- Trust and reputation are computed **per AETL window**, not per entity pair with peer aggregation
  (CICIoMT2024 has no entity-interaction graph).
- Constants not fixed in the paper: window 32, EWMA alpha 0.3, 4 z-slices, tau_min 0.30,
  ZTA weights (0.4, 0.35, 0.25). See `truheal/config.py`.
- The 19-to-6 class mapping is inferred. Metrics are macro-averaged over classes present in the test split.

## Known limitations
- With default settings the reputation score is close to 1.0 for all classes, so the Zero-Trust
  layer allows nearly all traffic. Set `REP_ANCHOR_WEIGHT` (e.g. 1.0) in `config.py` to supervise reputation.
- Component ablation differences are within run-to-run noise on the binary task.
- `run_synthetic.py` is a noise-perturbed replay of test windows, not independent data.
- `run_ns3.py` is an integration demo. The simulator produces TCP/UDP traffic only, so
  ICMP/ARP/recon/MQTT scenario labels are nominal and accuracy is low (domain shift).

## Citation
BibTeX will be added once the paper is published.

## License
MIT
