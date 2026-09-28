import os
import random
from pathlib import Path

import numpy as np
import torch

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GiB = 1024 ** 3


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------- Paths (override with environment variables) ----------------
DATA_ROOT = Path(os.environ.get("IOMT_DATA_DIR", "/kaggle/input/datasets/ishangupta0711/iomt2024"))
WORK_DIR = Path(os.environ.get("TRUHEAL_WORK_DIR", "./work"))
CACHE_DIR = WORK_DIR / "cache"
CKPT_DIR = WORK_DIR / "ckpt"
RESULT_DIR = WORK_DIR / "results"
for _d in (WORK_DIR, CACHE_DIR, CKPT_DIR, RESULT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------- Hyperparameters ----------------
# Not fixed numerically in the paper (engineering choices):
WINDOW_SIZE = 32          # AETL window, Eq.(1)
EWMA_ALPHA = 0.3          # Eq.(2)-(3)
N_SLICES = 4              # Type-3 z-slices: 0.25/0.5/0.75/1.0
TAU_MIN = 0.30            # Zero-Trust minimum threshold
ZTA_WEIGHTS = (0.4, 0.35, 0.25)   # Eq.(27) weights

EPS = 1e-8
NOISE_DIM = 16
BATCH_SIZE = 4096
EPOCHS_FULL = 100
EPOCHS_QUICK = 8
GAN_ABL_EPOCHS = 20
N_CRITIC = 3
LAMBDA_GP = 10.0          # Eq.(21)
LR_G, LR_D, LR_CLS = 1e-4, 1e-4, 1e-3
# 0.0 = faithful to original notebook (reputation trained only via adversarial + cls gradients).
# >0 adds MSE(reputation, benign/attack target) so reputation becomes discriminative.
REP_ANCHOR_WEIGHT = 0.0

# ---------------- CICIoMT2024 features / taxonomy ----------------
CSV_FEATURE_NAMES = [
    "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count",
    "fin_count", "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
    "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC", "Tot sum", "Min", "Max",
    "AVG", "Std", "Tot size", "IAT", "Number", "Magnitue", "Radius", "Covariance",
    "Variance", "Weight",
]
FEATURE_DIM = len(CSV_FEATURE_NAMES)   # 45
EMBED_DIM = FEATURE_DIM * 3            # mean + var + skew per window

CLASSES = [
    "Benign", "DDoS", "DoS", "Spoofing", "Recon", "MQTT",
    "ARP_Spoofing", "SQL_Injection", "XSS", "RCE", "Backdoor",
    "DoS_UDP", "Port_Scan", "Recon_Vuln", "ICMP_Flood", "DoS_SYN",
    "DNS_Spoofing", "DoS_TCP", "HTTP_Flood",
]
LABEL19_ID = {c: i for i, c in enumerate(CLASSES)}
N19 = len(CLASSES)

LABEL6 = ["Benign", "DDoS", "DoS", "Spoofing", "Recon", "MQTT"]
LABEL6_ID = {c: i for i, c in enumerate(LABEL6)}

# Inferred (not stated in the paper): 19 fine labels -> 6 parent families.
SIX_CLASS_PARENT = {
    "Benign": "Benign", "DDoS": "DDoS", "DoS": "DoS",
    "Spoofing": "Spoofing", "ARP_Spoofing": "Spoofing", "DNS_Spoofing": "Spoofing",
    "Recon": "Recon", "Port_Scan": "Recon", "Recon_Vuln": "Recon",
    "MQTT": "MQTT",
    "SQL_Injection": "DoS", "XSS": "DoS", "RCE": "DoS", "Backdoor": "DoS",
    "DoS_UDP": "DoS", "DoS_SYN": "DoS", "DoS_TCP": "DoS",
    "ICMP_Flood": "DDoS", "HTTP_Flood": "DoS",
}
assert set(SIX_CLASS_PARENT) == set(CLASSES)


def to_6class(name):
    return SIX_CLASS_PARENT[name]


TASKS = {"binary": 2, "six": 6, "nineteen": N19}
TASK_LABELS = {"binary": ["Benign", "Attack"], "six": LABEL6, "nineteen": CLASSES}
