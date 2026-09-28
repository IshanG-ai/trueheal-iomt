import csv
import glob
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numba import njit

from .config import (CACHE_DIR, CLASSES, CSV_FEATURE_NAMES, DATA_ROOT, DEVICE, EMBED_DIM,
                     EPS, EWMA_ALPHA, GiB, LABEL19_ID, LABEL6_ID, WINDOW_SIZE, to_6class)


# ---------------- AETL: EWMA normalisation + window embedding (Eq.1-3) ----------------
@njit(cache=True, fastmath=True)
def ewma_normalize(x, alpha, eps):
    n, d = x.shape
    out = np.empty_like(x)
    mu = x[0].copy()
    var = np.ones(d, dtype=x.dtype)
    for t in range(n):
        for j in range(d):
            v = x[t, j]
            mu[j] = alpha * v + (1 - alpha) * mu[j]
            var[j] = alpha * (v - mu[j]) ** 2 + (1 - alpha) * var[j]
            out[t, j] = (v - mu[j]) / math.sqrt(var[j] + eps)
    return out


@njit(cache=True, fastmath=True)
def window_embed(x_norm, window):
    n, d = x_norm.shape
    n_windows = n // window
    out = np.empty((n_windows, d * 3), dtype=x_norm.dtype)
    for w in range(n_windows):
        block = x_norm[w * window:(w + 1) * window]
        for j in range(d):
            col = block[:, j]
            m = col.mean()
            v = ((col - m) ** 2).mean()
            s = ((col - m) ** 3).mean() / (v ** 1.5) if v > 1e-12 else 0.0
            out[w, j] = m
            out[w, d + j] = v
            out[w, 2 * d + j] = s
    return out


# ---------------- Filename -> class / split ----------------
def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


CLASS_LOOKUP = {_norm(n): i for i, n in enumerate(CLASSES)}
ALIAS_LOOKUP = {
    _norm("MQTT_DDoS_Connect_Flood"): LABEL19_ID["MQTT"],
    _norm("MQTT_DDoS_Publish_Flood"): LABEL19_ID["MQTT"],
    _norm("MQTT_DoS_Connect_Flood"): LABEL19_ID["MQTT"],
    _norm("MQTT_DoS_Publish_Flood"): LABEL19_ID["MQTT"],
    _norm("MQTT_Malformed_Data"): LABEL19_ID["MQTT"],
    _norm("Recon_Port_Scan"): LABEL19_ID["Port_Scan"],
    _norm("Recon_OS_Scan"): LABEL19_ID["Recon_Vuln"],
    _norm("Recon_VulScan"): LABEL19_ID["Recon_Vuln"],
    _norm("Recon_Ping_Sweep"): LABEL19_ID["Recon"],
    _norm("ARP_Spoofing"): LABEL19_ID["ARP_Spoofing"],
    _norm("Benign"): LABEL19_ID["Benign"],
}


def _strip_csv_suffix(fname):
    return re.sub(r"(\.pcap)?\.csv$", "", Path(fname).name, flags=re.IGNORECASE)


def identify_class(fname):
    stem = re.sub(r"_(?:train|test)$", "", _strip_csv_suffix(fname), flags=re.IGNORECASE)
    stem = re.sub(r"^TCP[_-]IP[_-]", "", stem, flags=re.IGNORECASE)
    stem_nd = re.sub(r"\d+$", "", stem)
    for cand in (stem, stem_nd):
        key = _norm(cand)
        if key in CLASS_LOOKUP:
            return CLASS_LOOKUP[key]
        if key in ALIAS_LOOKUP:
            return ALIAS_LOOKUP[key]
    m = re.fullmatch(r"(DDoS|DoS)[_-](TCP|UDP|SYN|ICMP)(?:[_-]?Flood)?", stem_nd, flags=re.IGNORECASE)
    if m:
        attack, proto = m.group(1).upper(), m.group(2).upper()
        if proto == "ICMP":
            return LABEL19_ID["ICMP_Flood"]
        if attack == "DDoS":
            return LABEL19_ID["DDoS"]
        return LABEL19_ID[f"DoS_{proto}"]
    return None


def identify_split(fname):
    name = _strip_csv_suffix(fname).lower()
    if name.endswith("_train"):
        return "train"
    if name.endswith("_test"):
        return "test"
    return None


def build_manifest():
    rows = []
    for fp in glob.glob(str(DATA_ROOT / "**" / "*.csv"), recursive=True):
        with open(fp, "r", encoding="utf-8-sig", newline="") as f:
            header = next(csv.reader(f))
        if header != CSV_FEATURE_NAMES:
            continue
        cid, split = identify_class(fp), identify_split(fp)
        if cid is None or split is None:
            continue
        rows.append({"path": fp, "name": Path(fp).name, "class_id": cid,
                     "class_name": CLASSES[cid], "split": split})
    manifest = pd.DataFrame(rows).drop_duplicates("name").reset_index(drop=True)
    if manifest.empty:
        raise RuntimeError(f"No usable CICIoMT2024 CSVs found under {DATA_ROOT}. "
                           f"Set IOMT_DATA_DIR to the dataset folder.")
    present = set(manifest["class_name"])
    print(f"Eligible files: {len(manifest)} | classes present: {len(present)}/{len(CLASSES)}")
    print("Missing classes:", [c for c in CLASSES if c not in present])
    print(manifest.groupby(["class_name", "split"]).size().unstack(fill_value=0))
    return manifest


def embed_files(manifest):
    cache = CACHE_DIR / "embeddings"
    cache.mkdir(exist_ok=True)
    shards = []
    for row in manifest.to_dict("records"):
        key = hashlib.sha256(
            f"{row['path']}|{Path(row['path']).stat().st_size}|w{WINDOW_SIZE}|a{EWMA_ALPHA}|v1".encode()
        ).hexdigest()[:16]
        out = cache / f"{key}.npz"
        if not out.exists():
            arr = pd.read_csv(row["path"], encoding="utf-8-sig", dtype=np.float32)[CSV_FEATURE_NAMES]
            arr = np.nan_to_num(arr.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            if len(arr) < WINDOW_SIZE:
                z = np.empty((0, EMBED_DIM), dtype=np.float32)
            else:
                z = window_embed(ewma_normalize(arr, EWMA_ALPHA, EPS), WINDOW_SIZE).astype(np.float32)
            np.savez(out, z=z)
            print("EMBEDDED:", row["name"], "->", z.shape[0], "windows", flush=True)
        shards.append({**row, "npz": str(out)})
    return shards


def build_arrays(shards):
    arr_dir = CACHE_DIR / "arrays"
    arr_dir.mkdir(exist_ok=True)
    zp, yp, sp = arr_dir / "Z.npy", arr_dir / "y19.npy", arr_dir / "split.npy"
    if not (zp.exists() and yp.exists() and sp.exists()):
        sizes = [np.load(r["npz"])["z"].shape[0] for r in shards]
        total = int(sum(sizes))
        Z = np.lib.format.open_memmap(zp, mode="w+", dtype=np.float32, shape=(total, EMBED_DIM))
        y19 = np.empty(total, dtype=np.int64)
        split = np.empty(total, dtype=np.uint8)
        pos = 0
        for r, n in zip(shards, sizes):
            Z[pos:pos + n] = np.load(r["npz"])["z"]
            y19[pos:pos + n] = r["class_id"]
            split[pos:pos + n] = int(r["split"] == "test")
            pos += n
        Z.flush()
        np.save(yp, y19)
        np.save(sp, split)
    return np.load(zp, mmap_mode="r"), np.load(yp), np.load(sp)


def fit_scaler(Z, train_idx, chunk=200_000):
    path = CACHE_DIR / "arrays" / "scaler.json"
    if path.exists():
        return json.loads(path.read_text())
    n_seen, mean, m2 = 0, np.zeros(EMBED_DIM), np.zeros(EMBED_DIM)
    for i in range(0, len(train_idx), chunk):
        block = np.nan_to_num(np.array(Z[train_idx[i:i + chunk]], dtype=np.float64))
        bn, bm = len(block), block.mean(0)
        block -= bm
        bm2 = np.einsum("ij,ij->j", block, block)
        total = n_seen + bn
        delta = bm - mean
        m2 += bm2 + delta ** 2 * n_seen * bn / total
        mean += delta * bn / total
        n_seen = total
    std = np.sqrt(m2 / n_seen)
    std[std < 1e-8] = 1.0
    scaler = {"mean": mean.tolist(), "std": std.tolist()}
    path.write_text(json.dumps(scaler))
    return scaler


class IoMTData:
    """Holds embeddings, labels, split indices and (optionally) a GPU feature cache."""

    def __init__(self):
        self.manifest = build_manifest()
        shards = embed_files(self.manifest)
        self.Z, y19, split = build_arrays(shards)
        self.total = len(y19)
        y6 = np.array([LABEL6_ID[to_6class(CLASSES[c])] for c in y19], dtype=np.int64)
        yb = (y19 != LABEL19_ID["Benign"]).astype(np.int64)

        train_idx = np.flatnonzero(split == 0)
        test_idx = np.flatnonzero(split == 1)
        print(f"Windows total={self.total} | train={len(train_idx)} | test={len(test_idx)}")

        self.scaler = fit_scaler(self.Z, train_idx)
        self.mean = torch.tensor(self.scaler["mean"], dtype=torch.float32, device=DEVICE)
        self.std = torch.tensor(self.scaler["std"], dtype=torch.float32, device=DEVICE)

        self.X_gpu = None
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            if free - self.total * EMBED_DIM * 4 > 4 * GiB:
                self.X_gpu = torch.empty((self.total, EMBED_DIM), dtype=torch.float32, device=DEVICE)
                for i in range(0, self.total, 500_000):
                    j = min(i + 500_000, self.total)
                    self.X_gpu[i:j] = self._load(np.arange(i, j))
                print("Feature cache on GPU:", tuple(self.X_gpu.shape))
        if self.X_gpu is None:
            print("Using disk-backed per-batch loading.")

        self.y = {k: torch.from_numpy(v).to(DEVICE) for k, v in
                  {"binary": yb, "six": y6, "nineteen": y19}.items()}
        self.train_ids = torch.from_numpy(train_idx).to(DEVICE)
        self.test_ids = torch.from_numpy(test_idx).to(DEVICE)
        self.present_class_ids = sorted(self.manifest["class_id"].unique().tolist())

    def _load(self, indices):
        block = np.nan_to_num(np.array(self.Z[indices], dtype=np.float32))
        return (torch.from_numpy(block).to(DEVICE) - self.mean) / self.std

    def get_features(self, ids):
        if self.X_gpu is not None:
            return self.X_gpu.index_select(0, ids)
        return self._load(ids.cpu().numpy())
