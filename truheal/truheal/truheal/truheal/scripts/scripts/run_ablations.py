"""Ablations: component (Table IX), trust variant (X), noise robustness (XI), reputation learner (XII)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from torch import nn
from torch.nn import functional as F

from truheal.config import (BATCH_SIZE, DEVICE, EMBED_DIM, EPOCHS_QUICK, GAN_ABL_EPOCHS, LAMBDA_GP,
                            NOISE_DIM, RESULT_DIR, SEED, TASKS, set_seed)
from truheal.data import IoMTData
from truheal.engine import evaluate_truheal, make_balanced_perm, measure_latency, train_truheal
from truheal.model import Critic, Generator, gradient_penalty


def component_ablation(data):
    variants = [
        ("AETL only (baseline)", dict(use_trust=False, use_reputation=False, use_fusion=False)),
        ("AETL + T3FL (trust only)", dict(use_trust=True, use_reputation=False, use_fusion=False)),
        ("AETL + WGAN-GP (reputation only)", dict(use_trust=False, use_reputation=True, use_fusion=False)),
        ("AETL + Fusion (no trust)", dict(use_trust=False, use_reputation=True, use_fusion=True)),
        ("Full TRuHEAL (all)", dict(use_trust=True, use_reputation=True, use_fusion=True)),
    ]
    rows = []
    for name, kw in variants:
        slug = name.replace(" ", "_")
        m = train_truheal(data, "binary", "t3fl", epochs=EPOCHS_QUICK, tag=slug, **kw)
        met, _ = evaluate_truheal(data, m, "binary", RESULT_DIR / f"ablation_component/{slug}")
        rows.append({"Model Variant": name, "Accuracy": met["Accuracy"], "F1-score": met["F1-Score"],
                     "FPR": met["FPR"], "Latency (ms/window)": round(measure_latency(data, m), 5)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "table_ix_component_ablation.csv", index=False)
    print(df)


def trust_ablation(data):
    rows = []
    for v in ["type1", "it2fl", "t3fl"]:
        m = train_truheal(data, "binary", v, True, True, epochs=EPOCHS_QUICK, tag=f"trustabl_{v}")
        met, _ = evaluate_truheal(data, m, "binary", RESULT_DIR / f"ablation_trust/{v}")
        rows.append({"Trust Module": v, **{k: met[k] for k in ["Accuracy", "F1-Score", "ROC_AUC", "MCC"]}})
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "table_x_trust_ablation.csv", index=False)
    print(df)


@torch.inference_mode()
def robustness(data, model, task="six", noise_std=0.15):
    n_classes, ids = TASKS[task], data.test_ids
    truth = data.y[task][ids].cpu().numpy()
    x_clean = data.get_features(ids)
    x_noisy = x_clean + noise_std * torch.randn_like(x_clean)
    model.eval()

    def run(x):
        preds, trust = [], []
        for l in range(0, len(x), 32768):
            out = model(x[l:l + 32768], stochastic=False)
            preds.append(out["logits"].argmax(-1).cpu().numpy())
            trust.append(out["trust"].cpu().numpy())
        return np.concatenate(preds), np.concatenate(trust)

    pc, tc = run(x_clean)
    pn, tn_ = run(x_noisy)
    cm = confusion_matrix(truth, pn, labels=list(range(n_classes)))
    present = [c for c in range(n_classes) if (truth == c).any()]
    fpr, fnr = [], []
    for c in present:
        tp = cm[c, c]; fn = cm[c].sum() - tp; fp = cm[:, c].sum() - tp; tn = cm.sum() - tp - fn - fp
        fpr.append(fp / (fp + tn + 1e-9)); fnr.append(fn / (fn + tp + 1e-9))
    return {"FPR (noisy)": float(np.mean(fpr)), "FNR (noisy)": float(np.mean(fnr)),
            "Accuracy Drop": float((pc == truth).mean() - (pn == truth).mean()),
            "Trust Score Stability": float(1 - np.mean(np.abs(tc - tn_)))}


def robustness_ablation(data):
    rows = []
    for v, label in [("type1", "A1"), ("it2fl", "A2"), ("t3fl", "A3")]:
        m = train_truheal(data, "six", v, True, True, epochs=EPOCHS_QUICK, tag=f"robust_{v}")
        rows.append({"Trust Module": label, **robustness(data, m)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "table_xi_robustness_ablation.csv", index=False)
    print(df)


# ---------------- Table XII: reputation learner baselines ----------------
def _split_np(data, ids):
    with torch.inference_mode():
        return data.get_features(ids).cpu().numpy(), data.y["binary"][ids].cpu().numpy()


def _recon_baseline(data, variational, epochs=EPOCHS_QUICK, latent=16):
    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(EMBED_DIM, 64), nn.ReLU())
            self.mu = nn.Linear(64, latent)
            self.logvar = nn.Linear(64, latent) if variational else None
            self.dec = nn.Sequential(nn.Linear(latent, 64), nn.ReLU(), nn.Linear(64, EMBED_DIM))

        def forward(self, x):
            h = self.enc(x)
            if self.logvar is not None:
                mu, lv = self.mu(h), self.logvar(h)
                return self.dec(mu + torch.exp(0.5 * lv) * torch.randn_like(mu)), mu, lv
            return self.dec(self.mu(h)), None, None

    model = AE().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    benign = data.train_ids[data.y["binary"][data.train_ids] == 0]
    for _ in range(epochs):
        perm = torch.randperm(len(benign), device=DEVICE)
        for l in range(0, len(benign), BATCH_SIZE):
            x = data.get_features(benign[perm[l:l + BATCH_SIZE]])
            out, mu, lv = model(x)
            loss = F.mse_loss(out, x)
            if variational:
                loss = loss - 0.5 * torch.mean(1 + lv - mu.square() - lv.exp()) * 0.001
            opt.zero_grad(); loss.backward(); opt.step()

    def err(ids):
        res = []
        with torch.no_grad():
            for l in range(0, len(ids), 32768):
                x = data.get_features(ids[l:l + 32768])
                res.append(((model(x)[0] - x) ** 2).mean(1).cpu().numpy())
        return np.concatenate(res)

    thr = np.percentile(err(benign), 90)      # threshold from TRAIN benign only (no test leakage)
    return (err(data.test_ids) > thr).astype(int)


def _gan_variant(data, loss_type, Yte, epochs=GAN_ABL_EPOCHS):
    gen, disc = Generator(EMBED_DIM).to(DEVICE), Critic(EMBED_DIM).to(DEVICE)
    g_opt = torch.optim.Adam(gen.parameters(), lr=(2e-4 if loss_type == "wgangp" else 5e-5), betas=(0.5, 0.9))
    d_opt = torch.optim.Adam(disc.parameters(), lr=1e-4,
                             betas=(0.0, 0.9) if loss_type in ("wgan", "wgangp") else (0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    n_critic = {"bce": 1, "wgan": 5, "wgangp": 3}[loss_type]
    yb = data.y["binary"]
    best_acc, best_state = -1.0, None

    for ep in range(epochs):
        perm = make_balanced_perm(yb, data.train_ids, 2)
        for l in range(0, len(perm), BATCH_SIZE):
            ids = data.train_ids[perm[l:l + BATCH_SIZE]]
            x, y = data.get_features(ids), yb[ids]
            real = (1.0 - y.float()) * 0.9 + 0.05
            for _ in range(n_critic):
                with torch.no_grad():
                    fake = gen(torch.randn(len(x), NOISE_DIM, device=DEVICE), x)
                dr, df_ = disc(real, x), disc(fake, x)
                if loss_type == "bce":
                    d_loss = bce(dr, torch.ones_like(dr)) + bce(df_, torch.zeros_like(df_))
                elif loss_type == "wgan":
                    d_loss = df_.mean() - dr.mean()
                else:
                    d_loss = df_.mean() - dr.mean() + gradient_penalty(disc, real, fake, x, LAMBDA_GP)
                d_opt.zero_grad(); d_loss.backward()
                nn.utils.clip_grad_norm_(disc.parameters(), 5.0)
                d_opt.step()
                if loss_type == "wgan":
                    for p in disc.parameters():
                        p.data.clamp_(-0.01, 0.01)
            fake = gen(torch.randn(len(x), NOISE_DIM, device=DEVICE), x)
            adv = disc(fake, x)
            if loss_type == "bce":
                g_loss = bce(adv, torch.ones_like(adv))
            elif loss_type == "wgan":
                g_loss = -adv.mean()
            else:   # supervised anchor + bounded adversarial term (prevents sigmoid collapse)
                g_loss = F.mse_loss(fake, real) + 0.02 * torch.tanh(-adv.mean())
            g_opt.zero_grad(); g_loss.backward()
            nn.utils.clip_grad_norm_(gen.parameters(), 5.0)
            g_opt.step()

        if (ep + 1) % 2 == 0 or ep == epochs - 1:     # checkpoint selection on TRAIN probe only
            with torch.no_grad():
                pid = data.train_ids[make_balanced_perm(yb, data.train_ids, 2)[:4096]]
                r = gen(torch.randn(len(pid), NOISE_DIM, device=DEVICE), data.get_features(pid))
                acc = accuracy_score(yb[pid].cpu().numpy(), (r.cpu().numpy() < 0.5).astype(int))
            print(f"  [{loss_type}] epoch {ep + 1}/{epochs} R mean={r.mean():.3f} std={r.std():.3f} probe-acc={acc:.4f}")
            if acc > best_acc:
                best_acc, best_state = acc, {k: v.clone() for k, v in gen.state_dict().items()}

    gen.load_state_dict(best_state)
    with torch.no_grad():
        R = np.concatenate([gen(torch.randn(min(32768, len(data.test_ids) - l), NOISE_DIM, device=DEVICE),
                                data.get_features(data.test_ids[l:l + 32768])).cpu().numpy()
                            for l in range(0, len(data.test_ids), 32768)])
    p = (R < 0.5).astype(int)
    return {"Accuracy": accuracy_score(Yte, p), "F1-score": f1_score(Yte, p), "Recall": recall_score(Yte, p)}


def reputation_ablation(data):
    Xtr, _ = _split_np(data, data.train_ids)
    _, Yte = _split_np(data, data.test_ids)
    Xte, _ = _split_np(data, data.test_ids)
    rows = []
    p = (IsolationForest(random_state=SEED, contamination=0.3).fit(Xtr).predict(Xte) == -1).astype(int)
    rows.append({"Model": "Isolation Forest", "Accuracy": accuracy_score(Yte, p),
                 "F1-score": f1_score(Yte, p), "Recall": recall_score(Yte, p)})
    for name, var in [("Autoencoder", False), ("Variational Autoencoder", True)]:
        p = _recon_baseline(data, var)
        rows.append({"Model": name, "Accuracy": accuracy_score(Yte, p),
                     "F1-score": f1_score(Yte, p), "Recall": recall_score(Yte, p)})
    for name, lt in [("Vanilla GAN (BCE)", "bce"), ("WGAN (no GP)", "wgan"), ("WGAN-GP", "wgangp")]:
        rows.append({"Model": name, **_gan_variant(data, lt, Yte)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "table_xii_gan_ablation.csv", index=False)
    print(df)


if __name__ == "__main__":
    set_seed()
    d = IoMTData()
    component_ablation(d)
    trust_ablation(d)
    robustness_ablation(d)
    reputation_ablation(d)
