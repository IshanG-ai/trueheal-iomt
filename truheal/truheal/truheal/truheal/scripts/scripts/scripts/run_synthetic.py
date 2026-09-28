"""Noise-perturbed replay evaluation (sanity check, NOT independent data).

Seeds are TEST windows; each is jittered with Gaussian noise (std NOISE_STD) and re-classified.
Also reports per-class mean trust / reputation / fused score.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from truheal.config import (CLASSES, DEVICE, EPOCHS_FULL, LABEL19_ID, LABEL6, LABEL6_ID, N19,
                            RESULT_DIR, set_seed, to_6class)
from truheal.data import IoMTData
from truheal.engine import train_truheal

N_PER_CLASS = 2000
NOISE_STD = 0.05


@torch.inference_mode()
def generate(data, model, class_ids):
    model.eval()
    X, y = [], []
    for cid in class_ids:
        seeds = data.test_ids[data.y["nineteen"][data.test_ids] == cid]
        if len(seeds) == 0:
            continue
        chosen = seeds[torch.randint(len(seeds), (N_PER_CLASS,), device=seeds.device)]
        x = data.get_features(chosen)
        out = model(x, stochastic=True)
        print(f"  {CLASSES[cid]:14s} trust={out['trust'].mean():.3f} rep={out['reputation'].mean():.3f} "
              f"fused={out['fused'].mean():.3f}")
        X.append((x + NOISE_STD * torch.randn_like(x)).cpu().numpy())
        y.append(np.full(N_PER_CLASS, cid, np.int64))
    return np.vstack(X), np.concatenate(y)


@torch.inference_mode()
def predict(model, X, n_cls):
    model.eval()
    out = []
    for l in range(0, len(X), 4096):
        lg = model(torch.from_numpy(X[l:l + 4096]).float().to(DEVICE), stochastic=False)["logits"]
        out.append((lg.sigmoid() >= 0.5).long().cpu().numpy() if n_cls == 2 else lg.argmax(-1).cpu().numpy())
    return np.concatenate(out)


def main():
    set_seed()
    data = IoMTData()
    m_bin = train_truheal(data, "binary", epochs=EPOCHS_FULL, tag="full")
    m_6 = train_truheal(data, "six", epochs=EPOCHS_FULL, tag="full")
    m_19 = train_truheal(data, "nineteen", epochs=EPOCHS_FULL, tag="full")

    present = data.present_class_ids
    X, y19 = generate(data, m_19, present)
    y6 = np.array([LABEL6_ID[to_6class(CLASSES[c])] for c in y19])
    yb = (y19 != LABEL19_ID["Benign"]).astype(int)

    pb, p6, p19 = predict(m_bin, X, 2), predict(m_6, X, 6), predict(m_19, X, N19)
    res = {
        "binary": {"accuracy": accuracy_score(yb, pb), "macro_f1": f1_score(yb, pb, average="macro")},
        "six": {"accuracy": accuracy_score(y6, p6), "macro_f1": f1_score(y6, p6, average="macro")},
        "nineteen": {"accuracy": accuracy_score(y19, p19),
                     "macro_f1": f1_score(y19, p19, average="macro", labels=present)},
    }
    print(json.dumps(res, indent=2))
    print(classification_report(y19, p19, labels=present, target_names=[CLASSES[i] for i in present], zero_division=0))
    (RESULT_DIR / "synthetic_eval.json").write_text(json.dumps(res, indent=2))

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    tasks = list(res)
    xs = np.arange(3)
    ax[0].bar(xs - 0.175, [res[t]["accuracy"] for t in tasks], 0.35, label="Accuracy")
    ax[0].bar(xs + 0.175, [res[t]["macro_f1"] for t in tasks], 0.35, label="Macro-F1")
    ax[0].set_xticks(xs); ax[0].set_xticklabels(["Binary", "6-Class", "19-Class"]); ax[0].set_ylim(0, 1.1)
    ax[0].legend(); ax[0].set_title("Noise-perturbed replay")
    cm = confusion_matrix(y6, p6, labels=list(range(6))).astype(float)
    sns.heatmap(cm / cm.sum(1, keepdims=True).clip(1), annot=True, fmt=".2f", cmap="Blues",
                xticklabels=LABEL6, yticklabels=LABEL6, ax=ax[1], cbar=False)
    ax[1].set_title("6-class confusion (row-normalised)")
    f1 = f1_score(y19, p19, average=None, labels=present, zero_division=0)
    ax[2].barh([CLASSES[i] for i in present], f1); ax[2].set_xlim(0, 1.05); ax[2].set_title("Per-class F1")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "synthetic_evaluation.png", dpi=200)


if __name__ == "__main__":
    main()
