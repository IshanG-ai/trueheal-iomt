"""Train + evaluate TRuHEAL on binary / 6-class / 19-class (paper Tables III-VII)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from truheal.config import DEVICE, EPOCHS_FULL, RESULT_DIR, set_seed
from truheal.data import IoMTData
from truheal.engine import evaluate_truheal, train_truheal


def main():
    set_seed()
    print("Device:", DEVICE)
    data = IoMTData()
    summary = {}
    for task, out_name in [("binary", "binary"), ("six", "six_class"), ("nineteen", "nineteen_class")]:
        model = train_truheal(data, task, "t3fl", True, True, epochs=EPOCHS_FULL, tag="full")
        metrics, df = evaluate_truheal(data, model, task, RESULT_DIR / out_name)
        summary[task] = {"overall": metrics, "macro_over_present_classes": df.loc["Average (present)"].to_dict()}
        plt.figure(figsize=(5, 3))
        plt.plot([h["loss"] for h in model.history])
        plt.title(f"Training loss - {task}")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.tight_layout()
        plt.savefig(RESULT_DIR / f"convergence_{task}.png", dpi=150)
        plt.close()
    (RESULT_DIR / "final_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("Saved to", RESULT_DIR)


if __name__ == "__main__":
    main()
