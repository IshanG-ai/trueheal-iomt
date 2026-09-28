"""Build ns-3.40, simulate 19 IoMT-like scenarios, extract flow features, classify with trained models.

Integration demo only: the simulator emits TCP/UDP OnOff traffic, so ICMP/ARP/recon/MQTT labels are nominal.
"""
import argparse
import glob
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from truheal.config import (DEVICE, EMBED_DIM, EPOCHS_FULL, EPS, EWMA_ALPHA, N19, RESULT_DIR,
                            TASK_LABELS, TASKS, WINDOW_SIZE, set_seed, to_6class)
from truheal.data import IoMTData, ewma_normalize, window_embed
from truheal.engine import train_truheal

SCENARIOS = {
    "benign": "Benign", "arp_spoofing": "ARP_Spoofing", "recon_pingsweep": "Recon",
    "recon_vulnscan": "Recon_Vuln", "recon_osscan": "Recon_Vuln", "recon_portscan": "Port_Scan",
    "mqtt_malformed": "MQTT", "mqtt_dos_connect": "MQTT", "mqtt_ddos_connect": "MQTT",
    "mqtt_dos_publish": "MQTT", "mqtt_ddos_publish": "MQTT",
    "dos_tcp": "DoS_TCP", "dos_icmp": "ICMP_Flood", "dos_syn": "DoS_SYN", "dos_udp": "DoS_UDP",
    "ddos_tcp": "DoS_TCP", "ddos_icmp": "ICMP_Flood", "ddos_syn": "DoS_SYN", "ddos_udp": "DoS_UDP",
}


def sh(cmd, cwd=None):
    subprocess.run(cmd, shell=True, cwd=cwd, check=True)


def build_and_simulate(ns3_dir, pcap_dir, duration, do_build):
    ns3_dir = Path(ns3_dir)
    if do_build:
        if not (ns3_dir / "ns3").exists():
            sh(f"git clone --depth 1 --branch ns-3.40 https://gitlab.com/nsnam/ns-3-dev.git {ns3_dir}")
        sh("./ns3 configure --disable-examples --disable-tests --disable-werror", cwd=ns3_dir)
        (ns3_dir / "scratch" / "iomt.cc").write_text((ROOT / "ns3" / "iomt.cc").read_text())
        sh("./ns3 build scratch/iomt", cwd=ns3_dir)     # first build takes a long time
    pcap_dir.mkdir(parents=True, exist_ok=True)
    for sc in SCENARIOS:
        r = subprocess.run(f'./ns3 run "scratch/iomt --scenario={sc} --pcapDir={pcap_dir} --duration={duration}"',
                           shell=True, cwd=ns3_dir, capture_output=True, text=True)
        print(f"[{'OK' if r.returncode == 0 else 'FAIL'}] {sc}")
        if r.returncode:
            print("  ", r.stderr[-300:])
    print(len(glob.glob(str(pcap_dir / "*.pcap"))), "pcap files")


def pcap_to_X(path, min_dur=1e-3):
    from scapy.all import ARP, IP, TCP, UDP, PcapReader
    flows = defaultdict(lambda: dict(ts=[], lens=[], flags=[], proto=0, sport=0, dport=0, arp=False))
    with PcapReader(str(path)) as rd:
        for pkt in rd:
            if ARP in pkt:
                a = pkt[ARP]
                f = flows[("arp", a.psrc, a.pdst)]
                f["ts"].append(float(pkt.time)); f["lens"].append(len(pkt)); f["flags"].append(0); f["arp"] = True
                continue
            if IP not in pkt:
                continue
            ip, sp, dp, fl = pkt[IP], 0, 0, 0
            if TCP in pkt:
                sp, dp, fl = pkt[TCP].sport, pkt[TCP].dport, int(pkt[TCP].flags)
            elif UDP in pkt:
                sp, dp = pkt[UDP].sport, pkt[UDP].dport
            f = flows[(ip.src, ip.dst, ip.proto, sp, dp)]
            f["ts"].append(float(pkt.time)); f["lens"].append(len(pkt)); f["flags"].append(fl)
            f["proto"], f["sport"], f["dport"] = ip.proto, sp, dp

    rows = []
    for fd in flows.values():
        if len(fd["ts"]) < 2:
            continue
        ts, ls = np.array(fd["ts"]), np.array(fd["lens"], dtype=np.float64)
        fl, proto, ports = fd["flags"], fd["proto"], {fd["sport"], fd["dport"]}
        n, dur = len(ts), max(ts[-1] - ts[0], min_dur)
        f = np.zeros(45, dtype=np.float32)          # indices follow CSV_FEATURE_NAMES
        f[0], f[1], f[2] = ls.mean(), proto, ts[-1] - ts[0]
        f[3] = f[4] = n / dur
        for idx, bit in [(6, 0x01), (7, 0x02), (8, 0x04), (9, 0x08), (10, 0x10)]:
            f[idx] = sum(1 for x in fl if x & bit)
        f[13], f[14], f[15], f[16] = f[10], f[7], f[6], f[8]
        f[17], f[18], f[19], f[20], f[22] = (float(p in ports) for p in (80, 443, 53, 23, 22))
        f[24], f[25], f[27], f[28], f[30] = float(proto == 6), float(proto == 17), float(fd["arp"]), float(proto == 1), 1.0
        f[32], f[33], f[34], f[35], f[36], f[37] = ls.sum(), ls.min(), ls.max(), ls.mean(), ls.std(), ls.sum()
        f[38], f[39], f[40] = np.diff(ts).mean(), n, np.sqrt((ls ** 2).sum())
        f[41], f[43], f[44] = np.sqrt(ls.var()), ls.var(), float(n)
        rows.append(f)
    return np.stack(rows) if rows else None


@torch.inference_mode()
def classify(data, model, X_raw, task):
    n_cls, names = TASKS[task], TASK_LABELS[task]
    if X_raw is None or len(X_raw) == 0:
        return "NoData"
    normed = ewma_normalize(np.nan_to_num(X_raw.astype(np.float64)), EWMA_ALPHA, EPS)
    if len(normed) < WINDOW_SIZE:
        normed = np.vstack([normed, np.tile(normed[-1:], (WINDOW_SIZE - len(normed), 1))])
    Z = np.nan_to_num(window_embed(normed, WINDOW_SIZE).astype(np.float32))
    sig = Z.std(0); sig[sig < 1e-8] = 1.0
    Z = (Z - Z.mean(0)) / sig                                   # per-scenario standardisation ...
    Z = Z * np.array(data.scaler["std"], np.float32) + np.array(data.scaler["mean"], np.float32)  # ... mapped to train scale
    Z = np.clip(np.nan_to_num(Z), -10, 10)
    # model expects train-standardised features
    x = (torch.from_numpy(Z).to(DEVICE) - data.mean) / data.std
    model.eval()
    lg = model(x, stochastic=False)["logits"]
    preds = (lg.sigmoid() >= 0.5).long().cpu().numpy() if n_cls == 2 else lg.argmax(-1).cpu().numpy()
    return names[int(np.bincount(preds, minlength=n_cls).argmax())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns3-dir", default="./ns-3.40")
    ap.add_argument("--pcap-dir", default="./ns3_pcaps")
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--build", action="store_true", help="clone + configure + build ns-3.40")
    ap.add_argument("--skip-sim", action="store_true", help="reuse existing pcaps")
    a = ap.parse_args()
    pcap_dir = Path(a.pcap_dir).resolve()

    if not a.skip_sim:
        build_and_simulate(a.ns3_dir, pcap_dir, a.duration, a.build)

    set_seed()
    data = IoMTData()
    models = {t: train_truheal(data, t, epochs=EPOCHS_FULL, tag="full") for t in TASKS}

    rows = []
    for sc, true19 in SCENARIOS.items():
        parts = [pcap_to_X(p) for p in glob.glob(str(pcap_dir / f"{sc}-*.pcap"))]
        parts = [p for p in parts if p is not None]
        X = np.vstack(parts) if parts else None
        pb, p6, p19 = (classify(data, models[t], X, t) for t in ("binary", "six", "nineteen"))
        true6, trueb = to_6class(true19), ("Benign" if true19 == "Benign" else "Attack")
        rows.append({"Scenario": sc, "True-19": true19, "Pred-Bin": pb, "Bin_ok": pb == trueb,
                     "True-6": true6, "Pred-6": p6, "6_ok": p6 == true6,
                     "Pred-19": p19, "19_ok": p19 == true19})
        print(f"{sc:20s} bin={pb:8s} 6={p6:9s} 19={p19}")
    df = pd.DataFrame(rows)
    print(f"\nBinary {df.Bin_ok.mean():.1%} | 6-class {df['6_ok'].mean():.1%} | 19-class {df['19_ok'].mean():.1%}")
    df.to_csv(RESULT_DIR / "ns3_results.csv", index=False)


if __name__ == "__main__":
    main()
