import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import N_SLICES, NOISE_DIM, TAU_MIN, ZTA_WEIGHTS

# ---------------- Stage I: Type-3 fuzzy trust (Eq.4-15) ----------------
PAPER_INPUTS = ["packet_irregularity", "protocol_deviation", "latency_variance",
                "retransmission_rate", "entropy_fluctuation", "communication_frequency",
                "telemetry_sparsity", "protocol_switching", "timing_stability",
                "historical_trust", "behavior_consistency"]
N_INPUTS = len(PAPER_INPUTS)

# labels: 0=Low 1=Medium 2=High. Rules follow Table II.
RULES = [
    dict(feats=(0, 1), labels=(0, 0), out=(0.80, 1.00)),   # R1 High
    dict(feats=(0, 2), labels=(1, 1), out=(0.40, 0.60)),   # R2 Medium
    dict(feats=(0, 1), labels=(2, 1), out=(0.10, 0.30)),   # R3 Low
    dict(feats=(3, 4), labels=(2, 2), out=(0.00, 0.15)),   # R4 Very Low
    dict(feats=(5, 6), labels=(0, 2), out=(0.30, 0.70)),   # R5 Uncertain
    dict(feats=(7, 8), labels=(2, 0), out=(0.10, 0.30)),   # R6 Low
    dict(feats=(9, 10), labels=(2, 2), out=(0.80, 1.00)),  # R7 High
]
LABEL_CENTERS = torch.tensor([0.10, 0.50, 0.90])


def km_type_reduce(w_low, w_high, consequents):
    """Closed-form Karnik-Mendel type reduction over sorted consequents."""
    order = torch.argsort(consequents)
    c, lo, hi = consequents[order], w_low[..., order], w_high[..., order]
    delta = hi - lo
    pw = F.pad(delta.cumsum(-1), (1, 0))
    pv = F.pad((delta * c).cumsum(-1), (1, 0))
    den_l = lo.sum(-1, keepdim=True) + pw
    num_l = (lo * c).sum(-1, keepdim=True) + pv
    left = (num_l / den_l.clamp_min(1e-30)).masked_fill(den_l <= 0, float("inf")).amin(-1)
    den_r = hi.sum(-1, keepdim=True) - pw
    num_r = (hi * c).sum(-1, keepdim=True) - pv
    right = (num_r / den_r.clamp_min(1e-30)).masked_fill(den_r <= 0, -float("inf")).amax(-1)
    return left, right


class T3FLTrust(nn.Module):
    """kind: 'T3' (z-slice type-3), 'IT2' (interval type-2), 'T1' (type-1)."""

    def __init__(self, embed_dim, kind="T3", n_slices=N_SLICES):
        super().__init__()
        self.kind = kind
        self.projector = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(),
                                       nn.Linear(64, N_INPUTS), nn.Sigmoid())
        feats = torch.tensor([r["feats"] for r in RULES], dtype=torch.long)
        labels = torch.tensor([r["labels"] for r in RULES], dtype=torch.long)
        self.register_buffer("feats", feats)
        self.register_buffer("centers", LABEL_CENTERS[labels])
        self.register_buffer("out_lo", torch.tensor([r["out"][0] for r in RULES]))
        self.register_buffer("out_hi", torch.tensor([r["out"][1] for r in RULES]))
        self.raw_width = nn.Parameter(torch.tensor(float(np.log(np.expm1(0.20)))))
        z = torch.linspace(1.0 / n_slices, 1.0, n_slices) if kind == "T3" else torch.ones(1)
        self.register_buffer("z", z)

    def forward(self, embed):
        x = self.projector(embed)
        dist = (x[..., self.feats] - self.centers).square()          # [B, R, 2]
        width = F.softplus(self.raw_width) + 0.05
        if self.kind == "T1":
            spread = torch.zeros_like(self.z)
        elif self.kind == "IT2":
            spread = torch.full_like(self.z, 0.25)
        else:
            spread = 0.35 * self.z
        s1 = (width * (1 - spread))[:, None, None, None]             # narrower  (Eq.7)
        s2 = (width * (1 + spread))[:, None, None, None]             # wider     (Eq.8)
        mu_lo = torch.exp(-dist[None] / (2 * s1.square().clamp_min(1e-12))).amin(-1)   # [Z,B,R]
        mu_hi = torch.exp(-dist[None] / (2 * s2.square().clamp_min(1e-12))).amin(-1)

        if self.kind == "T1":
            mid = 0.5 * (self.out_lo + self.out_hi)
            crisp = (mu_lo * mid).sum(-1) / mu_lo.sum(-1).clamp_min(1e-30)
            return crisp.mean(0).clamp(0, 1)

        yl, _ = km_type_reduce(mu_lo, mu_hi, self.out_lo)            # Eq.9
        _, yr = km_type_reduce(mu_lo, mu_hi, self.out_hi)            # Eq.10
        zw = self.z[:, None]
        yl3 = (zw * yl).sum(0) / self.z.sum()                        # Eq.11
        yr3 = (zw * yr).sum(0) / self.z.sum()                        # Eq.12
        return (0.5 * (yl3 + yr3)).clamp(0, 1)                       # Eq.13


# ---------------- Stage II: relational WGAN-GP reputation (Eq.21-22) ----------------
class Generator(nn.Module):
    def __init__(self, context_dim, noise_dim=NOISE_DIM):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(context_dim + noise_dim, 128), nn.LeakyReLU(0.2),
                                   nn.Linear(128, 64), nn.LeakyReLU(0.2))
        self.out = nn.Linear(64, 1)
        nn.init.uniform_(self.out.weight, -0.05, 0.05)
        nn.init.zeros_(self.out.bias)

    def forward(self, noise, ctx):
        logit = self.out(self.trunk(torch.cat([noise, ctx], 1))).squeeze(1)
        return torch.sigmoid(logit.clamp(-6.0, 6.0))   # clamp avoids permanent sigmoid saturation


class Critic(nn.Module):
    def __init__(self, context_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(context_dim + 1, 128), nn.LeakyReLU(0.2),
                                 nn.Linear(128, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))

    def forward(self, R, ctx):
        return self.net(torch.cat([R[:, None], ctx], 1)).squeeze(1)


def gradient_penalty(critic, real_R, fake_R, ctx, lam):
    eps = torch.rand(len(real_R), device=real_R.device)
    interp = (eps * real_R + (1 - eps) * fake_R).requires_grad_(True)
    grad = torch.autograd.grad(critic(interp, ctx).sum(), interp, create_graph=True)[0]
    return lam * (grad.abs() - 1).square().mean()


# ---------------- Stage III: fusion (Eq.24-25) and ZTA decision (Eq.26-27) ----------------
class AdaptiveFusion(nn.Module):
    def __init__(self, meta_dim=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(meta_dim + 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, trust, rep, meta):
        alpha = torch.sigmoid(self.net(torch.cat([trust[:, None], rep[:, None], meta], 1))).squeeze(1)
        return alpha * trust + (1 - alpha) * rep, alpha


def make_meta_from_scores(score):
    """[sensitivity_proxy, threat_proxy, access_freq_proxy]; threat = 1 - score."""
    risk = (1.0 - score).clamp(0, 1)
    half = torch.full_like(risk, 0.5)
    return torch.stack([half, risk, half], dim=1)


def zero_trust_decision(t_final, meta, w=ZTA_WEIGHTS, tau_min=TAU_MIN):
    """0 = Allow, 1 = Challenge, 2 = Deny."""
    tau = torch.sigmoid(w[0] * meta[:, 0] + w[1] * meta[:, 1] + w[2] * meta[:, 2])
    dec = torch.where(t_final >= tau, 0, torch.where(t_final >= tau_min, 1, 2))
    return dec, tau


class BehavioralScorer(nn.Module):
    """Trust-free signal used only by the 'AETL + Fusion (no trust)' ablation."""

    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TRuHEAL(nn.Module):
    def __init__(self, embed_dim, n_classes, trust_variant="t3fl",
                 use_trust=True, use_reputation=True, use_fusion=None):
        super().__init__()
        if use_fusion is None:
            use_fusion = use_trust and use_reputation
        self.use_trust, self.use_reputation, self.use_fusion = use_trust, use_reputation, use_fusion
        self.n_classes = n_classes
        kind = {"t3fl": "T3", "it2fl": "IT2", "type1": "T1"}[trust_variant]
        self.trust = T3FLTrust(embed_dim, kind=kind) if use_trust else None
        self.behavior = BehavioralScorer(embed_dim) if (use_fusion and not use_trust) else None
        ctx_dim = embed_dim + (1 if use_trust else 0)
        self.generator = Generator(ctx_dim) if use_reputation else None
        self.critic = Critic(ctx_dim) if use_reputation else None
        self.fusion = AdaptiveFusion() if (use_fusion and use_reputation) else None
        cls_in = embed_dim + (1 if (use_trust or use_reputation) else 0)
        self.cls_head = nn.Sequential(nn.Linear(cls_in, 128), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(128, 64), nn.ReLU(),
                                      nn.Linear(64, 1 if n_classes == 2 else n_classes))
        self.history = []

    def main_parameters(self):
        """Every parameter except the critic's (critic has its own optimiser)."""
        if not self.use_reputation:
            return self.parameters()
        critic_ids = {id(p) for p in self.critic.parameters()}
        return (p for p in self.parameters() if id(p) not in critic_ids)

    def forward(self, x, stochastic=True):
        B = x.shape[0]
        trust = self.trust(x) if self.use_trust else None
        ctx = torch.cat([trust[:, None], x], 1) if self.use_trust else x
        rep = None
        if self.use_reputation:
            noise = (torch.randn(B, NOISE_DIM, device=x.device) if stochastic
                     else torch.zeros(B, NOISE_DIM, device=x.device))
            rep = self.generator(noise, ctx)

        alpha = None
        if self.use_fusion and self.use_reputation:
            signal = trust if self.use_trust else self.behavior(x)
            meta = make_meta_from_scores((0.5 * signal + 0.5 * rep).detach())
            fused, alpha = self.fusion(signal, rep, meta)
        elif self.use_trust:
            fused = trust
        elif self.use_reputation:
            fused = rep
        else:
            fused = None

        feat = x if fused is None else torch.cat([x, fused[:, None]], 1)
        logits = self.cls_head(feat)
        logits = logits.squeeze(-1) if self.n_classes == 2 else logits
        return {"logits": logits, "trust": trust, "reputation": rep,
                "context": ctx, "fused": fused, "alpha": alpha}
