"""Does the between-experiment covariance term of the paper's Section 3.2 track the JCI-PC gain?

Per light-tunnel purchase list behind the 5000-row table: the between-experiment
share of pooled variance, and how the off-diagonal mass of the between term
splits between true edges and non-edges. Reads
`runs/rebuttal-jci-gain-diagnose.parquet`. No PC run.

    uv run python -m evaluation.chamber_pipeline.rebuttal_section32_check
"""

import numpy as np
import pandas as pd

from agent_contracts.integrations.causalchamber import create_contracted_chamber_agent
from evaluation.chamber_pipeline.rescore import parse_selection

a = create_contracted_chamber_agent(chamber="lt", configuration="standard", intervention_budget=80)
gt = a.ground_truth()
nodes = list(gt.index)
menu = list(a.available_experiments())
G = gt.to_numpy().astype(bool)
adj = G | G.T
raw = {m: a.query_intervention(m)[nodes].to_numpy(float) for m in menu}
allx = np.vstack(list(raw.values()))
sd = allx.std(0)
sd[sd < 1e-12] = 1
st = {m: (len(x), (x / sd).mean(0), np.cov((x / sd).T, bias=True)) for m, x in raw.items()}
d = pd.read_parquet("runs/rebuttal-jci-gain-diagnose.parquet")


def feats(names):
    n = np.array([st[x][0] for x in names], float)
    pi = n / n.sum()
    mu = np.array([st[x][1] for x in names])
    m = (pi[:, None] * mu).sum(0)
    between = sum(p * np.outer(u - m, u - m) for p, u in zip(pi, mu, strict=True))
    within = sum(p * st[x][2] for p, x in zip(pi, names, strict=True))
    off = ~np.eye(len(nodes), dtype=bool)
    tot = np.abs((between + within)[off]).sum()
    true = np.abs(between[adj & off]).sum()
    spur = np.abs(between[(~adj) & off]).sum()
    diag_b = np.diag(between)
    diag_t = np.diag(between + within)
    ok = diag_t > 1e-12
    return {
        "between_share": (diag_b[ok] / diag_t[ok]).mean(),
        "b_true": true / tot,
        "b_spurious": spur / tot,
        "spur_frac": spur / (true + spur),
    }


F = pd.DataFrame([feats(parse_selection(x)) for x in d.chosen_experiments])
d = pd.concat([d.reset_index(drop=True), F], axis=1)
print(
    "SPREAD of between_share within budget (sd):",
    d.groupby("budget_k").between_share.std().round(3).to_dict(),
    "means",
    d.groupby("budget_k").between_share.mean().round(2).to_dict(),
)
print(
    "pooled over all budgets: corr(gain, between_share) =",
    round(d.gain.corr(d.between_share), 2),
    " corr(f1_pc, between_share) =",
    round(d.f1_pc.corr(d.between_share), 2),
)
print(
    "between-experiment covariance, off-diagonal absolute mass relative to pooled, on TRUE edges vs NON-edges:"
)
print(d.groupby("budget_k")[["b_true", "b_spurious", "spur_frac"]].mean().round(3).to_string())
for k in (6, 30, 45):
    s = d[d.budget_k == k]
    print(
        f"k={k}: corr with (gain, f1_pc, f1_jci):",
        {
            c: (
                round(s[c].corr(s.gain), 2),
                round(s[c].corr(s.f1_pc), 2),
                round(s[c].corr(s.f1_jci), 2),
            )
            for c in ("between_share", "b_true", "b_spurious")
        },
    )
print(
    d[d.budget_k == 30]
    .groupby("agent_name")[["b_true", "b_spurious", "gain", "f1_pc"]]
    .mean()
    .round(3)
    .to_string()
)
