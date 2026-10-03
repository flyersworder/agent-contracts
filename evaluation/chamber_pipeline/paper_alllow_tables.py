"""Every all-low number the AAMAS paper's robustness section and supplement S2 use.

The all-low re-run (`docs/superpowers/specs/2026-09-29-low-negotiation-prereg.md`)
replaces the corpus loop, team and varsplit arms; the coverage rule, random and
single call make no coordination call and keep their corpus scores. Three
outputs, each from the files named beside it:

  robustness   Table `tab:robust`: PC 300/1500, GES 1500, UT-IGSP (core-20),
               JCI-PC 1500. VPS/OpenBLAS (`lownego_robustness.sh`).
  pool5000     Table `tab:pool5000`: PC and JCI-PC (per-regime) at 5000 rows,
               LT k=6/30/45. Mac/Accelerate (`lownego_pool5000.sh`), beside the
               rule and random rows of `rescored-t4-*` from the same machine.
  exo          Supplement Table `tab:s-exo` (paper PR #3): PC with the menu's
               manipulable variables exogenous, 300/1500 rows, all budgets.
               VPS/OpenBLAS; non-team arms from `vps-exo-pass.jsonl`.

Run from the repository root:
    PYTHONPATH=. uv run python -m evaluation.chamber_pipeline.paper_alllow_tables {robustness,pool5000,exo,covlaw}
Contrasts are design-level (one row per distinct purchase list, F1 averaged
over PC seeds), Welch 95% CI, resolved when |delta| > the unequal-n MDE, as in
`paper/aamas2027/figures/make_figures.py`, whose `contrast` this reuses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
_FIGURES = ROOT / "paper" / "aamas2027" / "figures"
if not (_FIGURES / "make_figures.py").exists():
    # The paper repo is private and nested; its contrast and MDE code is the
    # one every paper number uses, so reuse it rather than copy it.
    raise SystemExit(f"needs the paper repo checked out at {_FIGURES.parent}")
sys.path.insert(0, str(_FIGURES))
import make_figures as mf  # noqa: E402

TEAM_ARMS = ("llm_pc", "team", "team_varsplit")


def design_frame(prefix: str, metric: str = "f1", backend: str | None = None) -> pd.DataFrame:
    """One row per (design_key, agent_name): `metric` averaged over PC seeds.

    `backend`, if given, is asserted on every scored row, so a table never
    mixes BLAS backends (register: Accelerate and OpenBLAS diverge).
    """
    cells = pd.read_parquet(RUNS / f"{prefix}.parquet")
    cells = cells[(cells["status"] == "ok") & cells["chosen_experiments"].notna()]
    bykey = pd.read_parquet(RUNS / f"{prefix}-bykey.parquet")
    if backend is not None:
        assert set(bykey["blas_backend"]) == {backend}, (prefix, set(bykey["blas_backend"]))
    scores = bykey.groupby("design_key")[metric].mean().rename("f1")
    d = cells.drop_duplicates(["design_key", "agent_name"])[
        ["design_key", "agent_name", "chamber", "budget_k", "chosen_experiments"]
    ]
    return d.join(scores, on="design_key").dropna(subset=["f1"])


def fmt(c: dict[str, float]) -> str:
    s = f"{c['d']:+.3f}"
    return f"$\\mathbf{{{s}}}$" if abs(c["d"]) > c["mde"] else f"${s}$"


def cell(d: pd.DataFrame, a: str, b: str, ch: str, k: int) -> dict[str, float]:
    x, y = mf.arm(d, a, ch, k), mf.arm(d, b, ch, k)
    assert len(x) >= 2 and len(y) >= 2, (a, b, ch, k, len(x), len(y))
    return mf.contrast(x, y)


# ---------------------------------------------------------------- robustness
def robustness() -> None:
    ob = "scipy-openblas"
    low = {
        "PC300": pd.concat(
            [design_frame(f"lownego-{c}-rescored-rows300", backend=ob) for c in ("lt", "wt")]
        ),
        "PC1500": pd.concat(
            [design_frame(f"lownego-{c}-rescored-rows1500", backend=ob) for c in ("lt", "wt")]
        ),
        "GES": pd.concat(
            [design_frame(f"lownego-{s}-ges1500", backend=ob) for s in ("lt30", "wt21")]
        ),
        "UT-IGSP": design_frame("lownego-lt30-utigsp", "f1_core", backend=ob),
        "JCI": pd.concat(
            [design_frame(f"lownego-{s}-jci1500", backend=ob) for s in ("lt30", "wt21")]
        ),
    }
    corpus = {
        "PC300": design_frame("rescored-vps-rows300"),
        "PC1500": design_frame("rescored-vps-rows1500"),
        "GES": design_frame("rescored-vps-ges-rows1500-subset"),
        "UT-IGSP": design_frame("rescored-vps-utigsp-lt", "f1_core"),
        "JCI": design_frame("rescored-vps-jci-rows1500"),
    }
    rows = [
        ("coverage rule $-$ random", "lt", 30, mf.RULE["lt"], "random", corpus),
        ("", "wt", 21, mf.RULE["wt"], "random", corpus),
        ("team $-$ loop", "lt", 30, "team", "llm_pc", low),
        ("", "wt", 21, "team", "llm_pc", low),
        ("varsplit team $-$ team", "lt", 30, "team_varsplit", "team", low),
        ("", "wt", 21, "team_varsplit", "team", low),
    ]
    for label, ch, k, a, b, frames in rows:
        out = []
        for name, d in frames.items():
            if name == "UT-IGSP" and ch == "wt":
                out.append("--")
                continue
            c = cell(d, a, b, ch, k)
            out.append(f"{fmt(c)} [{c['lo']:+.3f},{c['hi']:+.3f}] mde {c['mde']:.3f}")
        print(f"{label or '':<26s} {ch.upper()} {k}: " + " | ".join(out))


# ---------------------------------------------------------------- pool5000
def pool5000() -> None:
    acc = "accelerate"
    low = {
        "PC": design_frame("lownego-lt-pc5000", backend=acc),
        "JCI-PC": design_frame("lownego-lt-jcireg5000", backend=acc),
    }
    t4 = {
        "PC": pd.concat(
            [design_frame(f"rescored-t4-{s}-pc5000", backend=acc) for s in ("k30", "ends")]
        ),
        "JCI-PC": pd.concat(
            [design_frame(f"rescored-t4-{s}-jcireg5000", backend=acc) for s in ("k30", "ends")]
        ),
    }
    arms = [
        ("llm_pc", low),
        ("team", low),
        ("team_varsplit", low),
        (mf.RULE["lt"], t4),
        ("random", t4),
    ]
    for name, src in arms:
        for method in ("PC", "JCI-PC"):
            vals = [mf.arm(src[method], name, "lt", k) for k in (6, 30, 45)]
            print(
                f"{name:<16s} {method:<6s} " + " ".join(f"{v.mean():.3f}(n{len(v)})" for v in vals)
            )
    for method in ("PC", "JCI-PC"):
        cs = [cell(low[method], "team", "llm_pc", "lt", k) for k in (6, 30, 45)]
        print(
            f"team - loop      {method:<6s} "
            + " ".join(f"{fmt(c)} [{c['lo']:+.3f},{c['hi']:+.3f}]" for c in cs)
        )
    # The text's "loop falls from X at 1500 rows to Y at 5000": both on the Mac.
    at1500 = mf.arm(design_frame("lownego-lt45-loop-pc1500mac", backend=acc), "llm_pc", "lt", 45)
    at5000 = mf.arm(low["PC"], "llm_pc", "lt", 45)
    print(f"loop LT45 PC: {at1500.mean():.3f} at 1500 rows -> {at5000.mean():.3f} at 5000 (Mac)")


# ---------------------------------------------------------------- exo (PR #3)
def exo() -> None:
    """Rule, random and single call from PR #3's VPS pass; the three team arms
    from the all-low lists, plain PC from their re-score and `pc_exo` from
    `lownego_robustness.sh`. Writes the supplement table."""
    probe = RUNS / "estimator-probe-2026-09-25"
    v = pd.read_json(probe / "vps-exo-pass.jsonl", lines=True)
    assert set(v["blas"]) == {"scipy-openblas"}
    des = pd.read_parquet(probe / "designs-1662.parquet")
    old = des[~des["agent_name"].isin(TEAM_ARMS)].merge(
        v, on=["chamber", "design_key"], validate="many_to_many"
    )
    old = old.rename(columns={"cap": "rows"})[
        ["chamber", "budget_k", "agent_name", "design_key", "rows", "mode", "f1"]
    ]
    new = []
    for ch in ("lt", "wt"):
        for rows in (300, 1500):
            for mode, prefix in (
                ("plain", f"lownego-{ch}-rescored-rows{rows}"),
                ("exo", f"lownego-{ch}-exo{rows}"),
            ):
                d = design_frame(prefix, backend="scipy-openblas")
                d = d[d["agent_name"].isin(TEAM_ARMS)].assign(rows=rows, mode=mode)
                new.append(
                    d[["chamber", "budget_k", "agent_name", "design_key", "rows", "mode", "f1"]]
                )
    d = pd.concat([old, *new], ignore_index=True)

    arms = [
        ("random", "random"),
        ("one_shot", "single call"),
        ("team", "team"),
        ("team_varsplit", "var.-split"),
        ("llm_pc", "loop"),
    ]
    lines, mdes, above = [], [], []
    for ch, name in (("lt", "LT"), ("wt", "WT")):
        for k in mf.BUDGETS[ch]:
            for rows in (300, 1500):
                g = d[(d["chamber"] == ch) & (d["budget_k"] == k) & (d["rows"] == rows)]
                p, e = g[g["mode"] == "plain"], g[g["mode"] == "exo"]
                rp = p[p["agent_name"] == mf.RULE[ch]]["f1"]
                re_ = e[e["agent_name"] == mf.RULE[ch]]["f1"]
                cells = [f"${rp.mean():.3f}$", f"${re_.mean():.3f}$"]
                for a, _ in arms:
                    c = mf.contrast(e[e["agent_name"] == a]["f1"], re_)
                    cells.append(fmt(c))
                    mdes.append(c["mde"])
                    if c["d"] > c["mde"]:
                        above.append(("rule", a, ch, k, rows, round(c["d"], 3)))
                loop = e[e["agent_name"] == "llm_pc"]["f1"]
                for a in ("team", "team_varsplit"):
                    c = mf.contrast(e[e["agent_name"] == a]["f1"], loop)
                    if c["d"] > c["mde"]:
                        above.append(("loop", a, ch, k, rows, round(c["d"], 3)))
                    if a == "team":  # the only loop column the table shows
                        cells.append(fmt(c))
                        mdes.append(c["mde"])
                lines.append(f"{name} & {k} & {rows} & " + " & ".join(cells) + r" \\")
            lines.append(r"\addlinespace")
    print("resolved above (rule or loop):", above or "none")
    gains = d[d["agent_name"].isin(mf.RULE.values())].pivot_table(
        index=["chamber", "budget_k", "rows"], columns="mode", values="f1"
    )
    gains = gains["exo"] - gains["plain"]
    print("rule gain range", round(gains.min(), 3), round(gains.max(), 3))
    print("MDE range", round(min(mdes), 3), round(max(mdes), 3))
    print("\n".join(lines[:-1]))
    tex = probe / "exo-table-rows.tex"
    tex.write_text("\n".join(lines[:-1]) + "\n")
    print(f"rows written to {tex}; mde range for the caption above")

    # Moderator (PR #3's S2 paragraph): each list's gain on the distinct
    # variables it covers, with a fixed effect per budget, per chamber and cap.
    import statsmodels.formula.api as smf

    from evaluation.chamber_pipeline import wt_menu_taxonomy as wtt
    from evaluation.chamber_pipeline.oracle_probe import _load

    wt_nodes = _load("wt")[3]
    lists = pd.concat(
        [
            des[~des["agent_name"].isin(TEAM_ARMS)][
                ["chamber", "budget_k", "design_key", "chosen_experiments"]
            ]
        ]
        + [
            design_frame(f"lownego-{ch}-rescored-rows300").query("agent_name in @TEAM_ARMS")[
                ["chamber", "budget_k", "design_key", "chosen_experiments"]
            ]
            for ch in ("lt", "wt")
        ]
    ).drop_duplicates(["chamber", "design_key"])
    lists["cov"] = [
        mf.lt_variable_count(s)
        if ch == "lt"
        else len({wtt.experiment_variable(x, wt_nodes) for x in s.split(",")})
        for ch, s in zip(lists["chamber"], lists["chosen_experiments"], strict=True)
    ]
    w = (
        d.drop_duplicates(["chamber", "design_key", "rows", "mode"])
        .pivot_table(index=["chamber", "design_key", "rows"], columns="mode", values="f1")
        .reset_index()
    )
    w["gain"] = w["exo"] - w["plain"]
    w = w.merge(lists[["chamber", "design_key", "cov", "budget_k"]], on=["chamber", "design_key"])
    for rows in (300, 1500):
        x = w[w["rows"] == rows]
        fit = smf.ols("gain ~ cov*chamber + C(budget_k)", x).fit()
        out = [f"interaction p={fit.pvalues['cov:chamber[T.wt]']:.1e}"]
        for ch in ("lt", "wt"):
            xc = x[x["chamber"] == ch]
            rr = smf.ols("gain ~ cov + C(budget_k)", xc).fit()
            out.append(f"{ch} slope {rr.params['cov']:+.4f} p={rr.pvalues['cov']:.1e} n={len(xc)}")
        print(f"moderator {rows}: " + " | ".join(out))


# ---------------------------------------------------------------- covlaw
def covlaw() -> None:
    """`COVLAW_ROWS` of make_figures.py (Figure 3, Table 3): distinct variables
    per arm, pred = RATE x delta_v, measured contrast at 300 rows."""
    from evaluation.chamber_pipeline import wt_menu_taxonomy as wtt
    from evaluation.chamber_pipeline.oracle_probe import _load

    wt_nodes = _load("wt")[3]

    def n_vars(ch: str, s: str) -> int:
        if ch == "lt":
            return mf.lt_variable_count(s)
        return len({wtt.experiment_variable(x, wt_nodes) for x in s.split(",")})

    d = mf.alllow_frame(300)
    d["nv"] = [n_vars(c, s) for c, s in zip(d["chamber"], d["chosen_experiments"], strict=True)]
    for ch in ("lt", "wt"):
        for k in mf.BUDGETS[ch]:
            g = d[(d["chamber"] == ch) & (d["budget_k"] == k)]
            mv = {a: g[g["agent_name"] == a]["nv"].mean() for a in TEAM_ARMS}
            for a, b, name in (
                ("team", "llm_pc", "team - loop"),
                ("team_varsplit", "team", "varsplit - team"),
            ):
                c = cell(d, a, b, ch, k)
                dv = mv[a] - mv[b]
                print(
                    f"{name:<16s} {ch.upper()} {k}: delta_v {dv:+.2f} "
                    f"pred {mf.RATE[(ch, 300)] * dv:+.4f} meas {c['d']:+.4f} mde {c['mde']:.4f}"
                )


if __name__ == "__main__":
    {"robustness": robustness, "pool5000": pool5000, "exo": exo, "covlaw": covlaw}[sys.argv[1]]()
