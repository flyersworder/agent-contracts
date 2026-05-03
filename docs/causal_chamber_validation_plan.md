# Causal Chamber Validation: Mainstream-Venue Extension Plan

**Status**: Planning
**Created**: 2026-05-03
**Owner**: qingye
**Target venues**: AAMAS 2027 (primary, ~Oct 2026 deadline), ECAI 2027 Athens (secondary, ~Apr 2027 deadline)
**Prerequisite**: COINE 2026 oral presentation (Paphos, May 25–26, 2026) ✅ accepted

---

## 1. Purpose of this document

This is the design plan for extending our peer-reviewed COINE 2026 paper
(`paper/paper.qmd`) into a mainstream-conference-grade submission by adding a
**verifiable empirical pillar** built on the Causal Chamber project
(<https://causalchamber.ai/>). It is not the paper itself, not the whitepaper
(`docs/whitepaper.md`), and not an implementation spec — it is the strategic
and technical blueprint we work from when implementing the extension step by
step.

When ambiguity arises about scope, defer to this document; when ambiguity
arises about the framework's formal definitions, defer to `docs/whitepaper.md`;
when ambiguity arises about what got peer-reviewed, defer to
`paper/paper.qmd`.

---

## 2. Strategic context

### 2.1 What we have

| Artifact | Location | Status |
|---|---|---|
| Theoretical framework | `docs/whitepaper.md` | Stable, implemented |
| Peer-reviewed paper | `paper/paper.qmd` (43KB Quarto, ~14pg LNCS) | **Accepted, oral, COINE 2026** |
| PyPI package | `ai-agent-contracts` v0.3.1 | Released |
| Core implementation | `src/agent_contracts/core/` | 81%+ coverage, 623+ tests |
| Framework integrations | `src/agent_contracts/integrations/` | LiteLLM, LangChain, LangGraph, Google ADK, Claude Agent SDK |
| Empirical pipeline #1 | `evaluation/research_pipeline/` | Multi-agent report generation, 25 topics, predictability finding |
| Empirical pipeline #2 | `evaluation/code_review_pipeline/` | Coder↔reviewer loop, 70 problems, 525× variance reduction |

### 2.2 Why a mainstream venue is reachable now

COINE acceptance for oral presentation is meaningful evidence the framework
already passed peer review. The reasons it does not yet reach AAMAS-main /
ECAI / NeurIPS standards are reviewer-known weaknesses we can articulate and
address:

1. **Quality measured by LLM-as-judge.** Both empirical pipelines depend on a
   Gemini-based evaluator. We mitigated this with the indeterminacy-aware
   evaluator (NeurIPS 2025 framework, `evaluation/indeterminacy_evaluator.py`),
   but reviewers can still ask "is the *judge* right?" — there is no ground
   truth to anchor against.
2. **Contribution framed as governance, not falsifiable performance.** The
   COINE paper claims contracts produce *predictable* execution and *enforce
   organizational policies*. These are correct claims but evaluated only via
   distributional metrics (variance, CV, tail percentiles). A reviewer can
   accept the framing without being convinced the framework solves a problem
   harder than "add a token counter."
3. **No comparison against an external benchmark.** Both pipelines are
   self-defined. The contract is the thing being measured *and* the thing
   defining what success looks like.

The Causal Chamber pillar fixes all three at once: ground-truth graphs replace
LLM-as-judge, edge accuracy and CI calibration are falsifiable performance
metrics, and the chambers are an externally maintained benchmark used by other
papers (Gamella et al. 2024 in *Nature Machine Intelligence* 2025).

### 2.3 Venue and timeline strategy

| Date | Event | Action |
|---|---|---|
| 2026-05-25 → 26 | COINE 2026 oral, Paphos | Present existing paper. **Capture reviewer feedback in person** (see §10). |
| 2026-05-27 → 31 | Recovery + integration kickoff | Write up COINE feedback. Spin up chamber adapter scaffolding. |
| 2026-06 → 09 | Chamber pillar implementation + experiments | See §9 milestones. |
| 2026-10 (target) | AAMAS 2027 submission | Primary mainstream target. Oral COINE version cited in cover letter as evidence of prior peer review. |
| 2026-11 → 2027-04 | Revision window | Strengthen for ECAI 2027. If AAMAS rejects, ECAI gets a stronger paper with reviewer feedback baked in. |
| 2027-04 (target) | ECAI 2027 submission | Backup mainstream target, Athens (EU-guaranteed). |

The dependency chain is one-way: COINE feedback → chamber experiments → AAMAS
submission → optional ECAI strengthening. No critical path passes through
US-located venues.

### 2.4 Extension delta vs COINE paper (≥30% novelty bar)

AAMAS and ECAI both expect substantial extension when re-submitting workshop
material. Our delta:

| Element | COINE 2026 | Extended (AAMAS/ECAI 2027) |
|---|---|---|
| Empirical pillars | 1 (LLM pipelines, Section 8 "Empirical Evaluation") | **2** (LLM pipelines + chamber benchmark) |
| Ground-truth available | No (LLM-as-judge) | **Yes** (known causal graphs) |
| Contract tightness sweep | No (single budget per condition) | **Yes** (Pareto frontier across 5 budget levels) |
| Falsifiable claims | Predictability, conservation | **+ Edge recovery accuracy, + CI calibration coverage** |
| Cross-domain validation | LLM only | **LLM + causal discovery** (governance gain transfers) |
| Run counts | 25 research topics × 2 conditions; 140 code-review trials (70 problems × 2) | **+ 1080 chamber runs** (900 CONTRACTED + 180 UNCONTRACTED, see §6.1) |
| Total new pages | — | ~6–8 pages of new content |

This comfortably clears the 30% novelty threshold. The extended paper is not
"COINE plus an appendix"; it is the COINE paper *recontextualized* as Pillar B
of a two-pillar empirical study, with chambers as Pillar A providing the
verifiable backbone.

---

## 3. The Causal Chamber pillar: what it gives us

### 3.1 What the chambers are

Two physical experimental devices built by Gamella et al. at ETH Zurich:

- **Light Tunnel** (`lt`): controllable RGB light source + rotating polarizers
  + photodiodes + camera. Standard configuration: **38 nodes, 57 edges**
  in the ground-truth causal graph (sparse, density ≈ 0.04).
- **Wind Tunnel** (`wt`): controllable fans + pressure sensors + microphones
  + tachometers. Standard configuration: **32 nodes, 42 edges**. Also has a
  `pressure-control` configuration with 32 nodes, 44 edges.

Ground-truth causal graphs are *known by physical construction* (they reflect
the wiring and known physical laws), validated against randomized control
experiments in the published manuscript appendices, and accessible
programmatically as adjacency-matrix DataFrames via
`causalchamber.ground_truth.graph(chamber, configuration)`.

### 3.2 Three execution paths (and which we use)

| Path | Mechanism | Cost | Realism | Decision |
|---|---|---|---|---|
| **A. Offline replay** | Pre-recorded interventional experiments per chamber, indexed by `intervention` column. Agent picks `k` of `M` available to "spend" budget on, sees real measurements. Menu sizes differ by dataset: LT `lt_interventions_standard_v1` has **M=59** experiments × 1000 samples; WT `wt_walks_v1` has **M=28** experiments × ~320K samples. | $0, infinite reruns | High — real physical-system measurements | **Primary** |
| **B. Simulator** | `causalchamber.simulators.Simulator` provides calibrated mechanistic models. Agent issues arbitrary intervention values. | $0, slower (CPU only) | Medium — calibrated model, not raw hardware | **Secondary** (counterfactual robustness check) |
| **C. Remote Lab** | Live chamber-time via subscription. Not in the Python package (no `causalchamber.remote` module — verified). | Subscription, opaque pricing, gated access | Highest, but only marginal vs A | **Skip** for the paper. Revisit only if a reviewer demands it. |

The crucial property of Path A: the agent's "intervention budget" maps to
*how many of the M pre-recorded experiments it queries*, not to physical
chamber-time. There is no quota, no fee, no application process. The data is
real; the budget is virtual; the contract framework gates the budget. Because
M differs across chambers (LT=59, WT=28), budget levels in §6.1 are expressed
as **fractions of the menu** rather than absolute counts, so Pareto curves
remain comparable across chambers.

### 3.3 Feasibility verified hands-on

Verified 2026-05-03 by ephemeral install (`uv run --no-project --with causalchamber`):

- Package installs cleanly, 14 dependencies, no friction
- 20 datasets enumerated via `causalchamber.datasets.list_available()`
- Datasets hosted on AWS `eu-central-1` (Frankfurt) — fast download from EU
- Ground-truth graphs accessible: confirmed 38/57 for `lt/standard`,
  32/42 for `wt/standard`, 32/44 for `wt/pressure-control`
- Sample edges look correct (`hatch → rpm_in`, `red → ir_1`, etc. — physically
  plausible)
- LT interventional dataset (`lt_interventions_standard_v1`, 3.91 MB) downloaded
  and parsed: 59 experiments, 1000 samples × 46 columns each, with explicit
  `intervention` column logging which variable was perturbed per row
- WT analog dataset (`wt_walks_v1`, 46.5 MB): 28 experiments × 320K samples × 37
  columns, also with `intervention` column. Different menu size and sample
  density from LT, but same shape of access pattern
- Simulator base class `causalchamber.simulators.Simulator` exists with
  `simulate_from_inputs()`, `inputs_names`, `outputs_names`, `parameters`

No blockers identified.

---

## 4. Mapping chambers onto the contract framework

### 4.1 The mapping is tight; no new primitives needed

Existing framework primitives — verified by reading
`src/agent_contracts/core/contract.py` and `src/agent_contracts/core/monitor.py`
— cover everything needed:

| Chamber concept | Contract framework primitive | Source |
|---|---|---|
| Intervention budget (`k` of menu) | `ResourceConstraints.per_tool_limits["intervene"]` | `contract.py` (added Dec 23) |
| Observation budget (passive samples) | `ResourceConstraints.per_tool_limits["observe"]` | same |
| Cost ceiling | `ResourceConstraints.cost_usd` | `contract.py` |
| Wall-clock deadline | `TemporalConstraints.deadline: datetime` | `contract.py` |
| Per-tool tracking | `ResourceUsage.tool_usage_by_name: dict[str, int]` | `monitor.py:49` |
| Edge-accuracy validator | `Contract.success_criteria` (Φ) | `contract.py` |
| CI-coverage validator | `Contract.success_criteria` (Φ) | `contract.py` |
| Chamber metadata | `Contract.metadata: dict[str, Any]` | `contract.py` |

Implication: the chamber benchmark is *evidence the framework was already
designed correctly*. We are not stretching primitives to fit; the primitives
already do what the benchmark requires. This is a story we should tell in the
paper itself.

### 4.2 The new integration adapter

A new file `src/agent_contracts/integrations/causalchamber.py` slots into the
existing integrations directory alongside `litellm_wrapper.py`,
`langchain.py`, etc. Same pattern: optional dependency, graceful import
fallback, registered in `integrations/__init__.py`.

```python
# src/agent_contracts/integrations/causalchamber.py
# ILLUSTRATIVE SKETCH — final API decided during M1-M2.
# This shows the SHAPE of the integration, not a ready-to-paste implementation.
# Tool-event wiring follows the same pattern used by langchain.py / litellm_wrapper.py
# (see those files for the exact callback / wrapper machinery).

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

try:
    from causalchamber.datasets import Dataset
    from causalchamber.ground_truth import graph as gt_graph
    CAUSAL_CHAMBER_AVAILABLE = True
except ImportError:
    CAUSAL_CHAMBER_AVAILABLE = False

from agent_contracts.core.contract import Contract, ResourceConstraints


ChamberId = Literal["lt", "wt"]
ConfigId = Literal["standard", "pressure-control"]

# Per-chamber dataset selection (LT and WT use different dataset names because
# their interventional designs differ — LT: 59-experiment uniform menu, WT:
# 28-experiment random-walk menu).
DATASET_FOR_CHAMBER: dict[ChamberId, str] = {
    "lt": "lt_interventions_standard_v1",
    "wt": "wt_walks_v1",
}


@dataclass
class ChamberContract:
    """Contract scoped to a Causal Chamber discovery task.

    Attaches a known ground-truth graph and a fixed intervention menu to a
    standard Contract. The agent's tool calls — query_intervention(),
    query_observation() — emit tool events tracked under per_tool_limits.
    """
    chamber: ChamberId
    configuration: ConfigId
    intervention_budget: int   # max k of M interventional queries (M chamber-specific)
    observation_budget: int = 0
    contract: Contract = field(init=False)
    _dataset: object = field(init=False, repr=False)
    _ground_truth: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._dataset = Dataset(
            name=DATASET_FOR_CHAMBER[self.chamber],
            root="./data/causalchamber",
            download=True,
        )
        self._ground_truth = gt_graph(
            chamber=self.chamber,
            configuration=self.configuration,
        )
        self.contract = Contract(
            resources=ResourceConstraints(
                per_tool_limits={
                    "intervene": self.intervention_budget,
                    "observe": self.observation_budget,
                },
            ),
            success_criteria=[
                edge_recovery_check(reference=self._ground_truth),
                ci_coverage_check(reference=self._ground_truth, alpha=0.05),
            ],
            metadata={
                "chamber": self.chamber,
                "configuration": self.configuration,
                "n_nodes": self._ground_truth.shape[0],
                "n_edges": int((self._ground_truth.values > 0).sum()),
                "menu_size": len(self._dataset.available_experiments()),
            },
        )

    def query_intervention(self, experiment_name: str) -> pd.DataFrame:
        """Tool the agent calls to spend one unit of intervention budget.

        The actual budget enforcement (incrementing
        ResourceUsage.tool_usage_by_name["intervene"], checking against
        per_tool_limits, raising ContractViolationError on overshoot) is
        wired the same way litellm_wrapper.py wires per-call token tracking.
        Concrete wiring decided during M2.
        """
        # ... emit tool event "intervene" via the same mechanism existing
        # integrations use (see litellm_wrapper.ContractedLLM)
        return self._dataset.get_experiment(experiment_name).as_pandas_dataframe()
```

This is an illustrative sketch only. Three details deliberately left
under-specified, to be locked in during M1–M2:

- **How tool events get emitted.** The existing integrations (`litellm_wrapper.py`,
  `langchain.py`) emit usage events via callback chains and wrapper methods,
  not via a global monitor. The chamber adapter follows whichever pattern is
  closest to the existing surface — copy, don't invent.
- **Where the validators live.** See §4.3.
- **Whether `ChamberContract` is a dataclass or a factory function.** Both
  styles exist in the current integrations; we'll match the most common one.

### 4.3 New scoring functions

Two validators slot into `Contract.success_criteria` (the Φ component of the
7-tuple, defined in `paper/paper.qmd` §4 "The Agent Contract Framework" /
`docs/whitepaper.md` §2.1):

- **Structural Hamming Distance (SHD)** between agent's reported adjacency
  matrix and `gt_graph()` reference. Standard metric in causal-discovery
  literature; bounded above by `n²`, lower is better.
- **CI calibration coverage**: agent reports a 95% CI on each edge's
  presence-probability; we measure the fraction of true edges (and absences)
  whose ground-truth indicator falls inside the reported interval. Target:
  coverage ≥ 0.95 with the smallest possible mean interval width (precision-
  coverage tradeoff).

These live in either a new file `src/agent_contracts/validators/causal.py` or
inside the integration module itself; final placement decided during
implementation.

### 4.4 Optional dependency wiring

```toml
# pyproject.toml addition
[project.optional-dependencies]
chambers = [
    "causalchamber>=0.1.5",
    "numpy>=1.26",      # transitively required, pinned to avoid yanked 2.4.0
]
```

`integrations/__init__.py` gets a parallel `try/except ImportError` block
matching the existing pattern for `litellm`, `langchain`, etc.

---

## 5. Agent variants under test

Three agent designs, all under the same contract:

1. **LLM-only ICL.** Pure in-context-learning agent (Claude Sonnet via the
   Claude Agent SDK integration). Sees observation summaries, decides which
   intervention to query next, eventually emits an adjacency matrix + edge
   confidences. Tests whether contract budgets force LLMs to be strategic
   about which interventions matter.
2. **LLM + PC.** LLM plans intervention sequence; the classical PC algorithm
   (constraint-based causal discovery) does the actual graph inference from
   the resulting data. Tests whether contracts let LLMs orchestrate
   classical methods well.
3. **LLM + GES.** Same as #2 but with greedy equivalence search instead of
   PC. GES uses score-based methods so the failure modes differ from PC.

All three use the same `ChamberContract`, the same `query_intervention` /
`query_observation` tools, the same scoring functions. Only the planning and
inference differ.

The point of having three variants: contract tightness sweeps may produce
*different Pareto frontiers per agent design*. If contracts dominate
identically across agents, that is a strong governance claim. If contracts
dominate differently, that is a finding about which agent designs benefit
most from governance — also publishable.

---

## 6. Experimental design

### 6.1 Full sweep

The headline experiment grid uses **menu-fraction budgets** so curves are
comparable across chambers despite different absolute menu sizes
(LT M=59, WT M=28). Two run families:

**CONTRACTED Pareto sweep** (the headline figure):

```
2 chambers          (lt with standard config; wt with standard config)
× 5 budget levels   (k/M ∈ {0.10, 0.25, 0.50, 0.75, 1.00})
                    → LT: k ∈ {6, 15, 30, 45, 59}
                    → WT: k ∈ {3,  7, 14, 21, 28}
× 3 agent variants  (LLM-only, LLM+PC, LLM+GES)
× 30 seeds          (statistical power)
= 900 runs
```

**UNCONTRACTED baseline** (single point per agent, for §6.2 comparison):

```
2 chambers × 3 agent variants × 30 seeds = 180 runs
```

**Total: 1080 runs.**

(WT `pressure-control` configuration has only 1 dataset experiment available
— too few for a budget sweep — so it's excluded from the main grid. May be
referenced in robustness discussion.)

Budget fractions chosen so the lower end forces real strategic choice
(k/M=0.10 means picking ~3–6 interventions for graphs of 32–38 nodes) while
the upper end (k/M=1.00) lets the agent observe every available intervention.
This range produces a non-trivial Pareto curve in (intervention budget ×
edge accuracy) space, plotted with the **fraction k/M** on the x-axis so LT
and WT lines share the same domain.

### 6.2 Comparison conditions

For each cell, two conditions:

- **CONTRACTED**: `per_tool_limits["intervene"] = k`; agent receives the
  budget in its system prompt; framework enforces violations as
  `ContractViolationError`.
- **UNCONTRACTED**: no per-tool limit; agent runs to its self-determined stop
  condition. In practice the menu is bounded (LT M=59, WT M=28), so an
  uncontracted agent's hard ceiling is M; the question is how often it
  *self-stops* short of that vs always exhausting the menu.

This is the same CONTRACTED/UNCONTRACTED pattern used by the existing
research and code-review pipelines, for narrative consistency.

### 6.3 Metrics collected per run

| Metric | Source | Used for |
|---|---|---|
| `interventions_used` | `tool_usage_by_name["intervene"]` | Budget compliance |
| `observations_used` | `tool_usage_by_name["observe"]` | Detect compensation |
| `wall_clock_seconds` | `ResourceUsage.compute_seconds` | Time-vs-quality tradeoff |
| `tokens_consumed` | `ResourceUsage.tokens` | Cost transferability vs LLM pipelines |
| `shd` | edge-recovery validator | Primary quality metric |
| `f1_edges` | derived from confusion matrix | Secondary quality metric |
| `ci_coverage` | CI-calibration validator | Calibration claim |
| `mean_ci_width` | CI-calibration validator | Precision-coverage tradeoff |
| `contract_state` | `Contract.state` enum | Violation rate |

### 6.4 Cost estimate

LLM cost per run ≈ $0.15 (≈30K tokens × $5/M for Claude Sonnet, conservative).
1080 total runs ≈ **~$165**. CPU cost on existing development hardware is
negligible (PC and GES are O(n³) at worst on a 38-node graph: seconds per
fit). Total experiment cost is well under what the existing research pipeline
cost.

### 6.5 Headline figure

The single figure that has to land for AAMAS reviewers: a Pareto plot with
**intervention budget on the x-axis** and **SHD on the y-axis** (lower = better).
One line per (chamber, configuration, agent_variant) combination. Error
bands from the 30 seeds.

What success looks like:

- A clear monotonic relationship between budget and quality (validates the
  framework controls a meaningful resource).
- Diminishing returns at high budget (validates that strategic intervention
  selection matters — agents do better than random).
- Different agent variants produce different curves (validates the framework
  is sensitive to agent design, not measuring noise).
- CI coverage ≥ 0.95 on the calibration sub-figure (validates the falsifiable
  uncertainty claim).

### 6.6 Reproducibility

Seed everything we can: numpy and the agent's tool-selection RNG are fully
seedable; LLM determinism is best-effort (Claude does not currently offer
strict seed-based reproducibility, so per-run LLM variance is captured by
running 30 seeds). Pin `causalchamber` version and rely on the package's
built-in checksum verification of dataset downloads. All experiment configs
live as YAML in `evaluation/chamber_pipeline/configs/`. Results dumped as
Parquet for fast aggregation.

---

## 7. Cross-pillar consistency study

The most reviewer-friendly experiment in this plan, and arguably the highest-
leverage paragraph in the paper. After the chamber sweep is done, re-run a
small subset (~10%) of the existing research and code-review pipeline
experiments at *matched contract tightness levels* — e.g., scale per-tool
limits in those domains so they correspond to chamber budget percentiles.

Goal: show that the *governance gains* observed in the chamber pillar
(reduced variance, predictable quality, runaway prevention) replicate in the
LLM pipelines, despite the LLM pipelines lacking ground-truth scoring.

This is the bridge between the two pillars. Without it, a reviewer can
plausibly claim "your chamber results are real but specific to causal
discovery." With it, the claim is "governance gains are domain-general; the
chamber pillar provides ground truth, the LLM pipelines provide breadth."

Scoping: a single subsection in the paper, ≤ 1 figure. This is not a
re-execution of the existing pipelines from scratch, just a targeted
re-running at matched tightness.

---

## 8. The chamber pipeline as code

New directory `evaluation/chamber_pipeline/`, mirroring
`evaluation/research_pipeline/` and `evaluation/code_review_pipeline/`:

```
evaluation/chamber_pipeline/
├── __init__.py
├── README.md                 # what this experiment does
├── RESULTS.md                # written after the sweep, like sister pipelines
├── configs/
│   ├── lt_standard.yaml
│   ├── wt_standard.yaml
│   └── wt_pressure_control.yaml
├── agents.py                 # LLM-only, LLM+PC, LLM+GES
├── scoring.py                # SHD, F1, CI coverage
├── orchestrator.py           # one experiment cell end-to-end
├── run_experiment.py         # CLI entry point; full sweep
├── analyze_results.py        # aggregation + Pareto figure generation
└── figures/                  # generated plots, parquet results
```

The structural parity with the existing pipelines is intentional. Anyone
reading the codebase should immediately recognize the chamber pipeline as
"another evaluation pipeline of the same shape," not as a special case.

---

## 9. Milestones and timeline

5 months between COINE presentation (May 26) and AAMAS submission (~Oct 1).

| # | Window | Milestone | Acceptance criterion |
|---|---|---|---|
| M1 | 2026-05-27 → 06-07 | COINE feedback writeup + chamber adapter scaffolding | New file `docs/coine_feedback.md` written; `integrations/causalchamber.py` stub committed; `chambers` extra in `pyproject.toml`; failing smoke test exists |
| M2 | 2026-06-08 → 06-21 | Adapter complete + ground-truth scoring functions | Smoke test passes: load `lt/standard` graph, run a fake agent that returns the ground truth, score reports SHD=0 and F1=1 |
| M3 | 2026-06-22 → 07-05 | Three baseline agents implemented | All three variants run end-to-end on a single budget cell; produce coherent adjacency-matrix outputs |
| M4 | 2026-07-06 → 07-19 | Pilot sweep | 1 chamber × 3 budgets × 30 seeds = 90 runs; preliminary Pareto curve looks monotonic |
| M5 | 2026-07-20 → 08-23 | Full sweep | All 1080 runs (900 CONTRACTED + 180 UNCONTRACTED) complete; results in Parquet; headline figure generated |
| M6 | 2026-08-24 → 09-06 | Cross-pillar consistency study | Subset re-runs of research / code-review pipelines at matched tightness; one bridging figure |
| M7 | 2026-09-07 → 09-27 | Paper extension drafted | `paper/paper-extended.qmd` (or branch of `paper.qmd`) contains new Section 8 ("Empirical Evaluation") subsections; intro and abstract rewritten to reflect two-pillar structure |
| M8 | 2026-09-28 → 10-XX | Submission polish | All AAMAS formatting requirements met; cover letter cites COINE acceptance |

Each milestone unblocks the next. The plan has buffer baked in (M5 has 5
weeks for what should be 3 weeks of compute), specifically because the agent
implementations in M3 are the highest-uncertainty piece.

---

## 10. COINE feedback capture (May 25–26 in Paphos)

Workshop attendance is part of this plan, not a side trip. Specific things
to actively probe at the oral session:

- **Reviewers' counterexamples.** Did anyone surface a multi-agent scenario
  where contracts fail or are easily evaded? These become test cases in the
  chamber experiments.
- **Quality-evaluation pushback.** If reviewers raise the LLM-as-judge
  concern, articulate the chamber plan and ask whether ground-truth
  validation addresses their concern. If yes, lock in framing. If no,
  understand what would.
- **Agents-community priors.** AAMAS reviewers will overlap heavily with the
  COINE audience. What metrics do they expect to see in a contracts paper?
  What baselines? What axes of comparison?
- **Calibrated-CI feasibility.** Ask anyone in the causal-inference / MAS
  intersection whether 95% coverage on edge presence is reasonable to claim,
  or whether we should soften to "well-calibrated edge confidences."

Output: a writeup at `docs/coine_feedback.md` by 2026-06-07 (start of M1
window), feeding directly into the chamber experiment design before the
adapter is locked.

---

## 11. Risks and mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Causal-discovery agent is a new paradigm; agent implementations are harder than expected | Medium | High (delays M3 → M5) | Start M3 with LLM-only (simplest); fall back to LLM+PC only as the main result if LLM+GES proves problematic. Three variants is a stretch goal, two is the floor. |
| R2 | Calibrated CI coverage too hard for LLM-based agents | Medium | Medium (weakens calibration claim) | Soften to "edge confidences" (no formal coverage guarantee) in v1; add bootstrap-based coverage in v2 if time allows. The SHD/F1 results stand independently. |
| R3 | AAMAS reviewers reject because the chamber pillar isn't on real hardware | Low | Medium | Cite Gamella et al. — the offline data *is* real-hardware measurements, just pre-recorded. The 59 experiments per chamber are the same data their own published validation rests on. |
| R4 | `causalchamber` package breaks (yanked numpy 2.4.0 already a yellow flag) | Low | Low | Pin a working version range in the `chambers` extra. Vendor the offline datasets to our own storage if upstream becomes unreliable. |
| R5 | COINE attendance reveals a substantive flaw in the framework itself | Low | High | Address in M1; if it's framework-level (not just experiment-level), reassess whether AAMAS is reachable on the original timeline or whether we need to push to ECAI 2027 only. |
| R6 | AAMAS 2027 location announced in late May 2026 lands somewhere we can't travel to | Medium | Medium | Already mitigated by parallel ECAI 2027 (Athens, confirmed) plan. AAMAS becomes optional rather than primary if location is bad. |
| R7 | Compute cost overruns | Very low | Low | Budget is ~$200; even 5× overrun is trivially absorbed. |

No risk in this list is severe enough to threaten the plan. R1 is the most
work-likely, R5 is the most damage-likely; both are addressed up-front.

---

## 12. Open questions (decisions deferred to implementation)

These are deliberate non-decisions, listed here so we know to make them when
we get there:

1. **Adapter API signature.** The §4.2 sketch is approximate. The exact split
   between a `ChamberContract` dataclass vs a function-style
   `create_chamber_contract()` factory follows whatever pattern the existing
   integrations use most consistently. Decide during M1.
2. **Where the validators live.** `validators/causal.py` (new top-level
   submodule) vs inline in `integrations/causalchamber.py`. Probably the
   former if we anticipate other ground-truth domains; the latter if
   chambers stay the only such domain. Decide during M2.
3. **Whether to add the simulator path (Path B from §3.2).** Adds robustness
   evidence but doubles experiment cost and complexity. Decision: include
   only if M5 finishes ahead of schedule; otherwise defer to a v2 / journal
   version.
4. **Paper source organization.** Branch `paper/paper.qmd` vs new
   `paper/paper-extended.qmd`. Branch is cleaner version-control; new file is
   safer if both COINE-archival and AAMAS-extension versions need to coexist.
   Decide during M7.
5. **Whether the chamber benchmark gets extracted as a standalone artifact.**
   Could be released as `agent-contracts-bench` on PyPI alongside the main
   package, giving other framework authors a standard reference experiment
   to run. High community-impact upside but not on the critical path. Decide
   post-submission.

---

## 13. References

### Primary sources

- Gamella, J. L., Peters, J., Bühlmann, P. (2025). *Causal chambers as a
  real-world physical testbed for AI methodology.* Nature Machine
  Intelligence. <https://doi.org/10.1038/s42256-024-00964-x>
  (arXiv preprint: <https://arxiv.org/abs/2404.11341>)
- Causal Chamber project: <https://causalchamber.ai/>
- `causalchamber` Python package: <https://pypi.org/project/causalchamber/>
- Package source: <https://github.com/juangamella/causal-chamber-package>
- Datasets repository: <https://github.com/juangamella/causal-chamber>

### Internal references

- Framework whitepaper: `docs/whitepaper.md`
- Peer-reviewed paper (COINE 2026 oral): `paper/paper.qmd`
- COINE 2026 submission record: `paper/SUBMISSION_PLAN.md`
- Existing pipelines: `evaluation/research_pipeline/`,
  `evaluation/code_review_pipeline/`
- Indeterminacy-aware evaluator: `evaluation/indeterminacy_evaluator.py`
  (NeurIPS 2025 Guerdan et al. framework)

### Venue references (for AAMAS / ECAI submission)

- AAMAS 2026 (where COINE is co-located): Paphos, Cyprus, May 25–29, 2026
- AAMAS 2027 location: TBD — to be announced at AAMAS 2026 Cyprus
- ECAI 2027 (confirmed): Athens, Greece, October 2027
- ECAI 2028 (confirmed fallback): Helsinki, Finland

---

*End of plan. Edits to this document during implementation should be tracked
in commit messages, not in a changelog block here — the file's git history is
the changelog.*
