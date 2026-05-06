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

The Causal Chamber pillar **directly fixes #1 and #3**: ground-truth graphs
replace LLM-as-judge, and the chambers are an externally maintained benchmark
used by other papers (Gamella et al. 2024 in *Nature Machine Intelligence*
2025).

**Weakness #2 requires more than the chamber data alone.** A single LLM agent
calling a single budgeted tool only validates that "budget caps produce
predictable quality" — true of `if k >= limit: break` and not specifically a
contract-framework finding. To convert the chamber substrate into a
falsifiable claim about *the contract framework*, three additional design
choices are load-bearing:

- **Multi-agent scenarios under conservation laws** (§5) so the framework's
  delegation primitives — not just `per_tool_limits` — are exercised. Without
  this, the chamber pillar is single-agent work submitted to a multi-agent
  venue.
- **Non-LLM principled baselines** (§5) so the comparison set is not "three
  flavors of LLM." Without this, the headline reads "LLMs do causal discovery
  under budgets" rather than "the contract framework helps causal-discovery
  agents."
- **Cross-pillar governance-transfer evidence** (§7 — promoted from
  subsection to full section) so chamber findings about governance gains are
  *linked* to LLM-pipeline findings. Without this, the two pillars stand
  independently and the paper has two contributions instead of one joint
  contribution.

Restated: chambers buy us the *substrate* for a falsifiable claim. §5 and §7
below convert that substrate into a falsifiable claim about *the contract
framework*, which is what AAMAS reviewers care about.

### 2.3 Venue and timeline strategy

| Date | Event | Action |
|---|---|---|
| **2026-05-06 → 05-24** | **Pre-COINE M1 head start** | Chamber adapter scaffolding (pulled forward from original 05-27 start; see §9 / §10 for why this is now possible). |
| 2026-05-25 → 26 | COINE 2026 oral, Paphos | Present existing paper. 15-min slot, **low-information signal** — see §10. |
| 2026-05-27 → 06-01 | Light COINE writeup (1 page max) | Note any genuinely surprising audience question. Empty note is acceptable. |
| 2026-06 → 09 | Chamber pillar implementation + cross-pillar transfer experiments | See §9 milestones. |
| 2026-10 (target) | AAMAS 2027 submission | Primary mainstream target. Oral COINE version cited in cover letter as evidence of prior peer review. |
| 2026-11 → 2027-04 | Revision window | Strengthen for ECAI 2027. If AAMAS rejects, ECAI gets a stronger paper. |
| 2027-04 (target) | ECAI 2027 submission | Backup mainstream target, Athens (EU-guaranteed). |

The dependency chain is one-way: chamber experiments → AAMAS submission →
optional ECAI strengthening. **COINE is parallel, not gating.** No critical
path passes through US-located venues.

### 2.4 Extension delta vs COINE paper (≥30% novelty bar)

AAMAS and ECAI both expect substantial extension when re-submitting workshop
material. Our delta:

| Element | COINE 2026 | Extended (AAMAS/ECAI 2027) |
|---|---|---|
| Empirical pillars | 1 (LLM pipelines, Section 8 "Empirical Evaluation") | **2 + bridge** (LLM pipelines + chamber benchmark + cross-pillar transfer study) |
| Ground-truth available | No (LLM-as-judge) | **Yes** (known causal graphs) |
| Contract tightness sweep | No (single budget per condition) | **Yes** (Pareto frontier across 5 budget levels) |
| Falsifiable claims | Predictability, conservation | **+ Edge recovery accuracy, + CI calibration coverage, + cross-domain governance transfer** |
| Cross-domain validation | LLM only | **LLM + causal discovery** with explicit transfer experiment (§7) |
| Non-LLM baselines | None | **Random + GreedyIG-lite** (principled non-LLM, isolates LLM contribution) |
| Multi-agent under conservation law | Implicit in research / code-review pipelines | **Explicit** (Planner+Reasoner variant, §5) |
| Run counts | 25 research topics × 2 conditions; 140 code-review trials (70 problems × 2) | **+ ~1800 chamber runs** (1080 LLM-bearing) **+ ~2750 cross-pillar re-runs** at matched tightness (see §6.1, §7) |
| Total new pages | — | ~7–9 pages of new content |

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

## 5. Agent variants and baselines under test

Five variants on two axes of variation. The matrix is sized to isolate the
contract framework's contribution from the LLM's contribution, so the
chamber pillar's findings are claims about *the framework* rather than
"LLMs do causal discovery."

**Axis A — architecture:** single-agent vs multi-agent. Single-agent designs
validate that the framework can govern one agent's tool use. Multi-agent
designs validate the framework's *unique* contribution: conservation laws
across delegated budgets, structured violation events surfacing at boundaries,
policy enforcement at handoffs. AAMAS is a multi-agent venue; the chamber
design needs at least one multi-agent variant for venue fit alone.

**Axis B — method:** LLM-orchestrated vs principled-classical vs naive.
Without a non-LLM baseline, the comparison set is "three flavors of LLM" and
the framework looks like a wrapper around prompting.

### 5.1 The five variants

| # | Variant | Architecture | Method | Why included |
|---|---|---|---|---|
| 1 | **Random** | Single-agent | Naive | Absolute floor of the Pareto plot. Picks `k` interventions uniformly at random, runs the same graph-inference step as LLM+PC. ~50 LOC; cost ~free. |
| 2 | **GreedyIG-lite** | Single-agent | Non-LLM, principled | Reference line for "what a principled non-LLM achieves." Fits a linear-Gaussian SCM to current data, picks the next intervention greedily by approximate variance reduction in the MAP-graph posterior (Tong & Koller 2001 in spirit; Hauser & Bühlmann 2014 in form). Full-posterior version deferred to a v2 / journal extension. |
| 3 | **LLM-only** | Single-agent | LLM throughout | Pure in-context-learning (ICL) agent (**DeepSeek v4 Flash via OpenRouter**, accessed through the framework's LiteLLM integration). LLM picks each intervention and emits final adjacency matrix + edge confidences. |
| 4 | **LLM+PC** | Single-agent | LLM-orchestrated classical | LLM plans intervention sequence; classical PC algorithm infers the graph from the resulting data. |
| 5 | **Planner+Reasoner** ⭐ | **Multi-agent** | LLM throughout, two roles | Planner agent picks interventions under sub-budget A; Reasoner agent proposes graph under sub-budget B. **Conservation law: A + B ≤ total intervention budget.** Exercises the framework's delegation primitives. |

**Variant 5 is the contribution-load-bearing variant.** Variants 1–4
establish the Pareto landscape; variant 5 does something the alternatives
literally cannot — enforce a budget *across* an agent boundary with
structured violation events at the handoff.

GES (greedy equivalence search) and other score-based discovery methods are
deferred to a v2 / journal extension where method-axis breadth becomes the
primary contribution. For the present submission, two principled-method
cells (LLM+PC and the Planner+Reasoner multi-agent design) plus two non-LLM
baselines (Random, GreedyIG-lite) and one pure-LLM ICL variant cover the
contribution claim.

**Model choice across both pillars: DeepSeek v4 Flash via OpenRouter**
($0.14/M input, $0.28/M output, 1M-token context window), accessed through
the framework's existing LiteLLM integration. Used for all LLM-bearing
variants in the chamber pillar (variants 3, 4, 5 above) **and** for §7's
cross-pillar re-runs of the research and code-review pipelines (which
parameterize `--model` and previously defaulted to Gemini 2.5 Flash Lite).
Single-model consistency across pillars is deliberate — variance in the
cross-pillar transfer figures (§7) cannot be attributed to model
differences, only to domain.

Chosen for: (a) cost efficiency at experimental scale (~30× cheaper than
Claude Sonnet on a typical 25K-input/5K-output run); (b) open-weights
reproducibility — third parties can replicate the full paper with one
OpenRouter key; (c) the 1M context window so the agent never has to
truncate or summarize prior observations during an intervention sequence
(no windowing logic to complicate the experimental setup).

A higher-capacity robustness sweep using **DeepSeek v4 Pro** (same model
family, ~3× Flash cost, $0.435/$0.870 per M tok in/out) runs at the
cell-grid scope per §6.7. Within-family scale-up isolates *capacity* as
the only axis of variation; a cross-vendor sensitivity check would
confound capacity, vendor, and training-pipeline differences.

### 5.2 Implications for the experimental matrix

The §6.1 cell count grows but only LLM-bearing cells incur real cost:

```
CONTRACTED Pareto sweep:
  2 chambers × 5 budget levels × 5 variants × 30 seeds = 1500 runs
  (was 900; +600 from substituting Random/GreedyIG-lite/Planner+Reasoner for GES)
  Of which LLM-bearing: 3 variants × 300 cells = 900 LLM runs (unchanged)

UNCONTRACTED baseline:
  2 chambers × 5 variants × 30 seeds = 300 runs (was 180)
  Of which LLM-bearing: 3 variants × 60 cells = 180 LLM runs (unchanged)

Total: 1800 runs (+67% over original 1080 total); LLM-bearing: 1080 (unchanged)
```

Random and GreedyIG-lite are CPU-only (no LLM calls). Planner+Reasoner
roughly doubles tokens per run since two agents communicate, so it
dominates the LLM-bearing cost. Chamber-pillar cost at DeepSeek v4 Flash
pricing: **~$7**. See §6.4 for the grand total across all components.

If M5 timeline pressure forces a cut, the cut order — from most to least
expendable — is: (1) **LLM-only** (LLM+PC subsumes its claim for the main
contribution), (2) **GreedyIG-lite** seeds reduced from 30 to 10 and only
3 of 5 budget levels exercised. **The Pareto-floor + main-hybrid +
multi-agent triplet (Random + LLM+PC + Planner+Reasoner) is protected**,
matching R1's floor in §11.

### 5.3 Headline figure (updated)

The Pareto plot — x-axis intervention-budget-fraction k/M, y-axis SHD —
carries **five lines per chamber**, each with an explicit interpretive role:

- **Random**: absolute floor. LLM and principled methods both must clear it.
- **GreedyIG-lite**: principled non-LLM reference. Gap between it and LLM
  variants is "what the LLM adds."
- **LLM-only**: what an LLM does *without* classical infrastructure.
- **LLM+PC**: what an LLM does *with* classical infrastructure.
- **Planner+Reasoner**: what the *contract framework* does when budget is
  delegated across an agent boundary.

The story the figure tells: if Planner+Reasoner sits *on or above* LLM+PC at
matched total budget, that is direct evidence the framework's conservation
laws preserve quality under delegation — i.e., the framework adds value, not
overhead. If it sits below, that is *also* a publishable finding (delegation
overhead has measurable cost), and the paper's framing rotates accordingly
(see R8 in §11).

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
× 5 agent variants  (Random, GreedyIG-lite, LLM-only, LLM+PC, Planner+Reasoner)
× 30 seeds          (statistical power)
= 1500 runs
   of which 900 are LLM-bearing (3 LLM variants × 300 cells)
```

**UNCONTRACTED baseline** (single point per agent, for §6.2 comparison):

```
2 chambers × 5 agent variants × 30 seeds = 300 runs
   of which 180 are LLM-bearing
```

**Total: 1800 runs (1080 LLM-bearing).**

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

All LLM-bearing runs use DeepSeek v4 Flash via OpenRouter ($0.14/M input,
$0.28/M output) for both pillars. The §6.7 robustness sweep adds DeepSeek
v4 Pro ($0.435/$0.870 per M tok) at cell-grid scope.

**Chamber pillar (Flash):**

- **LLM-only**: ~$0.005/run (≈25K input + 5K output tokens at Flash pricing)
- **LLM+PC**: ~$0.003/run (LLM only plans; PC inference is free CPU)
- **Planner+Reasoner**: ~$0.010/run (two LLM agents communicating, ~2× tokens)
- **Random, GreedyIG-lite**: ~free (no LLM calls; CPU-only fit + selection)

**Cross-pillar §7 re-runs (Flash, via existing pipelines' `--model` flag):**

- **Research pipeline**: ~$0.05/run (multi-agent Researcher→Analyzer→Reporter,
  ~250K tokens cumulative across agent turns)
- **Code-review pipeline**: ~$0.015/run (iterative Coder↔Reviewer loop,
  ~100K tokens cumulative across iterations)

**§6.7 DeepSeek v4 Pro robustness sweep:**

- 300 Pro runs at cell-grid scope (2 chambers × 3 budget levels × 5 variants
  × 10 seeds): ~$5

**Cost totals:**

| Component | Runs | Total |
|---|---|---|
| Chamber pillar (Flash) | 1080 LLM-bearing | ~$7 |
| Cross-pillar research (Flash) | 1250 | ~$62 |
| Cross-pillar code-review (Flash) | 1500 | ~$23 |
| Pro robustness sweep (cell-grid) | 300 | ~$5 |
| **Grand total** | **4130 LLM runs** | **~$97** |

CPU cost on existing development hardware is negligible — PC, GreedyIG-lite
linear-Gaussian fitting, and selection are O(n³) at worst on a 38-node
graph (seconds per fit). The project's compute budget ceases to be a
meaningful constraint; engineering time is the binding resource. Actual
per-run cost is instrumented via `ResourceUsage.cost_usd` and reported
in the paper, so the headline ~$97 figure is a budget ceiling rather than
a guess.

### 6.5 Headline figure

The single figure that has to land for AAMAS reviewers (**Figure 6.1**): a
Pareto plot with **intervention-budget fraction k/M on the x-axis** and
**SHD on the y-axis** (lower = better). **Five lines per chamber** (Random,
GreedyIG-lite, LLM-only, LLM+PC, Planner+Reasoner) with explicit
interpretive roles per §5.3. Error bands from the 30 seeds. A companion
calibration plot (**Figure 6.2**) reports CI coverage and mean interval
width per variant at each budget level.

What success looks like:

- All LLM variants clear the **Random** floor (validates the LLM does
  something better than picking interventions blind).
- LLM+PC and/or Planner+Reasoner clear **GreedyIG-lite** at most budget
  levels (validates the LLM adds value over a principled non-LLM baseline —
  defends against the "is the LLM even necessary?" reviewer attack).
- **Planner+Reasoner sits on or above LLM+PC** at matched *total* budget
  (validates the framework's conservation laws preserve quality under
  delegation — the contribution-load-bearing claim).
- A clear monotonic relationship between budget and quality across all
  variants (validates the framework controls a meaningful resource).
- Diminishing returns at high budget (validates that strategic intervention
  selection matters).
- CI coverage ≥ 0.95 on **Figure 6.2** (validates the falsifiable
  uncertainty claim).

### 6.6 Reproducibility

Seed everything we can: numpy and the agent's tool-selection RNG are fully
seedable; LLM determinism is best-effort (OpenRouter inference is non-
deterministic across requests even at temperature=0, so per-run LLM
variance is captured by running 30 seeds). Pin `causalchamber` version and
rely on the package's built-in checksum verification of dataset downloads.
Pin DeepSeek model versions via OpenRouter's stable model IDs
(`deepseek/deepseek-v4-flash`, `deepseek/deepseek-v4-pro`); record the
exact model snapshot in run metadata for forensic reproducibility. All
experiment configs live as YAML in `evaluation/chamber_pipeline/configs/`.
Results dumped as Parquet for fast aggregation.

### 6.7 DeepSeek v4 Pro robustness sweep

A higher-capacity sweep using **DeepSeek v4 Pro** ($0.435/$0.870 per M tok
in/out, same model family as Flash, ~3× cost, 1M-token context) replicates
a subset of the chamber Pareto sweep at the **cell-grid scope**:

- 2 chambers (LT, WT, both standard configuration)
- 3 budget levels (k/M ∈ {0.10, 0.50, 1.00} — endpoints + midpoint of
  the five-level Flash sweep)
- 5 variants (Random, GreedyIG-lite, LLM-only, LLM+PC, Planner+Reasoner)
- 10 seeds (reduced from 30 since Pro is for capacity-axis comparison,
  not primary statistical claims)
- = **300 Pro runs at ~$5**

The 3-level subset is chosen so the Pareto plot at Pro capacity remains
interpretable (low-, mid-, high-budget points) while keeping run count
small enough that results land inside M5's window without delaying M6.

**Visualization.** Pro Pareto points overlay onto the headline Figure 6.1
as dashed lines (Flash = solid, Pro = dashed). If the curves coincide, the
framework's claims are robust to capacity within the DeepSeek family. If
they diverge, that is itself a finding and the paper's "capacity-
invariance" framing rotates accordingly.

**Why same-family scaling.** Cross-vendor sensitivity (e.g., Claude
Sonnet) confounds three axes simultaneously: capacity, vendor, and
training-pipeline closedness. Same-family scaling (Flash → Pro) isolates
capacity as the only axis of variation, which is the cleaner robustness
argument and the one reviewers can interpret unambiguously.

**Scope flex.** If M5 finishes ahead of schedule, scope can expand to
full replication (1080 Pro runs ≈ $16, parallel curves at full budget
resolution). If M5 slips, scope can drop to one-cell minimum (50 runs
≈ $1, single sensitivity datapoint). The cell-grid (300 runs) is the
default.

---

## 7. Cross-pillar governance transfer

This was a subsection in the original plan. **Promoted to a full section**
because it is the load-bearing experiment for the two-pillar story. Without
it, chamber results stand as "interesting causal-discovery findings,"
LLM-pipeline results stand as "interesting governance findings," and the
paper has no joint claim.

### 7.1 What the cross-pillar evidence has to show

A single property: **governance gains observed in one pillar replicate in
the other at matched contract tightness.** Three concrete claims, one figure
each:

1. **Variance-reduction transfer.** CV reduction observed in chamber
   Planner+Reasoner replicates in research-pipeline
   Researcher→Analyzer→Reporter at matched tightness.
2. **Conservation-law transfer.** Delegated-budget compliance rate observed
   in chambers replicates in research-pipeline budget delegation across
   delegation depth.
3. **Runaway-prevention transfer.** Iteration-cap effectiveness in
   code-review Coder↔Reviewer replicates in chamber multi-agent variants
   under matched delegation depth.

### 7.2 Tightness matching across domains

Tightness levels from §6.1 (k/M ∈ {0.10, 0.25, 0.50, 0.75, 1.00}) map to
LLM-pipeline budgets by *quantile-matching the agent's natural usage
distribution*:

- For each LLM pipeline, derive the empirical distribution of total tool
  calls (research) or total iterations (code-review) **from the existing
  UNCONTRACTED runs already collected for the COINE paper** — no new
  calibration runs required.
- Tightness level *t* corresponds to the *t*-th percentile of that
  distribution. *t = 0.10* means "limit the agent to the 10th percentile of
  what it would naturally use."

This matching controls for the agent's *self-selected* effort rather than
for absolute counts — defensible across domains where absolute units don't
commensurate (chamber interventions ≠ research-pipeline tool calls).

### 7.3 The three figures

**Figure 7.1 — variance-reduction transfer.**

- x-axis: tightness level (5 quantiles)
- y-axis: CV of quality metric
- 4 lines: chamber-LT, chamber-WT, research pipeline, code-review pipeline
- Per-pillar quality metric: chamber=SHD, research=indeterminacy-aware
  quality score, code-review=pass@1
- Pattern that earns the claim: all four lines decrease monotonically with
  tightness. If they do not co-decrease, variance reduction is
  domain-specific and the paper says so explicitly.

**Figure 7.2 — conservation-law transfer (two-panel, shared y-axis).**

- y-axis (both panels): fraction of runs where the sub-agent's actual
  budget consumption exceeded its delegated allocation.
- **Left panel (chamber, Planner+Reasoner):** x-axis = realized sub-budget
  split ratio A/(A+B), binned into terciles {Planner-heavy, balanced,
  Reasoner-heavy} from the existing §6.1 runs. One line per chamber (LT,
  WT), ~100 runs per tercile per chamber. Tests whether conservation holds
  across the spectrum of allocation strategies the Planner adopts. **No
  new runs required** — the Planner allocates A:B dynamically per run, and
  we group post-hoc by realized split.
- **Right panel (research pipeline):** x-axis = handoff position
  (handoff 1 = Researcher→Analyzer; handoff 2 = Analyzer→Reporter). Single
  line; data points are violation rates at each handoff aggregated across
  the §7.4 cross-pillar runs.
- Pattern that earns the claim: in **both panels**, violation rate stays
  near zero (the framework's job) with bounded growth.
- The two-panel framing is honest about the fact that "delegation pressure"
  parameterizes differently per domain. Both panels defend the same claim
  — *the framework's conservation laws hold across delegation* — under the
  parameterization natural to each domain.

**Figure 7.3 — runaway-prevention transfer.**

- ECDF (or violin) of total tool calls per run
- Faceted by domain (chamber Planner+Reasoner, code-review Coder↔Reviewer);
  cap-on vs cap-off overlaid in each facet
- Pattern that earns the claim: cap-on truncates cleanly; cap-off has heavy
  right tail in both domains.

### 7.4 Scope of re-running

Not the original "≤10%" — the experiment needs real statistical power.

| Pipeline | Cells | Cost basis | Cost |
|---|---|---|---|
| Research | 25 topics × 5 tightness × 10 seeds = 1250 runs | ~$0.05/run (DeepSeek v4 Flash, multi-agent ~250K tok/run) | ~$62 |
| Code-review | 30 problems × 5 tightness × 10 seeds = 1500 runs | ~$0.015/run (DeepSeek v4 Flash, iterative ~100K tok/run) | ~$23 |
| **Cross-pillar total** | **2750 runs** | — | **~$85** |

Code-review subsamples 30 of the 70 LiveCodeBench problems for cost; the
30 are stratified by difficulty. The cost is now low enough that the
70-problem full set is genuinely cheap (~$53), so this could be
re-evaluated during M6 — kept at 30 for now to preserve the option of
stratified comparison to the COINE paper's own 70-problem analyses.

The model migration is a one-line CLI flag — both pipelines parameterize
`--model` and currently default to `gemini/gemini-2.5-flash-lite`.
Cross-pillar re-runs pass `--model openrouter/deepseek/deepseek-v4-flash`
instead. **§7's cross-pillar runs therefore use a different model than
the COINE paper's original §8 experiments**, which is intentional: the
cross-pillar's purpose is to test transfer of *governance behaviors*
under a single model held constant across both pillars, not to replicate
COINE numerically.

Combined with revised chamber pillar (~$7) and the §6.7 Pro robustness
sweep (~$5): **total experiment cost ~$97.** Real per-run cost is
instrumented via `ResourceUsage.cost_usd` and reported in the paper.

### 7.5 Timeline impact

§9 absorbs the expansion: M6 extends from 2 → 3 weeks, M7 compresses from
3 → 2 weeks. Net calendar is neutral. The compression is feasible only
because COINE feedback (§10) is descoped to "low-information signal,"
freeing M1 calendar that previously waited on the May 25–26 trip.

### 7.6 Why this is the highest-leverage section

If §6 results are clean but §7 doesn't transfer, we have a publishable
causal-discovery paper — but not a contract-framework paper. If §7
transfers, the entire COINE-paper corpus of governance findings is
*retroactively validated* by the chamber ground truth. That is the move
that turns this submission from "extending COINE with another experiment"
into "establishing that contract-framework governance gains are
domain-general."

This is the section reviewers will read most carefully. It deserves a full
section, not a subsection.

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
├── agents.py                 # Random, GreedyIG-lite, LLM-only, LLM+PC, Planner+Reasoner
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

5+ months between today (plan revision date 2026-05-06) and AAMAS submission
(~Oct 1). M1 is **pulled forward to start now** because COINE 2026 is a
15-minute oral talk — not a working session — so its feedback strand is
descoped (see §10). The technical strand of M1 no longer waits on Paphos.

| # | Window | Milestone | Acceptance criterion |
|---|---|---|---|
| M1 | 2026-05-06 → 06-07 | Chamber adapter scaffolding (pulled forward; COINE feedback strand descoped per §10) | `integrations/causalchamber.py` stub committed; `chambers` extra in `pyproject.toml`; failing smoke test exists; §12 Q1+Q2 decisions documented based on a read of existing integrations |
| M2 | 2026-06-08 → 06-21 | Adapter complete + ground-truth scoring functions | Smoke test passes: load `lt/standard` graph, run a fake agent that returns the ground truth, score reports SHD=0 and F1=1 |
| M3 | 2026-06-22 → 07-12 | **Five** baseline agents implemented (3 weeks, was 2) | All five variants (Random, GreedyIG-lite, LLM-only, LLM+PC, Planner+Reasoner) run end-to-end on a single budget cell; produce coherent adjacency-matrix outputs |
| M4 | 2026-07-13 → 07-26 | Pilot sweep | 1 chamber × 3 budgets × 5 variants × 30 seeds = 450 runs; preliminary Pareto curve monotonic; Random sits below LLM variants as sanity check |
| M5 | 2026-07-27 → 08-23 | Full chamber sweep + Pro robustness | All 1800 Flash chamber runs (1500 CONTRACTED + 300 UNCONTRACTED) complete; 1080 are LLM-bearing; **300 DeepSeek v4 Pro robustness runs (§6.7) complete**; results in Parquet; headline 5-line Pareto figure (Figure 6.1) generated with Flash/Pro overlay |
| M6 | 2026-08-24 → 09-13 | **Cross-pillar transfer study** (full section, 3 figures, was subsection) | 2750 LLM-pipeline re-runs at matched tightness; Figures 7.1, 7.2, 7.3 generated; transfer claim supported or refuted with statistical power |
| M7 | 2026-09-14 → 09-27 | Paper extension drafted (compressed 1 week to absorb M6 expansion) | `paper/paper-extended.qmd` (or branch) contains new chamber-pillar + cross-pillar transfer sections; intro and abstract rewritten to reflect two-pillar+bridge structure |
| M8 | 2026-09-28 → 10-XX | Submission polish | All AAMAS formatting requirements met; cover letter cites COINE acceptance; §10 1-pager attached as appendix or sidebar |

Each milestone unblocks the next. M3 grew from 2 → 3 weeks because we now
implement five variants instead of three (Random and GreedyIG-lite are cheap,
but Planner+Reasoner is novel work). M6 grew from 2 → 3 weeks because §7 is
now a full section. M7 compressed from 3 → 2 weeks; this compression is
feasible only because M1 was pulled forward, giving the technical track three
extra weeks of head start.

---

## 10. COINE attendance (descoped from "feedback capture")

COINE 2026 is a **15-minute oral slot, not a working session.** The original
plan treated it as a feedback-capture milestone gating M1; this revision
treats it as **low-information signal** and the technical work proceeds in
parallel.

Light-touch action: a 1-page note at `docs/coine_feedback.md` by 2026-06-01
documenting any genuinely surprising audience reaction or hallway question.
If the note is empty, that is fine — no calendar slips on its account. The
expected content is closer to "interesting question from X about Y" than
"structured feedback we have to address before submitting."

If — and only if — the COINE audience surfaces a substantive flaw in the
framework itself (R5), escalate immediately and reassess the AAMAS timeline.
Otherwise, proceed.

This descoping is what frees the entire May 6 → June 7 M1 window for the
technical strand (chamber adapter scaffolding), which in turn enables M7's
1-week compression in §9.

---

## 11. Risks and mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Causal-discovery agent is a new paradigm; agent implementations are harder than expected | Medium | High (delays M3 → M5) | Start M3 with Random + LLM-only (simplest); GreedyIG-lite second; LLM+PC third; Planner+Reasoner last. **Floor: LLM+PC + Planner+Reasoner + Random** (Pareto floor + main hybrid + multi-agent claim). GreedyIG-lite and LLM-only are stretch if M3 slips. |
| R2 | Calibrated CI coverage too hard for LLM-based agents | Medium | Medium (weakens calibration claim) | Soften to "edge confidences" (no formal coverage guarantee) in v1; add bootstrap-based coverage in v2 if time allows. The SHD/F1 results stand independently. |
| R3 | AAMAS reviewers reject because the chamber pillar isn't on real hardware | Low | Medium | Cite Gamella et al. — the offline data *is* real-hardware measurements, just pre-recorded. The 59 experiments per chamber are the same data their own published validation rests on. |
| R4 | `causalchamber` package breaks (yanked numpy 2.4.0 already a yellow flag) | Low | Low | Pin a working version range in the `chambers` extra. Vendor the offline datasets to our own storage if upstream becomes unreliable. |
| R5 | COINE attendance reveals a substantive flaw in the framework itself | Low | High | Address in M1; if it's framework-level (not just experiment-level), reassess whether AAMAS is reachable on the original timeline or whether we need to push to ECAI 2027 only. |
| R6 | AAMAS 2027 location announced in late May 2026 lands somewhere we can't travel to | Medium | Medium | Already mitigated by parallel ECAI 2027 (Athens, confirmed) plan. AAMAS becomes optional rather than primary if location is bad. |
| R7 | Compute cost overruns | Effectively zero | Negligible | Budget is **~$97** (chamber $7 + cross-pillar $85 + Pro robustness $5), all at DeepSeek v4 Flash/Pro on OpenRouter. Cost is now noise relative to engineering time; even 10× overrun is absorbable. Actual cost instrumented via `ResourceUsage.cost_usd` and reported in the paper. |
| R8 | Planner+Reasoner conservation law shows measurable but small effect over LLM+PC | Medium | Low | Report effect-size with CI rather than binary effect/no-effect framing. A small positive effect is a finding ("delegation is roughly free"); a small negative effect is also a finding ("delegation has measurable cost"). Either is publishable; the paper's framing rotates accordingly. |

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
