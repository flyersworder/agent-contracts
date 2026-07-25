# Causal Chamber Validation: Mainstream-Venue Extension Plan

**Status**: In execution (M4 ✅ complete)
**Created**: 2026-05-03
**Last status update**: 2026-05-18 (M4b PILOT COMPLETE, scope revisions below)
**Owner**: qingye
**Target venues**: AAMAS 2027 (primary, ~Oct 2026 deadline), ECAI 2027 Athens (secondary, ~Apr 2027 deadline)
**Prerequisite**: COINE 2026 oral presentation (Paphos, May 25–26, 2026) ✅ accepted

---

> ## 📍 POST-M4B SCOPE UPDATE (2026-05-18)
>
> **M4b pilot completed 2026-05-18** with dramatically stronger results than the
> plan originally anticipated:
>
> - **442/450 ok-cells** (8 timeouts, all `planner_reasoner k=59`)
> - **LLM-only at k/M=1.00: SHD=26, F1=0.75** vs every other variant clustered
>   at SHD≈53-57, F1≈0.40-0.42
> - M4 acceptance criteria PASS at 30 seeds (was a single-seed hypothesis from
>   May 14)
> - See `runs/m4-pilot-figs/pareto_shd.png` and `pareto_f1.png`
>
> **Scope revisions triggered by the strength of these findings**:
>
> 1. **M5 budget axis trimmed from 5 → 3 levels.** §6.1 originally specified
>    `k/M ∈ {0.10, 0.25, 0.50, 0.75, 1.00}`. With M4b showing strong
>    monotonicity at 3 budgets (0.10, 0.51, 1.00), the intermediate budgets
>    (0.25, 0.75) add curve-shape refinement without changing the headline
>    claim. **Saves ~33% of M5 wall time.** Detailed sub-rationale in §6.1
>    callout.
> 2. **M6 (cross-pillar transfer study) is now optional, not required.** §7
>    was written assuming chamber-pillar findings would be modest enough to
>    need the cross-pillar bridge for impact. M4b's dramatic effect lets the
>    chamber pillar stand alone as a publishable contribution. M6 becomes a
>    journal-extension candidate rather than a M7-blocker. Detailed
>    sub-rationale in §7 callout.
> 3. **M5 essentials unchanged**: WT chamber sweep + UNCONTRACTED baselines +
>    DeepSeek v4 Pro robustness. These remain non-negotiable — WT establishes
>    external validity beyond LT, UNCONTRACTED is the framework paper's
>    central claim, Pro robustness tests whether Flash dominance generalizes
>    across model scale.
>
> See §6.1 and §7 for in-section detail; §9 milestones table updated.

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
| Run counts | 25 research topics × 2 conditions; 140 code-review trials (70 problems × 2) | **~4295 new LLM runs**: 1380 chamber (1080 Flash + 300 Pro) + 2915 cross-pillar (165 calibration + 2750 re-runs); see §6.1, §6.7, §7 |
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
| 2 | **GreedyIG-lite** | Single-agent | Non-LLM, principled | Reference line for "what a principled non-LLM achieves." Picks the next intervention greedily by target-coverage as an I-MEC-reduction proxy (Hauser & Bühlmann 2014: single-target interventions on previously-unperturbed variables strictly reduce the interventional Markov equivalence class for linear-Gaussian SCMs). Full Bayesian variance-reduction version deferred to v2 / journal extension. **LT-only**: WT's random-walk + regime-jump experimental design has no discrete intervention targets, so target-coverage has no structure to exploit; variant 2 is skipped in WT cells of §6.1's sweep, and the WT Pareto plot in §5.3 has 4 lines instead of 5. |
| 3 | **LLM-only** | Single-agent | LLM throughout | Pure in-context-learning (ICL) agent (**DeepSeek v4 Flash via OpenRouter**, accessed through the framework's LiteLLM integration). LLM picks each intervention and emits final adjacency matrix as a directed-edge JSON map. (M3 implementation note: edge confidences were dropped — the implemented prompt asks for hard {0, 1} edges, matching the SHD scorer's binary input. Soft confidences are deferred to v2 / journal extension.) |
| 4 | **LLM+PC** | Single-agent | LLM-orchestrated classical | LLM plans intervention sequence; classical PC algorithm infers the graph from the resulting data. |
| 5 | **Planner+Reasoner** ⭐ | **Multi-agent** | LLM throughout, two roles | Planner agent picks interventions under sub-budget A; Reasoner agent picks additional interventions under sub-budget B; classical PC infers the graph from the union of A+B experiments. **Conservation law: A + B ≤ total intervention budget**, enforced by `ContractingCapability.create_subcontract` on the per-tool axis. Exercises the framework's delegation primitives. (M3 implementation note: both phases use selection so the §5.3 Pareto comparison vs variant 4 isolates the *delegation cost*, not the inference-method choice — if the Reasoner emitted the graph instead, the variant would confound delegation overhead with LLM-vs-PC inference quality.) |

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
  LT: 1 chamber × 5 budget levels × 5 variants × 30 seeds = 750 runs
  WT: 1 chamber × 5 budget levels × 4 variants × 30 seeds = 600 runs
  (WT skips variant 2 — GreedyIG-lite is LT-only per §5.1)
  Total CONTRACTED:                                       1350 runs
  Of which LLM-bearing: 3 variants × 300 cells          =  900 LLM runs

UNCONTRACTED baseline:
  LT: 1 chamber × 5 variants × 30 seeds = 150 runs
  WT: 1 chamber × 4 variants × 30 seeds = 120 runs (variant 2 skipped)
  Total UNCONTRACTED:                     270 runs
  Of which LLM-bearing: 3 variants × 60 cells = 180 LLM runs

Total: 1620 runs; LLM-bearing: 1080 (unchanged — variant 2 is non-LLM)
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
carries **five lines for LT, four for WT** (variant 2 / GreedyIG-lite
is LT-only per §5.1). Each line has an explicit interpretive role:

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

> **📍 Post-M4b scope revision (2026-05-18):** budget axis trimmed from 5 → 3
> levels. The original 5-level grid (0.10, 0.25, 0.50, 0.75, 1.00) was sized
> to detect *modest* effect sizes that might require intermediate points to
> resolve a Pareto curve. M4b's 30-seed data on LT shows monotonicity *and*
> dramatic separation at just 3 budgets (0.10, 0.51, 1.00) — adding 0.25 and
> 0.75 refines curve shape without changing the headline claim that
> LLM-only-with-summary dominates. The trimmed grid below saves ~33% of M5
> wall time and ~$3.50 of OpenRouter spend. If after M5 launch we observe
> curves that need intermediate-budget resolution to be interpretable, the
> 0.25/0.75 cells can be added incrementally via the checkpoint-resume
> mechanism (M4c) without re-running anything completed.

The headline experiment grid uses **menu-fraction budgets** so curves are
comparable across chambers despite different absolute menu sizes
(LT M=59, WT M=28). Two run families:

**CONTRACTED Pareto sweep** (the headline figure):

```
2 chambers          (lt with standard config; wt with standard config)
× 3 budget levels   (k/M ∈ {0.10, 0.50, 1.00})    [revised from 5; see callout]
                    → LT: k ∈ {6, 30, 59}
                    → WT: k ∈ {3, 14, 28}
× agent variants    LT: 5 (Random, GreedyIG-lite, LLM-only, LLM+PC, Planner+Reasoner)
                    WT: 4 (variant 2 / GreedyIG-lite skipped — LT-only per §5.1)
× 30 seeds          (statistical power)
= LT: 1 × 3 × 5 × 30 =  450 runs (already done — M4b, 2026-05-18)
+ WT: 1 × 3 × 4 × 30 =  360 runs (M5 scope)
                     =  810 runs total CONTRACTED
   of which 540 are LLM-bearing (3 LLM variants × 180 cells; variant 2
   is non-LLM and its WT-skip leaves the LLM-bearing count unchanged)
```

**UNCONTRACTED baseline** (single point per agent, for §6.2 comparison):

```
LT: 1 × 5 × 30 = 150 runs
WT: 1 × 4 × 30 = 120 runs (variant 2 skipped per §5.1)
             = 270 runs
   of which 180 are LLM-bearing
```

**Total under revised scope: 1080 runs (720 LLM-bearing)** [was 1620 / 1080
under the original 5-budget grid; trimmed per the §6.1 callout above].

Of the 1080: 450 are already complete (M4b LT pilot, 2026-05-18); the
remaining 630 are M5 scope (360 WT CONTRACTED + 270 UNCONTRACTED across
both chambers).

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

- 270 Pro runs at cell-grid scope (LT: 3 budgets × 5 variants × 10 seeds = 150;
  WT: 3 budgets × 4 variants × 10 seeds = 120 — variant 2 LT-only): ~$5

**Cost totals:**

| Component | Runs | Total |
|---|---|---|
| Chamber pillar (Flash) | 1080 LLM-bearing | ~$7 |
| §7.2 cross-pillar tightness calibration (Flash UNCONTRACTED) | 165 | ~$6 |
| Cross-pillar research re-runs (Flash) | 1250 | ~$62 |
| Cross-pillar code-review re-runs (Flash) | 1500 | ~$23 |
| Pro robustness sweep (cell-grid) | 270 | ~$5 |
| **Grand total** | **4265 LLM runs** | **~$103** |

CPU cost on existing development hardware is negligible — PC, GreedyIG-lite
linear-Gaussian fitting, and selection are O(n³) at worst on a 38-node
graph (seconds per fit). The project's compute budget ceases to be a
meaningful constraint; engineering time is the binding resource. Actual
per-run cost is instrumented via `ResourceUsage.cost_usd` and reported
in the paper, so the headline ~$103 figure is a budget ceiling rather than
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
- variants: 5 on LT (Random, GreedyIG-lite, LLM-only, LLM+PC, Planner+Reasoner),
  4 on WT (variant 2 / GreedyIG-lite is LT-only per §5.1)
- 10 seeds (reduced from 30 since Pro is for capacity-axis comparison,
  not primary statistical claims)
- = LT: 3 × 5 × 10 = 150 + WT: 3 × 4 × 10 = 120 = **270 Pro runs at ~$5**

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

> **📍 Post-M4b scope revision (2026-05-18):** this section is **now optional**
> rather than load-bearing. The original framing below ("without it, chamber
> results stand as 'interesting causal-discovery findings'…") assumed M4
> would produce a *modest* effect that would need cross-pillar bridging for
> impact. M4b delivered a *dramatic* effect (LLM-only at k/M=1.00 hits SHD=26
> / F1=0.75, with every other variant clustered at SHD≈53-57 / F1≈0.40-0.42).
> The chamber-pillar standalone is now strong enough to carry a submission
> on its own.
>
> **Two viable paths for §7 going forward**:
>
> 1. **Defer §7 to a journal extension.** Submit the chamber-pillar to
>    AAMAS 2027 as a single-pillar contribution. Cross-pillar transfer
>    becomes the natural next paper. Reduces M6 to ~0 and unblocks M7
>    drafting ~3 weeks earlier.
> 2. **Execute §7 as originally planned.** Two-pillar submission stays the
>    target. Worth doing if (a) reviewer feedback on a single-pillar
>    submission would likely demand it anyway, or (b) the chamber finding
>    is novel enough that a venue values both pillars in one paper.
>
> The decision below is now an *option*, not a *requirement*. The original
> §7 text below describes execution plan for path (2); path (1) simply
> skips it.

> **📍 DECIDED (2026-07-25): path (1).** Cross-pillar transfer is deferred to a
> journal extension, and **M6 becomes the loop-vs-graph topology benchmark**
> specified in §7.7. The trigger was external: "loop engineering" (named
> 2026-06-07) and "graph engineering" (named ~2026-07-18) put multi-agent
> topology on the field's agenda, and the critical reviews of both note there
> is no comparative benchmark. Our M4b pilot already contains two of the three
> arms one would need, so the timelier contribution is the benchmark rather
> than the transfer study. §7.1–§7.6 below stand as the journal-extension plan.

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
distribution* under the **DeepSeek v4 Flash model used by §7's cross-pillar
re-runs** (not the COINE paper's incumbent Gemini 2.5 Flash Lite):

- At the start of M6, run a small DeepSeek-Flash UNCONTRACTED calibration
  sweep: 3 seeds × 25 topics for research (75 runs at ~$4) and 3 seeds × 30
  problems for code-review (90 runs at ~$1.50). Total ~165 calibration
  runs at ~$6, ~half-day compute.
- Derive the empirical distribution of total tool calls (research) or
  total iterations (code-review) from these DeepSeek-Flash UNCONTRACTED
  runs.
- Tightness level *t* corresponds to the *t*-th percentile of that
  distribution. *t = 0.10* means "limit the agent to the 10th percentile of
  what it would naturally use *under DeepSeek v4 Flash*."

The COINE paper's existing UNCONTRACTED runs are not reusable for this
calibration: they were collected under Gemini 2.5 Flash Lite, and natural
usage distributions are model-dependent (verbosity, tool-call propensity,
reasoning-loop length all differ across models). Reusing Gemini-derived
percentiles to set caps for DeepSeek runs would defeat the percentile-
matching's purpose, which is to control for what *the current agent*
naturally does.

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
| §7.2 calibration | 75 (research) + 90 (code-review) = 165 UNCONTRACTED runs | DeepSeek v4 Flash, mixed | ~$6 |
| Research re-runs | 25 topics × 5 tightness × 10 seeds = 1250 runs | ~$0.05/run (DeepSeek v4 Flash, multi-agent ~250K tok/run) | ~$62 |
| Code-review re-runs | 30 problems × 5 tightness × 10 seeds = 1500 runs | ~$0.015/run (DeepSeek v4 Flash, iterative ~100K tok/run) | ~$23 |
| **Cross-pillar total** | **2915 runs** | — | **~$91** |

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
sweep (~$5): **total experiment cost ~$103.** Real per-run cost is
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

### 7.7 M6 (revised): the loop-vs-graph topology benchmark

**Research question.** At matched total budget, does graph topology help or
hurt?

The chamber budget `k/M` is the fraction of the 59 available experiments an
agent may run, so the conserved resource is *experiment selections* — the
per-tool budget the framework's delegation graph conserves (whitepaper §4.6).

**Three arms:**

```
Arm 1  LOOP (llm_only)          root ──────────────► 59 experiments
       already run in M4b

Arm 2  CHAIN (planner_reasoner) root ──► planner ──► reasoner
       already run in M4b              (delegated split)

Arm 3  FAN-IN (new)             root ──┬─► scout_a (⌈k/2⌉ exp) ─┐
       needs DAG conservation           └─► scout_b (⌊k/2⌋ exp) ─┴─► aggregator (0 exp)
                                        scouts always sum to exactly k
```

The split rule generalizes to every budget level: `scout_a` gets `⌈k/2⌉`,
`scout_b` gets `⌊k/2⌋`, the aggregator gets `0`. At `k/M = 0.10` (`k = 6`) that
is 3 and 3; at `0.50` (`k = 30`), 15 and 15. The aggregator spends tokens on the
merge but no experiment selections, so the conserved resource stays matched.

**Grant the aggregator an explicit zero**, not an omitted key. Per-tool limits
have no deny-by-default semantics, so an omitted key materializes as
*unconstrained*, not zero — an explicit `per_tool={"exp": 0}` is what makes the
matched-budget control enforceable rather than merely detectable after the fact.

**Matrix.** LT chamber, `k/M ∈ {0.10, 0.50, 1.00}`, 30 seeds. Arms 1 and 2
already exist in `runs/m4-pilot.parquet` at the same model, seeds, budgets, and
scoring, so **new compute is 90 cells** (~$1–2, hours not days). Arm 3 registers
as a new variant in the M4a AgentSpec registry. WT is a stretch goal contingent
on M5's WT data.

**Reuse-validity guard.** Before trusting the M4b arms, re-run a 5-cell subset
of `llm_only` (1 budget × 5 seeds) and confirm it reproduces the May numbers
within noise. If the harness drifted, we find out for 5 cells rather than in the
final figure.

**Pre-registered hypotheses:**

| | Hypothesis | Status |
|---|---|---|
| H-A | Chain underperforms loop at matched budget | Already supported — M4b: F1 0.397 vs 0.75 |
| H-B | Fan-in recovers delegation cost via exploration diversity | **Open — this is the experiment** |
| H-C | Conservation compliance is 100% across all arms | The governance claim |

H-B is deliberately the *strongest* form of the opposing claim: two scouts
select different experiments, so their union may cover more causal structure
than one agent's 59 sequential picks — the ensemble effect underlying
adversarial-verify and council patterns. Against it, each scout sees less data
and the aggregator may introduce merge error. Testing the strongest version is
what makes a negative result credible rather than a strawman.

H-C is what keeps this a contracting result rather than a causal-discovery
topology benchmark — the same reasoning by which the UNCONTRACTED baselines are
non-negotiable for M5. Note the scope from whitepaper §4.6: verification
certifies `B(root) + Σ refunds` once a node is abandoned, and M4b's timeouts are
exactly the abandonment case, so H-C should be reported as a certification
claim, not a containment claim.

**Metrics and figures.** Per cell: SHD, F1, plus `tokens_in` / `tokens_out`,
experiments used, `n_llm_calls`, conservation compliance, `wall_time_seconds`,
`status`. Every one of these is already a column in `runs/m4-pilot.parquet` —
the sweep needs no new instrumentation. Figures: (a) SHD
and F1 versus budget, one line per topology arm; (b) a "topology tax" plot
showing the delta against the single loop at matched budget. Following R8's
framing, results are effect sizes with confidence intervals rather than binary
effect/no-effect — fan-in winning identifies *when* graph structure pays, which
neither camp has established; fan-in losing is a strong negative result against
the field's central claim at 270 cells and 30 seeds. The outcome to avoid is not
a bad result but an underpowered one.

**Prerequisite.** The DAG conservation capability shipped in v0.4.0
(`core/delegation_graph.py`). Nothing else blocks this sweep.

> **Correction (2026-07-25).** An earlier revision of this section listed
> "wire up `ResourceUsage.add_iteration()`" as a gap to close before M6, on the
> grounds that the per-cell iteration metric would otherwise read zero. That was
> wrong on two counts. The chamber pipeline calls `litellm.completion` directly
> through its own `_CountingLLM` (`orchestrator.py:239`) and never touches a
> `ResourceMonitor`, so wiring the framework's integrations would not reach these
> cells at all. And the metric already exists: `n_llm_calls` is populated for
> every LLM-variant ok-cell in the M4b pilot (6/30/59-60 across the three
> budgets; null for `random` and `greedy_ig`, correctly, since they issue no LLM
> calls). The framework's `iterations` axis is a separate concern that M6 does
> not depend on — see the note under §9.

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

| # | Window (plan) | Actual | Milestone | Acceptance criterion |
|---|---|---|---|---|
| M1 | 2026-05-06 → 06-07 | ✅ 2026-05-06 | Chamber adapter scaffolding | `integrations/causalchamber.py` stub committed; `chambers` extra in `pyproject.toml`; failing smoke test exists; §12 Q1+Q2 decisions documented |
| M2 | 2026-06-08 → 06-21 | ✅ 2026-05-06 | Adapter complete + ground-truth scoring functions | Smoke test passes: load `lt/standard` graph, run a fake agent that returns the ground truth, score reports SHD=0 and F1=1 |
| M3 | 2026-06-22 → 07-12 | ✅ 2026-05-08 | **Five** baseline agents implemented | All five variants (Random, GreedyIG-lite, LLM-only, LLM+PC, Planner+Reasoner) run end-to-end on a single budget cell; produce coherent adjacency-matrix outputs |
| M4 | 2026-07-13 → 07-26 | ✅ **2026-05-18** | Pilot sweep + M4c checkpointing | LT × 3 budgets × 5 variants × 30 seeds = 450 runs; **442 ok / 8 timeouts** (1.8% error rate, all planner_reasoner k=59); acceptance criteria PASS (Pareto monotonic, Random dominated). M4c JSONL checkpoint sidecar landed (commit `856beb8`). Headline: LLM-only at k/M=1.00 hits SHD=26 / F1=0.75. |
| **M5** | 2026-07-27 → 08-23 | **next (target ~2-3 weeks from 05-19)** | **Trimmed M5**: WT sweep + UNCONTRACTED + Pro robustness | **Revised scope** (see §6.1 callout): 360 WT CONTRACTED + 270 UNCONTRACTED + 270 Pro robustness = **900 new runs** (LT 450 already done). Headline Pareto figure (Figure 6.1: 5 lines for LT, 4 for WT) generated with Flash/Pro overlay. Recommend running on VPS (`173.212.217.40`, provisioned 2026-05-18). |
| M6 | 2026-08-24 → 09-13 | **REVISED 2026-07-25** | **Loop-vs-graph topology benchmark (§7.7)** | Cross-pillar transfer (§7.1-7.6) deferred to a journal extension. M6 is now the three-arm topology study: loop / chain / fan-in at matched budget. **90 new cells** (arms 1-2 already in `runs/m4-pilot.parquet`), ~$1-2. Prerequisite shipped in v0.4.0 (`core/delegation_graph.py`); no open blockers. |

**Framework note, not an M6 blocker.** `ResourceConstraints.iterations` is
honored only by the Google ADK integration (mapped to `max_llm_calls`) and the
Claude Agent SDK integration (mapped to `max_turns`). LiteLLM, LangChain, and
LangGraph neither track nor enforce it, and `core/contract.py`'s docstring
claim that it maps to LangGraph's `recursion_limit` was inaccurate (corrected
2026-07-25). Closing that gap would benefit library users; it has no bearing on
the chamber sweeps, which bypass those integrations entirely.
| M7 | 2026-09-14 → 09-27 | Paper extension drafted | `paper/paper-extended.qmd` (or branch) contains new chamber-pillar section; intro and abstract rewritten. If M6 executed, also cross-pillar transfer section. |
| M8 | 2026-09-28 → 10-XX | Submission polish | All AAMAS formatting requirements met; cover letter cites COINE acceptance; §10 1-pager attached as appendix or sidebar |

**Schedule status as of 2026-05-18**: M1-M4 finished **~9 weeks early**
relative to the original plan (M4 closed May 18 vs planned Jul 26). That
headroom absorbs M5 scope-trim discussions, enables the VPS migration, and
makes M6 optionality a *deliberate choice* rather than a *deadline-forced
cut*.

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
| R7 | Compute cost overruns | Effectively zero | Negligible | Budget is **~$103** (chamber $7 + cross-pillar calibration $6 + cross-pillar re-runs $85 + Pro robustness $5), all at DeepSeek v4 Flash/Pro on OpenRouter. Cost is now noise relative to engineering time; even 10× overrun is absorbable. Actual cost instrumented via `ResourceUsage.cost_usd` and reported in the paper. |
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
