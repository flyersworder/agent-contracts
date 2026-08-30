"""Sweep orchestrator for the chamber pillar.

Owns the AgentSpec registry (the single source of truth for which
variants exist, which chambers each is compatible with, and what kwargs
each accepts), the per-cell runner `run_cell`, and the full-sweep
runner `run_sweep`. M4's `run_experiment.py` CLI is a thin wrapper
around `run_sweep`; the orchestrator itself has no CLI dependencies
and is fully testable with mocked LLM via the `llm` parameter.

Design (resolves the four open questions from M3 final review):

1. **Agent dispatch** (M3-review #5): AgentSpec registry. Each agent
   declares its name, callable, chamber compatibility, kwargs schema,
   and whether it accepts an injectable LLM. The orchestrator iterates
   the registry; per-cell dispatch is data-driven, not if/elif.

2. **Compatibility API** (M3-review #6): `AgentSpec.chambers` is a
   tuple of chamber-id strings the agent supports. The orchestrator
   filters BEFORE invoking — incompatible cells produce a "skipped"
   RunRecord, never a NotImplementedError mid-sweep. (The
   NotImplementedError raise inside the agent remains as a defensive
   double-check, but the registry is the contractual API.)

3. **Metadata aggregation** (M3-review #4): each cell installs a
   logging handler scoped to `evaluation.chamber_pipeline.inference`
   that captures PC-degeneracy warnings into a per-cell counter.
   Aggregated into the RunRecord's `n_pc_degeneracies` field — no
   log-scraping needed at analysis time.

4. **Sweep harness**: serial nested for-loop (chamber → budget →
   agent → seed). Parallelism deferred to M5 if M4 wall-time is a
   problem; the simpler design is easier to debug at pilot scale
   (450 cells) and the LLM API is the bottleneck anyway.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar

from agent_contracts.core.delegation import ConservationViolationError
from agent_contracts.integrations.causalchamber import (
    ChamberId,
    ConfigId,
    create_contracted_chamber_agent,
)

from . import inference as inference_module
from .agents import (
    coverage_max_agent,
    coverage_max_ms_agent,
    coverage_min_agent,
    coverage_min_ms_agent,
    critique_agents,
    fan_in_agents,
    greedy_ig_lite_agent,
    llm_only_agent,
    llm_pc_agent,
    one_shot_agent,
    planner_reasoner_agents,
    random_agent,
    team_agents,
    team_varsplit_agents,
    uncontracted_agent,
)
from .inference import pc_call_defaults, runtime_fingerprint
from .results import RunRecord, now_iso
from .scoring import f1_edges, shd

# Type alias for the LLM callable threaded through agents that accept it.
# Mirrors agents.py's LLMCallable. Production callers pass
# `litellm.completion`; tests pass FakeLLM.
LLMCallable = Callable[..., Any]


@dataclass(frozen=True)
class AgentSpec:
    """Descriptor for one chamber-pillar agent variant.

    The single source of truth for what a variant is, where it can
    run, and what kwargs it accepts. The orchestrator dispatches
    agents through this descriptor — callers don't import individual
    agent functions.

    Attributes:
        name: Short variant name (registry key). Matches the names
            used in `RunRecord.agent_name` and the §6.5 figure legend.
        run: The agent callable. Must match
            `agent(adapter, **kwargs) -> pd.DataFrame`.
        chambers: Tuple of chamber IDs this agent is compatible with.
            The orchestrator filters cells using this — incompatible
            chambers produce skipped RunRecords without ever invoking
            the agent. Tracks plan §5.1 footnotes (e.g.,
            GreedyIG-lite is LT-only: chambers=("lt",)).
        accepts_llm: True iff the agent accepts an `llm=` keyword.
            Lets the orchestrator decide whether to pass the
            injectable LLM callable (real `litellm.completion` or
            FakeLLM in tests).
        kind: Coarse classification — "non_llm" / "llm_single" /
            "llm_multi". Used by the orchestrator to decide whether
            to track LLM-specific metadata (`n_llm_calls`).
        extra_kwargs: Per-variant required kwargs that aren't
            seed/pc_alpha/llm. E.g., `planner_reasoner_agents`
            requires `planner_budget` and `reasoner_budget`. The
            orchestrator computes these per-cell from the cell's
            total budget.
    """

    name: str
    run: Callable[..., Any]
    chambers: tuple[ChamberId, ...]
    accepts_llm: bool = False
    kind: str = "non_llm"  # "non_llm" | "llm_single" | "llm_multi"
    extra_kwargs: tuple[str, ...] = ()
    # --- M6 ladder self-description -------------------------------------
    # These make an arm describe its own wiring instead of being special-cased
    # by name in four places. A new rung that forgets its roles is not a
    # silent fallthrough to the plain-role budget -- which would under-fund a
    # targeted scout 6.5x and produce conservation violations that read as
    # real overruns -- it simply is not a ladder arm.
    scout_roles: tuple[str, str] | None = None
    negotiation_rounds: int = 0
    # True for the UNCONTRACTED control: the adapter is built with the menu
    # size as its intervention cap instead of `k`. That is a physical limit
    # (only so many distinct experiments exist), not a governance bound --
    # building it with `k` would make "uncontracted" contracted after all,
    # and the arm would silently measure nothing.
    ignores_budget: bool = False
    # `MappingProxyType` so a caller cannot mutate the registry through
    # `get_spec(...).static_kwargs[k] = v`, which persisted globally.
    static_kwargs: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def is_ladder_arm(self) -> bool:
        return self.scout_roles is not None

    def is_compatible(self, chamber: ChamberId) -> bool:
        """True iff this agent runs on the given chamber.

        The orchestrator gates dispatch on this — never invokes the
        agent for an incompatible chamber. Defensive double-check
        inside the agent (NotImplementedError) remains, but this is
        the contractual API.
        """
        return chamber in self.chambers


# The five plan §5.1 variants. Edit this list to add/remove variants
# from sweeps; the orchestrator picks them up automatically.
#
# Compatibility per plan §5.1:
#   - Random, LLM-only, LLM+PC, Planner+Reasoner: both chambers (LT, WT)
#   - GreedyIG-lite: LT only (WT's experimental design has no discrete
#     intervention targets — see plan §5.1 row 2 footnote and the
#     NotImplementedError in `agents.greedy_ig_lite_agent`)
AGENT_REGISTRY: tuple[AgentSpec, ...] = (
    AgentSpec(
        name="random",
        run=random_agent,
        chambers=("lt", "wt"),
        accepts_llm=False,
        kind="non_llm",
    ),
    AgentSpec(
        name="coverage_max",
        run=coverage_max_agent,
        # LT only: the taxonomy these arms are built on parses
        # `uniform_<variable>_<strength>`, which WT names do not follow.
        chambers=("lt",),
        accepts_llm=False,
        kind="non_llm",
    ),
    AgentSpec(
        name="coverage_min",
        run=coverage_min_agent,
        chambers=("lt",),
        accepts_llm=False,
        kind="non_llm",
    ),
    AgentSpec(
        name="coverage_max_ms",
        run=coverage_max_ms_agent,
        chambers=("lt",),
        accepts_llm=False,
        kind="non_llm",
    ),
    AgentSpec(
        name="coverage_min_ms",
        run=coverage_min_ms_agent,
        chambers=("lt",),
        accepts_llm=False,
        kind="non_llm",
    ),
    AgentSpec(
        name="greedy_ig_lite",
        run=greedy_ig_lite_agent,
        chambers=("lt",),  # WT skipped per plan §5.1
        accepts_llm=False,
        kind="non_llm",
    ),
    AgentSpec(
        name="llm_only",
        run=llm_only_agent,
        chambers=("lt", "wt"),
        accepts_llm=True,
        kind="llm_single",
    ),
    AgentSpec(
        name="llm_pc",
        run=llm_pc_agent,
        chambers=("lt", "wt"),
        accepts_llm=True,
        kind="llm_single",
    ),
    AgentSpec(
        name="planner_reasoner",
        run=planner_reasoner_agents,
        chambers=("lt", "wt"),
        accepts_llm=True,
        kind="llm_multi",
        extra_kwargs=("planner_budget", "reasoner_budget"),
    ),
    # ---- UNCONTRACTED control -------------------------------------------
    # The framework's own comparison, absent from the chamber pillar until
    # now: every other arm here is contracted, so nothing measured what
    # governance costs. Matches the definition used by the research and
    # code-review pipelines -- no budget enforcement at all.
    AgentSpec(
        name="uncontracted",
        run=uncontracted_agent,
        chambers=("lt", "wt"),
        accepts_llm=True,
        kind="llm_single",
        ignores_budget=True,
    ),
    # ---- M6 coordination ladder -------------------------------------------
    # Rungs 1, 2 and 4. Rungs 0 (`llm_pc`) and 3 (`planner_reasoner`) are
    # reused from the M4b pilot unchanged, which is why nothing above this
    # comment may move.
    # Rung -1: the no-history control. One call for the whole budget, so the
    # running record the loop accumulates is absent rather than divided. Every
    # multi-agent rung SPLITS that record; without this arm the ladder prices
    # the cost of dividing a resource whose value was never established.
    AgentSpec(
        name="one_shot",
        run=one_shot_agent,
        chambers=("lt", "wt"),
        accepts_llm=True,
        kind="llm_single",
    ),
    # Executor-evaluator. The only multi-agent shape here where the second
    # agent takes NO share of the budget -- it advises, the proposer decides.
    # Three LLM calls regardless of k, so also the cheapest multi-agent arm by
    # a wide margin. `scout_roles` stays None: it is not a two-scout arm and
    # must not be handed the fan-in calibration.
    AgentSpec(
        name="critique",
        run=critique_agents,
        chambers=("lt", "wt"),
        accepts_llm=True,
        kind="llm_multi",
    ),
    AgentSpec(
        name="fan_in_homog",
        run=fan_in_agents,
        chambers=("lt", "wt"),
        accepts_llm=True,
        kind="llm_multi",
        extra_kwargs=("scout_a_budget", "scout_b_budget"),
        scout_roles=("plain", "plain"),
    ),
    AgentSpec(
        name="fan_in_spec",
        run=fan_in_agents,  # same function; `differentiate` is what differs
        chambers=("lt", "wt"),
        accepts_llm=True,
        kind="llm_multi",
        extra_kwargs=("scout_a_budget", "scout_b_budget"),
        scout_roles=("broad", "targeted"),
        static_kwargs=MappingProxyType({"differentiate": True}),
    ),
    # Ablation of rung 1, not a rung of the ladder. Identical wiring to
    # `fan_in_homog` except the aggregator's response SELECTS the pooled set
    # instead of being discarded. Exists to answer the review "your negative
    # fan-in result is an artifact of a null aggregator" with a measurement.
    AgentSpec(
        name="fan_in_agg",
        run=fan_in_agents,
        chambers=("lt", "wt"),
        accepts_llm=True,
        kind="llm_multi",
        extra_kwargs=("scout_a_budget", "scout_b_budget"),
        scout_roles=("plain", "plain"),
        static_kwargs=MappingProxyType({"honor_aggregator": True}),
    ),
    AgentSpec(
        name="team",
        run=team_agents,
        chambers=("lt", "wt"),
        accepts_llm=True,
        kind="llm_multi",
        extra_kwargs=("scout_a_budget", "scout_b_budget"),
        scout_roles=("plain", "plain"),
        negotiation_rounds=2,
    ),
    AgentSpec(
        name="team_varsplit",
        run=team_varsplit_agents,
        # LT only: the variable partition reads the LT menu taxonomy, which
        # WT names do not follow.
        chambers=("lt",),
        accepts_llm=True,
        kind="llm_multi",
        extra_kwargs=("scout_a_budget", "scout_b_budget"),
        # Identical to `team` so `_ladder_calibration` resolves the same
        # per-role and per-budget figures. The arm differs only in how pools
        # are partitioned, which costs no extra LLM call, so its budget
        # provisioning must not differ either.
        scout_roles=("plain", "plain"),
        negotiation_rounds=2,
    ),
)


def get_spec(name: str) -> AgentSpec:
    """Look up an AgentSpec by name. KeyError on unknown name."""
    for spec in AGENT_REGISTRY:
        if spec.name == name:
            return spec
    raise KeyError(
        f"Unknown agent name: {name!r}. Available: {sorted(s.name for s in AGENT_REGISTRY)}"
    )


class SweepConfigurationError(ValueError):
    """A mis-configured sweep, as distinct from a cell that failed at runtime.

    `run_cell` deliberately converts every in-cell exception into a
    `status="error"` record so one bad cell cannot abort a 20-hour sweep. That
    is right for runtime faults and wrong for configuration faults: an
    uncalibrated budget or an unpinned model is not a cell that failed, it is
    a sweep that should never have started. Swallowed, it presents as an error
    RATE -- and since `done_cell_keys` excludes errored cells so they can be
    retried, every resume re-attempts all of them, forever, while the message
    naming the fix scrolls past in a truncated `error_message` field.

    Raised before any budget is spent and re-raised through both sweep paths.
    """


# ---------------------------------------------------------------------------
# PC-degeneracy capture
# ---------------------------------------------------------------------------


class _PcDegeneracyHandler(logging.Handler):
    """Logging handler that counts PC singular-matrix fallback warnings.

    Installed on `evaluation.chamber_pipeline.inference` for the
    duration of one cell. The warning text is matched on a stable
    substring (`"fell back"`) — same wording the inference module
    uses, kept loose enough to survive minor message tweaks but tight
    enough to ignore unrelated future warnings.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        # The inference module's fallback warning starts with "PC
        # inference fell back to all-zeros adjacency". Match loosely
        # on "fell back" so wording tweaks don't silently break this.
        if "fell back" in record.getMessage().lower():
            self.count += 1


class _PcCollinearHandler(logging.Handler):
    """Counts how many columns PC dropped as numerically duplicate.

    Separate from `_PcDegeneracyHandler` on purpose: a collinear drop is a
    LOCAL loss (the dropped node makes no claim, the rest of the graph is
    still inferred), while a degeneracy fallback is a TOTAL loss (all-zeros
    for every node). Averaging them into one number would hide which of the
    two a sweep actually hit.

    Counted per cell so the rate is visible in the harness-validity report.
    A degradation path whose rate can vary with the experiment's independent
    variable and that leaves no trace is how a harness ends up measuring
    itself rather than the thing under study.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        # Inference logs "PC dropped N collinear column(s) at |r|>=...".
        # Match on "collinear" and recover N from the args so the count is
        # columns-dropped, not warnings-emitted.
        message = record.getMessage().lower()
        if "collinear" in message and "dropped" in message:
            first = record.args[0] if isinstance(record.args, tuple) and record.args else 1
            self.count += int(first) if isinstance(first, int) else 1


class _PcZeroVarianceHandler(logging.Handler):
    """Counts how many columns PC dropped as constant in the pooled data.

    The third and last of the PC degradation paths, and the one that most
    directly tracks the experiment's independent variable: a variable that no
    bought experiment perturbed is constant in the pooled view, gets dropped,
    and is padded back with zeros -- a guaranteed false negative for every
    edge incident to it. The fewer experiments bought, the more of the graph
    is answered by padding rather than by inference.

    That is not a confound between arms -- activating more variables is
    precisely what good selection does, so this is the causal pathway rather
    than a bias in it. It is counted so the budget curve can be decomposed
    into "PC saw more nodes" versus "PC inferred better edges", which without
    a counter is not recoverable from the recorded cells at all.

    Distinct from a collinear drop (a column with signal that duplicates
    another) and from a degeneracy fallback (total loss, all-zeros for every
    node). The three markers are asserted mutually exclusive by
    `test_pc_warning_markers_are_unambiguous`.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        # Inference logs "PC dropped N zero-variance column(s): ...".
        # Recover N from the args so the count is columns-dropped, not
        # warnings-emitted -- the same convention as the collinear handler.
        message = record.getMessage().lower()
        if "zero-variance" in message and "dropped" in message:
            first = record.args[0] if isinstance(record.args, tuple) and record.args else 1
            self.count += int(first) if isinstance(first, int) else 1


# ---------------------------------------------------------------------------
# LLM-call counting wrapper
# ---------------------------------------------------------------------------


def _response_has_finish_reason_error(response: Any) -> bool:
    """Return True iff any choice in the response carries finish_reason='error'.

    Used by `_CountingLLM` to detect body-encoded provider failures —
    OpenRouter returns HTTP 200 with `finish_reason: 'error'` (and the
    visible `content` empty) when an upstream provider rejects the
    request internally. Our retry path treats this as equivalent to
    a transient HTTP failure and rotates providers.

    Tolerant of both dict-shape and Pydantic-shape responses,
    mirroring `_response_text` in `llm_planner`. Returns False on
    structurally malformed responses (rather than raising) so the
    caller can fall through to its own parser-side empty-content
    handling.
    """
    try:
        choices = response["choices"] if isinstance(response, dict) else response.choices
        for choice in choices:
            finish_reason = (
                choice["finish_reason"]
                if isinstance(choice, dict)
                else getattr(choice, "finish_reason", None)
            )
            if finish_reason == "error":
                return True
        return False
    except (KeyError, AttributeError, TypeError, IndexError):
        return False


# Re-probed 2026-08-29 and REORDERED; the previous order had gone stale in a
# way that cost money on every call. It read (Parasail, SiliconFlow, Baidu,
# Novita), from a probe when all three of the first were $0.280/M. Since then
# SiliconFlow has moved to $0.660/M -- so the SECOND choice in the rotation had
# become the dearest of the four -- while Baidu sits at $0.090/M, the cheapest
# of thirty endpoints, in third place.
#
# Measured on a realistic late-loop prompt, price and throughput agree for once:
#
#   Baidu      $0.090/M fp8   4.2s   164 tok/s
#   CoreWeave  $0.280/M fp8  13.3s   136 tok/s
#   DeepInfra  $0.180/M fp8  19.1s    89 tok/s
#   Parasail   $0.280/M fp8  16.3s    58 tok/s
#
# All four fp8, so the rotation cannot silently change precision -- the defect
# that put 27 M6 cells on an fp4 endpoint. `Together` stays excluded despite a
# low price: it spends the whole token cap on reasoning and returns empty
# content, which degrades to `rng.choice`. `BaseTen` is excluded because it
# rate-limited on the probe itself.
_FLASH_PROVIDER_ORDER: tuple[str, ...] = (
    "Baidu",
    "CoreWeave",
    "DeepInfra",
    "Parasail",
)

# GLM's endpoints are uniformly fp8 -- Relace, Z.AI, Novita, DeepInfra and
# GMICloud all quote $0.250/M out at fp8 -- so unlike the deepseek family
# there is no precision decision to get wrong here. Ordered by vendor-first
# then the endpoints we already exercise for other models.
_GLM_PROVIDER_ORDER: tuple[str, ...] = (
    "Z.AI",
    "DeepInfra",
    "Novita",
    "GMICloud",
)


class _CountingLLM:
    """Per-cell LLM proxy that counts calls and accumulates token / cost.

    Wraps either a user-supplied LLM callable (FakeLLM in tests) or
    `litellm.completion` (production, lazy-imported on first call).
    Either way, exposes `.calls` (list, length = number of invocations
    in this cell) and three running totals
    (`total_input_tokens`, `total_output_tokens`, `total_cost_usd`).

    Why "always wrap":
        Before this, `run_cell` had two code paths — one for the
        FakeLLM-style `.calls` attribute, one fall-through that left
        `n_llm_calls=None` on production runs. Always wrapping
        unifies the paths: the orchestrator instantiates a fresh
        `_CountingLLM` per cell, the user's LLM (if any) is invoked
        through it, and the counter is read off the wrapper after
        the cell finishes.

    Token / cost extraction:
        Best-effort. A response is checked for OpenAI-shaped
        `usage.prompt_tokens` / `usage.completion_tokens` (LiteLLM
        normalizes to this) and `_hidden_params.response_cost` (when
        LiteLLM populates it). If any field is missing — e.g., a
        FakeLLM whose response doesn't carry `usage` — the running
        totals stay at zero and the orchestrator records None on the
        RunRecord. Both dict-shape and Pydantic-attr responses are
        accommodated, mirroring the pattern in
        `agents.llm_planner._response_text`.
    """

    def __init__(self, target: LLMCallable | None = None) -> None:
        # Capture the target callable (None = use real litellm.completion;
        # resolved lazily on first call so the import cost only happens
        # when an LLM-bearing variant actually runs).
        self._target = target
        self.calls: list[dict[str, Any]] = []
        # `calls` records one entry per provider ATTEMPT, deliberately, so
        # rotation shows up in cost attribution. That makes it the wrong
        # denominator for `fallback_rate`, whose numerator counts logical
        # selections: a cell that rotated would report a LOWER degradation
        # rate the worse the serving stack behaved. Counted separately.
        self.n_requests = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cost_usd: float = 0.0
        # Incremented by `_llm_select_loop` when a selection response cannot
        # be parsed and it falls back to a random unspent experiment. Recorded
        # per cell so a degraded run is visible in the results rather than
        # indistinguishable from a healthy one.
        self.selection_fallbacks: int = 0
        # Provenance: what actually served this cell. Recorded because a
        # pinned model snapshot does NOT pin behaviour -- DeepSeek raised the
        # default reasoning effort under unchanged 0423 weights on 2026-08-13,
        # and reconstructing May's effort level from token arithmetic was only
        # possible by luck. Anything that can change under us gets recorded.
        self.observed_models: set[str] = set()
        self.observed_efforts: set[str] = set()
        self.observed_providers: set[str] = set()

    # Default LiteLLM retry count for transient failures (rate limits,
    # network blips, 5xx). LiteLLM's default is 0 — meaning the first
    # 429 response from OpenRouter raises immediately. Setting this to
    # 3 enables exponential backoff that catches transient errors while
    # letting slow-but-OK responses complete normally. Verified against
    # the M4b smoke: pre-fix, OpenRouter throttling produced ~30-50%
    # cell-error rate on sustained LLM bursts.
    DEFAULT_NUM_RETRIES = 3

    # Inference-precision class per OpenRouter endpoint, read from
    # `GET /models/{id}/endpoints` on 2026-08-25. This is a table and not a
    # comment on purpose. The previous version of this block *asserted in
    # prose* that Novita and AtlasCloud were "both fp8". AtlasCloud is fp4,
    # and 27 of the 450 M6 ladder cells were served by it before anyone
    # checked — the precision class the comment existed to hold constant was
    # already broken. A comment cannot be verified; this mapping is, by
    # `test_default_provider_order_is_precision_homogeneous`.
    #
    # (The M6 contamination was measured and is not distorting: raw F1 on
    # fp4-touched cells differs at p<1e-4, but that is entirely arm x budget
    # composition. Residualised on arm x budget it is -0.004 vs +0.000,
    # Welch p=0.61. Reported as a limitation, not a correction.)
    PROVIDER_PRECISION: ClassVar[dict[str, str]] = {
        # fp8 — eligible for DEFAULT_PROVIDER_ORDER
        "Parasail": "fp8",
        "SiliconFlow": "fp8",
        "Baidu": "fp8",
        "CoreWeave": "fp8",
        "DeepInfra": "fp8",
        # GLM's endpoints, all fp8 at one price.
        "Z.AI": "fp8",
        "GMICloud": "fp8",
        "StreamLake": "fp8",
        "Novita": "fp8",
        # fp4 — INELIGIBLE, different numerics
        "AtlasCloud": "fp4",
        "Reka": "fp4",
        # unquantised/undeclared — INELIGIBLE, precision class unknown
        "Together": "unknown",
        "DeepSeek": "unknown",
        "Cloudflare": "unknown",
        "Fireworks": "unknown",
        "Alibaba": "unknown",
        "Venice": "unknown",
        "Phala": "unknown",
        "Wafer": "unknown",
    }

    # OpenRouter's `provider.order` preference for `deepseek-v4-flash`.
    #
    # Ordered by PRICE among verified-fp8 endpoints, not by throughput.
    # OpenRouter serves one model id from many endpoints at *different
    # prices*: on 2026-08-25 the identical `deepseek-v4-flash-0731` cost
    # $0.280/M output on Parasail, SiliconFlow and Baidu, and $1.320/M on
    # Novita, AtlasCloud, DeepSeek and Cloudflare — 4.7x for the same
    # weights. The M6 ladder ran 422/450 cells on Novita and cost $54.53;
    # the same sweep on Parasail would have cost ~$11.60.
    #
    # The May 2026 ordering put Novita first for throughput, which was
    # correct then (Parasail had degraded to ~9 t/s). Re-probed 2026-08-25
    # on a realistic late-loop selection prompt: Parasail 17-34s and always
    # valid, Novita 21s — no throughput reason left to pay 4.7x.
    #
    # `Together` is excluded despite the $0.280 price: it spent the entire
    # 32768-token cap on reasoning and returned EMPTY content
    # (finish_reason='length'), which this harness degrades to `rng.choice`.
    # That is the `_SELECTION_MAX_TOKENS` failure one level down — the
    # provider, not just the model, sets truncation risk.
    #
    # `StreamLake` is fp8 at $0.147/M (cheapest) but unprobed; add it only
    # after a throughput/validity check like the one above.
    #
    # Rotation in `__call__` advances on `finish_reason: 'error'`, and the
    # cell-timeout safety net bounds any genuinely-slow provider at
    # `cell_timeout_seconds`.
    # Per-model provider orders. The price and precision RANKING of endpoints
    # is model-specific, so one global order cannot be right for two models.
    # Measured 2026-08-27 from GET /models/{id}/endpoints:
    #
    #   deepseek-v4-pro   Baidu $1.58/M ... Parasail $3.48/M  (18 endpoints)
    #   flash-0731        Parasail $0.28/M ... Novita $1.32/M  (8 endpoints)
    #
    # Parasail is the CHEAPEST fp8 endpoint for flash and the MOST EXPENSIVE
    # one for pro. Running the flash order on pro would overpay 2.2x -- the
    # register §3-4 defect on a different model. Every provider listed here is
    # fp8 and present in `PROVIDER_PRECISION`; the homogeneity test walks all
    # declared orders, not just the default.
    PROVIDER_ORDER_BY_MODEL: ClassVar[dict[str, tuple[str, ...]]] = {
        # Every model the pipeline may run, named explicitly. The flash
        # snapshots previously reached their order through a silent default,
        # which meant an unlisted model reached it too.
        "deepseek-v4-flash-0731": _FLASH_PROVIDER_ORDER,
        "deepseek-v4-flash": _FLASH_PROVIDER_ORDER,
        "deepseek-v4-pro": ("Baidu", "StreamLake", "SiliconFlow", "Novita"),
        # A genuinely different vendor and architecture, for the mixed-model
        # arm: rung 1's two scouts currently differ only by sampling
        # temperature, which is one opinion drawn twice rather than two.
        "glm-5.3-flash": _GLM_PROVIDER_ORDER,
    }

    DEFAULT_PROVIDER_ORDER: tuple[str, ...] = _FLASH_PROVIDER_ORDER

    @classmethod
    def _provider_order_for(cls, model: str) -> tuple[str, ...]:
        """The pinned endpoint order for `model`, matched on the EXACT tag.

        Exact rather than substring, because a dated snapshot is a different
        product with a different endpoint set: `deepseek-v4-pro-0813` is not
        served by Baidu at all, and StreamLake serves `deepseek-v4-pro` at fp8
        while its `-0813` endpoint reports `unknown`. Quantization is a
        property of the (provider, model) pair, not of the provider. A
        substring match would hand an unlisted snapshot a pin naming an
        endpoint that does not serve it, and a precision claim that was never
        checked for it.

        Unknown models RAISE. They used to inherit `DEFAULT_PROVIDER_ORDER`,
        which was safe only while the pin was a preference. It is not: since
        `allow_fallbacks` became False (register entry 8) the order is a hard
        constraint, so an unlisted model gets four endpoints chosen and
        price/precision-verified for deepseek-v4-flash-0731 and nothing else.
        If none serve it every cell errors; if some do, it runs on a precision
        class `PROVIDER_PRECISION` never certified for it -- and the
        homogeneity test still passes, because it checks the pin, not the
        model. Same policy as `_ladder_calibration`: measure it, do not
        extrapolate from a neighbouring configuration.
        """
        tag = model.rsplit("/", 1)[-1]
        try:
            return cls.PROVIDER_ORDER_BY_MODEL[tag]
        except KeyError:
            raise SweepConfigurationError(
                f"no provider order is pinned for model {tag!r}; pinned models "
                f"are {sorted(cls.PROVIDER_ORDER_BY_MODEL)}. `allow_fallbacks` "
                "is False, so the order is a hard constraint and borrowing "
                "another model's endpoints either fails every cell or runs on "
                "an uncertified precision. Add the model to "
                "PROVIDER_ORDER_BY_MODEL (and PROVIDER_PRECISION) after "
                "probing which endpoints serve it."
            ) from None

    # Per-request socket-level timeout (seconds). Without this, a single
    # litellm.completion call can BLOCK FOREVER on the underlying SSL
    # socket read when the upstream provider accepts the request but
    # stops sending bytes mid-response. Discovered via root-cause
    # systematic debugging during M4b smoke: a stuck call kept the
    # python process at 0% CPU for 12+ minutes inside `_ssl__SSLSocket_read`.
    # Was 30.0 until 2026-08-25, with a comment calling that "generous for
    # normal completions (~1-15s)". True in May; invalidated twice since, by
    # the 32768 token caps and by August's provider-side reasoning increase.
    # By the WT gate the MEDIAN cell averaged 30.3s per call -- the timeout sat
    # at the middle of the latency distribution, with 21 of 42 cells above it
    # (LT M6: 26.4% above, surviving only because Novita ran a few seconds
    # faster).
    #
    # That is a MODERATOR, not noise. `num_retries` rescues most over-runs, so
    # failures appear only where latency is highest -- `team` and
    # `fan_in_spec` at the top budget -- making the error rate a function of
    # both the topology IV and the budget axis. The WT gate lost 3 of 45 cells
    # that way, all at k=21, all in the two heaviest arms.
    #
    # 300s is ~3.5x the measured p99 and ~2.7x the slowest legitimate call
    # observed (109.8s on SiliconFlow), while still surfacing the 12-minute
    # `_ssl__SSLSocket_read` hang this constant was introduced for, and still
    # far below `--cell-timeout-seconds`. num_retries handles *exceptions*;
    # without a timeout, hangs never raise → retries never trigger → the
    # process appears wedged. Both are required.
    # p99 of measured mean-seconds-per-call, WT gate 2026-08-25 (42 ok cells,
    # Parasail/SiliconFlow, deepseek-v4-flash-0731): 78.9s, max 86.1s. A direct
    # probe the same day saw a single SiliconFlow call at 109.8s. Recorded as a
    # constant, not prose, so `DEFAULT_REQUEST_TIMEOUT_SECONDS` can be checked
    # against it (`test_timeout_clears_measured_p99_with_margin`).
    MEASURED_CALL_P99_SECONDS = 86.0

    DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0

    def __call__(self, **kwargs: Any) -> Any:
        if self._target is None:
            from litellm import completion as _completion

            self._target = _completion

        self._note_request(kwargs)

        # Inject retry count if caller didn't specify one. FakeLLM's
        # `**_: Any` catch-all silently absorbs unknown kwargs, so this
        # is safe across both the production litellm path and test
        # paths. Caller-supplied num_retries (e.g., 0 to disable for a
        # specific cell) wins.
        self.n_requests += 1
        kwargs.setdefault("num_retries", self.DEFAULT_NUM_RETRIES)

        # Inject per-request timeout. Without this, a stuck SSL socket
        # read (provider accepted the request but stopped sending bytes)
        # blocks indefinitely. With timeout + num_retries, a stuck call
        # surfaces as a Timeout exception that retry-with-backoff can
        # recover from. See DEFAULT_REQUEST_TIMEOUT_SECONDS docstring
        # for the full root-cause analysis.
        kwargs.setdefault("timeout", self.DEFAULT_REQUEST_TIMEOUT_SECONDS)

        # If caller supplied their own provider config, honor it
        # verbatim and skip our rotation logic (single attempt).
        existing_extra = kwargs.get("extra_body") or {}
        caller_supplied_provider = "provider" in existing_extra

        if caller_supplied_provider:
            # Single attempt with caller's config; preserve call accounting.
            self.calls.append({"model": kwargs.get("model"), "idx": len(self.calls)})
            response = self._target(**kwargs)
            self._accumulate_usage(response)
            return response

        # Provider rotation loop.
        #
        # `allow_fallbacks` is FALSE, so OpenRouter may use only the
        # providers we list. It was True until 2026-08-26, and the 750-cell
        # WT sweep showed what that costs: 22 cells (2.9%) were served by
        # OpenInference, Relace or DigitalOcean -- none pinned, none in
        # `PROVIDER_PRECISION`, quantization unknown. Impact on that sweep
        # was nil (residualised on arm x budget, +0.0095 vs -0.0003, Welch
        # p=0.55), but the guarantee was the point: with fallbacks on,
        # `PROVIDER_PRECISION` and its homogeneity test certify what we
        # REQUEST, not what ran. That is the fp4/AtlasCloud lesson one layer
        # down -- there a pinned provider had drifted precision, here routing
        # left the pinned set entirely.
        #
        # The trade is availability for reproducibility: if all four pinned
        # fp8 endpoints fail, the call now fails instead of silently
        # succeeding elsewhere. That is the correct direction for a paper
        # that reports the serving stack, and the rotation below plus
        # `num_retries` still give four providers x four attempts of
        # redundancy inside the pinned set.
        #
        # Rotation exists because OpenRouter's own fallback (when it was
        # enabled) cycled `provider.order` on an HTTP error — but **NOT**
        # when a provider returns 200 with `finish_reason: 'error'` in the
        # response body
        # (a body-encoded soft failure). We saw this in the M4b
        # re-smoke (2026-05-14): Parasail returned ~40 body-encoded
        # errors over a 1-hr window, never triggering OpenRouter
        # fallback, and `litellm.num_retries=3` masked the failure
        # because it only retries on raised exceptions.
        #
        # The fix is to inspect `finish_reason` ourselves and, on
        # body-encoded error, rotate the provider list (failed primary
        # moves to the end) and retry. Each attempt is a separate HTTP
        # request to OpenRouter, so the rotated `provider.order` takes
        # effect. After exhausting the list, we surface the last
        # response — the agent's downstream parsers (parse_selection_response
        # / parse_adjacency_response) already handle empty content via
        # fallback paths.
        provider_order = list(self._provider_order_for(kwargs.get("model", "")))
        max_attempts = len(provider_order)
        response: Any = None
        for attempt in range(max_attempts):
            kwargs["extra_body"] = {
                **existing_extra,
                "provider": {
                    "order": provider_order,
                    "allow_fallbacks": False,
                },
            }

            # Record each attempt as a distinct call for cost-attribution
            # audit (provider rotation is a real cost; we don't hide it).
            self.calls.append(
                {
                    "model": kwargs.get("model"),
                    "idx": len(self.calls),
                    "attempt": attempt,
                    "primary_provider": provider_order[0],
                }
            )

            response = self._target(**kwargs)
            self._accumulate_usage(response)

            if not _response_has_finish_reason_error(response):
                return response

            # Failed with body-encoded error: rotate so the failed
            # primary is now at the end, and try the next provider.
            provider_order = provider_order[1:] + provider_order[:1]

        # All providers exhausted with body-encoded errors. Return the
        # last (still-bad) response; the caller's parser will fall back.
        return response

    def _note_request(self, kwargs: dict[str, Any]) -> None:
        """Record the model and reasoning effort this request asked for."""
        model = kwargs.get("model")
        if isinstance(model, str):
            self.observed_models.add(model)
        reasoning = (kwargs.get("extra_body") or {}).get("reasoning") or {}
        effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
        # "unset" is recorded explicitly rather than skipped: an unset
        # parameter tracking a provider default is the failure being guarded
        # against, so its absence must be visible in the results.
        self.observed_efforts.add(effort if isinstance(effort, str) else "unset")

    def _note_response(self, response: Any) -> None:
        """Record which upstream provider actually served the request."""
        try:
            provider = (
                response.get("provider")
                if isinstance(response, dict)
                else getattr(response, "provider", None)
            )
        except (AttributeError, TypeError):
            return
        if isinstance(provider, str) and provider:
            self.observed_providers.add(provider)

    def record_selection_fallback(self) -> None:
        """Note that one selection call degraded to a random pick."""
        self.selection_fallbacks += 1

    def _accumulate_usage(self, response: Any) -> None:
        """Best-effort usage / cost extraction; tolerant of all response shapes.

        Updates `total_input_tokens`, `total_output_tokens`, `total_cost_usd`.
        Responses may be dict-shape (most LiteLLM responses) or Pydantic-shape
        (some providers); missing fields silently leave totals at 0.
        """
        self._note_response(response)

        try:
            usage = (
                response.get("usage", {})
                if isinstance(response, dict)
                else getattr(response, "usage", {}) or {}
            )
            in_tok = (
                usage.get("prompt_tokens")
                if isinstance(usage, dict)
                else getattr(usage, "prompt_tokens", 0)
            ) or 0
            out_tok = (
                usage.get("completion_tokens")
                if isinstance(usage, dict)
                else getattr(usage, "completion_tokens", 0)
            ) or 0
            self.total_input_tokens += int(in_tok)
            self.total_output_tokens += int(out_tok)
        except (AttributeError, TypeError, ValueError):
            pass

        try:
            hidden = (
                response.get("_hidden_params", {})
                if isinstance(response, dict)
                else getattr(response, "_hidden_params", {}) or {}
            )
            cost = (
                hidden.get("response_cost", 0.0)
                if isinstance(hidden, dict)
                else getattr(hidden, "response_cost", 0.0)
            ) or 0.0
            self.total_cost_usd += float(cost)
        except (AttributeError, TypeError, ValueError):
            pass


# ---------------------------------------------------------------------------
# Per-cell runner
# ---------------------------------------------------------------------------


def _budget_fraction(budget_k: int, menu_size: int) -> float:
    """Compute k/M, clamped to [0, 1]. Returns 0.0 on empty menus."""
    if menu_size <= 0:
        return 0.0
    return min(1.0, budget_k / menu_size)


# Per-role MEDIAN call costs in INPUT+OUTPUT tokens -- what `token_meter`
# actually counts -- measured 2026-08-23 through the production provider
# order at k=30, n=8 per role.
#
# Median, not p95, and the reason is statistical rather than aesthetic. At
# n=8 a "p95" is just the maximum of eight draws, and this distribution has a
# long tail: the same plain selection call was measured at 2,205 (median) and
# 12,173 (one instrumented run). Sizing from that max under-provisioned
# `fan_in_homog` so badly it failed conservation on 2 of 2 live cells while
# the two better-funded arms passed 4 of 4 -- a budgeting artifact that would
# have read as a coordination result. The median is the robust statistic at
# this sample size; the tail is covered by the multiplier below.
#
# Two further things this table encodes:
#
#   * The roles are not interchangeable. Targeted costs 4.7x plain. Budgeting
#     both at one figure under-funds whichever scout reasons harder.
#   * Providers must be pinned to measure this at all. An unpinned run put
#     broad and targeted within 3% of each other; pinning the order the
#     orchestrator uses showed the 4.7x gap. Measure through the production
#     path, not a convenient approximation.
#
# `_PROVISION_MULTIPLE` is a single uniform rule applied to every role, NOT
# tuned per arm until each passes. Whatever compliance rate it yields is
# reported as an observed rate; tuning until H-C reads 100% would make the
# hypothesis vacuous.
_PROVISION_MULTIPLE = 4

# PER-CALL scout cost, keyed by (CHAMBER, ROLE).
#
# Was keyed by role alone, with every figure measured on LT. The WT gate
# (2026-08-25, 27 graph cells) shows the roles do not transfer: `targeted`
# costs 10,379 on LT and 2,868 on WT, a 3.6x gap, because the prompt carries
# the menu and WT's is 28 experiments against LT's 59. Carrying LT's number
# over would have over-provisioned that scout 3.6x and inflated H-C compliance
# into a statement about provisioning rather than about the framework.
#
# Medians, per the rule stated for the LT figures: the grant is
# `_PROVISION_MULTIPLE * c95 * calls`, so a median basis already carries 4x
# headroom.
#
# Budget-invariance -- flagged as unverified when these were LT-only -- is now
# CHECKED on WT: per-call cost varies 1.29x (targeted), 1.31x (plain) and
# 1.71x (broad) across k in {7,14,21}, comfortably inside the 4x multiple.
_ROLE_C95: dict[tuple[str, str], int] = {
    ("lt", "plain"): 2205,
    ("lt", "broad"): 3003,
    ("lt", "targeted"): 10379,
    ("wt", "plain"): 2112,
    ("wt", "broad"): 2050,
    ("wt", "targeted"): 2868,
}
# Aggregator (reconcile + any negotiation) spend per cell, in input+output
# tokens, keyed by BUDGET. Not one constant: reconcile prompts list both
# scouts' selections, so cost grows with k, and the gate at k=45 measured a
# median of 16,980 against the 8,557 previously applied at every budget. With
# the single constant, capacity was 1.5 * 8,557 = 12,836 while nine graph
# cells spent 9,783-25,168 -- 6 of 9 failed `verify()`. That would have been
# reported as 0% H-C compliance for the fan-in rungs and was pure
# provisioning.
#
# Medians, matching `_ROLE_C95`'s stated rule, NOT tuned upward until H-C
# reads 100%. Measured untruncated (post 32768-cap) through the production
# provider order on deepseek-v4-flash-0731.
#
# Unlisted budgets RAISE rather than fall back to a nearby value. A wrong
# `a95` does not fail loudly -- it yields plausible conservation numbers that
# are really statements about provisioning, which is exactly how the single
# constant survived until the k=45 gate.
# KEYED BY (CHAMBER, BUDGET). Every figure below was measured on LT, whose
# menu is 59 experiments; WT's is 28, so its reconcile prompt -- which lists
# both scouts' selections against the menu -- is a different length and a
# different cost. A dict keyed by `k` alone silently lends LT's numbers to any
# other chamber, which is the same silent-plausible-number failure that let a
# single `_A95_RECONCILE` survive until the k=45 gate.
_A95_RECONCILE_BY_K: dict[tuple[str, int], int] = {
    ("lt", 6): 7646,
    ("lt", 30): 11427,
    ("lt", 45): 18790,
    # WT, measured on the 2026-08-25 gate (9 graph cells per budget, p75).
    # Note the shape differs from LT's: LT rises monotonically with k while
    # WT peaks at its LOWEST budget. What the two share is that the lowest
    # budget is the least predictable -- spread 26x on WT k=7, 48.8x on LT
    # k=6 -- so P2's window is not demonstrable there in either chamber.
    ("wt", 7): 8593,
    ("wt", 14): 4764,
    ("wt", 21): 5491,
}

# (chamber, k) pairs whose calibration is a placeholder rather than a
# measurement. They exist for exactly one reason: a calibration gate on a new
# chamber has to RUN before anything can be measured, and `_ladder_calibration`
# refuses unmeasured budgets by design.
#
# The safety property that makes this acceptable: `run_cell` forces
# `conservation_certified` to None for any cell run under a provisional entry,
# so a gate cannot contribute H-C numbers. Token accounting is unaffected --
# node monitors record spend regardless of the granted budget, which is what
# the gate is there to measure.
#
# REMOVE each entry as its measurement lands. An entry left here after
# measurement silently voids conservation for the whole sweep.
_PROVISIONAL_CALIBRATION: frozenset[tuple[str, int]] = frozenset()

# Chambers where `_C95_NEGOTIATE` has been measured. It enters only through
# `spec.negotiation_rounds`, so it affects the `team` arm and nothing else --
# which is why it is tracked separately rather than voiding a whole chamber.
# WT's negotiate cost was never isolated by the gate, so WT `team` cells run
# on LT's figure and have their conservation voided; the fan-in arms, whose
# c95 and a95 ARE measured on WT, report conservation normally.
_NEGOTIATE_CALIBRATED_CHAMBERS: frozenset[str] = frozenset({"lt"})


def is_provisional_calibration(chamber: str, budget_k: int, *, negotiates: bool = False) -> bool:
    """True if this cell runs on any placeholder calibration figure.

    Consulted by `run_cell` to void `conservation_certified`. Two sources:
    an unmeasured (chamber, budget) pair, and -- for negotiating arms only --
    a chamber whose `_C95_NEGOTIATE` was never isolated.
    """
    if (chamber, budget_k) in _PROVISIONAL_CALIBRATION:
        return True
    return negotiates and chamber not in _NEGOTIATE_CALIBRATED_CHAMBERS


# 75th percentile of measured aggregator spend, NOT the median -- and the
# reason is a design asymmetry rather than a preference for the numbers.
#
# `_ROLE_C95` uses medians because the scouts' grant is `_PROVISION_MULTIPLE *
# c95 * calls`, so a median basis already carries 4x headroom. The aggregator
# gets `1.5 * a95` and NO multiple, deliberately: any margin would lift every
# single fragment above the call and a tree encoding would cope, destroying
# the P2 demonstration (spec §12). Carrying the median over to `a95` therefore
# imported the statistic without its multiplier, and a budget sized at the
# median overruns ~50% of executions by construction -- not a defensible
# provisioning rule in a paper about resource governance.
#
# Measured untruncated (post 32768-cap) on deepseek-v4-flash-0731, n=9 graph
# cells per budget, production provider order pinned. Cells conserving / cells
# inside the P2 window, median basis vs p75 basis:
#
#   k=6    5/9, 1/9   ->   7/9, 2/9      spread 48.8x (500 - 24,415)
#   k=30   8/9, 6/9   ->   8/9, 5/9      spread  5.2x (4,648 - 24,001)
#   k=45   9/9, 6/9   ->   9/9, 6/9      spread  2.6x (9,783 - 25,168)
#
# Better or equal everywhere except one P2 cell at k=30.
#
# The k=6 spread is the headline caveat. Aggregator cost there ranges
# 500-24,415 -- its MAXIMUM nearly equals k=45's -- because the reconcile
# prompt lists only 3+3 names, so cost is dominated by erratic reasoning
# length rather than prompt size. Against P2's window, only `n`-wide for `n`
# parents (2x here), no single constant can both conserve the 24,415 cell and
# keep the 500 cell inside the window. P2 is demonstrable at k=30 and k=45 and
# effectively not at k=6; report that as a scope limit.
#
# Consequence for H-C, and it needs saying in the paper: a conservation
# "failure" is a failure of OUR FORECAST, not of the mechanism. The aggregator
# really did consume 24,415 against its grant and `verify()` caught it every
# time. Reporting a bare compliance rate conflates "the framework enforces
# conservation" (it does, 100%) with "our provisioning predicted cost" (at k=6
# it cannot). Report the two separately.
_C95_NEGOTIATE = 4138

_LADDER_NODES = ("scout_a", "scout_b", "aggregator")


def _ladder_calibration(
    spec: AgentSpec, budget_k: int, chamber: str = "lt"
) -> tuple[int, int, int, int]:
    """`(c95_a, c95_b, a95, fixed_overhead)` for one ladder arm at one budget.

    Roles are read off the spec rather than matched on the arm's name: a
    name-keyed lookup silently returns plain-role figures for any arm it does
    not recognise.

    `a95` is keyed by BUDGET. An earlier version of this docstring asserted
    that per-call cost is "driven by the role's prompt, not by `k`" -- false
    for the aggregator, whose reconcile prompt lists both scouts' selections
    and therefore grows with k. See `_A95_RECONCILE_BY_K`.

    Note `c95` is still budget-INVARIANT here, and that is not yet verified at
    k=45: `RunRecord` records `aggregator_tokens` but no per-scout spend, so
    the gate could not tell whether the scouts also overran. Scout token
    fields are recorded from 2026-08-24 onward precisely so the next
    calibration can answer it.

    Raises:
        ValueError: If the arm is not a ladder arm, or `budget_k` has no
            measurement. Extrapolating would produce plausible-looking
            conservation numbers that are really statements about
            provisioning.
    """
    if spec.scout_roles is None:
        raise ValueError(f"{spec.name!r} is not a ladder arm")
    if (chamber, budget_k) not in _A95_RECONCILE_BY_K:
        known = sorted(k for c, k in _A95_RECONCILE_BY_K if c == chamber)
        raise SweepConfigurationError(
            f"aggregator spend is not calibrated for chamber={chamber!r} at "
            f"k={budget_k}; calibrated budgets for that chamber are {known}. "
            "Measure it rather than extrapolating -- a wrong a95 fails "
            "silently, yielding conservation numbers that are really "
            "statements about provisioning."
        )
    role_a, role_b = spec.scout_roles
    overhead = spec.negotiation_rounds * (_PROVISION_MULTIPLE * _C95_NEGOTIATE)
    for role in (role_a, role_b):
        if (chamber, role) not in _ROLE_C95:
            raise ValueError(
                f"scout role {role!r} is not calibrated for chamber={chamber!r}; "
                f"measured roles are "
                f"{sorted(r for c, r in _ROLE_C95 if c == chamber)}"
            )
    return (
        _ROLE_C95[(chamber, role_a)],
        _ROLE_C95[(chamber, role_b)],
        _A95_RECONCILE_BY_K[(chamber, budget_k)],
        overhead,
    )


def _build_agent_kwargs(
    spec: AgentSpec,
    budget_k: int,
    seed: int,
    pc_alpha: float,
    llm: LLMCallable | None,
    model: str | None = None,
) -> dict[str, Any]:
    """Construct the kwargs dict to pass to `spec.run`.

    Centralizes the per-variant kwargs assembly:
      - All variants get `seed`.
      - PC-using variants get `pc_alpha` (random, greedy_ig_lite,
        llm_pc, planner_reasoner — i.e., everyone except llm_only).
      - LLM-bearing variants get `llm` if provided.
      - Planner+Reasoner additionally needs `planner_budget` and
        `reasoner_budget` — split evenly with the remainder going to
        the planner (the more-defensive default per plan §5.3).
    """
    kwargs: dict[str, Any] = {"seed": seed}

    # llm_only doesn't take pc_alpha (no PC inference step).
    if spec.name != "llm_only":
        kwargs["pc_alpha"] = pc_alpha

    if spec.accepts_llm and llm is not None:
        kwargs["llm"] = llm

    if "planner_budget" in spec.extra_kwargs:
        # Even split with remainder to planner: floor(k/2) for reasoner,
        # k - reasoner for planner. Total exactly equals budget_k so
        # conservation is satisfied at the framework level too.
        reasoner_budget = budget_k // 2
        planner_budget = budget_k - reasoner_budget
        kwargs["planner_budget"] = planner_budget
        kwargs["reasoner_budget"] = reasoner_budget

    if "scout_a_budget" in spec.extra_kwargs:
        # Even split, remainder to scout_a -- matching the planner/reasoner
        # convention above so the ladder's rungs stay budget-comparable.
        kwargs["scout_b_budget"] = budget_k // 2
        kwargs["scout_a_budget"] = budget_k - budget_k // 2

    kwargs.update(spec.static_kwargs)

    # AFTER static_kwargs: an explicit `--model` is a deliberate operator
    # choice and must outrank a spec-level default, not be silently
    # overwritten by it. Guarded on `accepts_llm` because the non-LLM agents
    # have no `model` parameter and would raise TypeError.
    if model is not None and spec.accepts_llm:
        kwargs["model"] = model

    return kwargs


# Resolved once at import: `run_pc`'s bound defaults never change at runtime.
_PC_CALL_DEFAULTS = pc_call_defaults()
_RUNTIME = runtime_fingerprint()


def run_cell(
    spec: AgentSpec,
    chamber: ChamberId,
    configuration: ConfigId,
    budget_k: int,
    seed: int,
    pc_alpha: float = 0.05,
    llm: LLMCallable | None = None,
    cell_timeout_seconds: float | None = None,
    model: str | None = None,
) -> RunRecord:
    """Run one cell of the sweep grid and return a RunRecord.

    Catches all exceptions per cell — this method NEVER raises (catches
    `Exception`, not `BaseException`, so KeyboardInterrupt still
    propagates by design — Ctrl-C should kill the sweep, not silently
    convert to an error record). Skipped, ok, and error cells all
    produce well-formed RunRecords with `status` set accordingly.

    Args:
        spec: The agent variant to run.
        chamber: Which chamber to load.
        configuration: Chamber configuration.
        budget_k: Total intervention budget for this cell.
        seed: RNG seed.
        pc_alpha: PC independence-test significance level (ignored for
            llm_only; passed through for all others).
        llm: Injectable LLM callable. None means lazy-import
            `litellm.completion` for the production path. Either way,
            the orchestrator wraps it in a `_CountingLLM` per cell so
            `n_llm_calls` / `tokens_in` / `tokens_out` / `cost_usd`
            are populated on the RunRecord.
        cell_timeout_seconds: Wall-clock timeout for the agent
            invocation (in seconds). None = no timeout. On timeout,
            the cell is recorded as `status="error"` with
            `error_type="TimeoutError"`. Note: the underlying thread
            is NOT killed (Python doesn't support thread-level
            cancellation), but the cell's slot in the sweep is freed
            and the next cell starts immediately. For M4b's serial
            sweep this is acceptable; M5's parallelism case will need
            stronger isolation.

    Returns:
        A RunRecord with status "ok" / "skipped" / "error".
    """
    # TWO clocks on purpose. `started_at`/`finished_at` are wall-clock;
    # `wall_time_seconds` below comes from `time.perf_counter()`, which on
    # macOS is `mach_absolute_time` and DOES NOT ADVANCE WHILE THE SYSTEM IS
    # ASLEEP. Their difference is therefore suspend time, and on a laptop
    # running a multi-hour sweep on battery that is not a rounding error: the
    # 2026-08-25 WT gate spent 1.01h of active worker time inside a 6.66h
    # wall-clock span, with 100.4% of the 70,029s gap matched to `pmset -g
    # log` sleep windows.
    #
    # Keep both. `wall_time_seconds` is the honest cost figure (sweep
    # estimates must use it); the wall-clock pair is what makes an
    # invisible suspend visible -- see `suspend_seconds` in
    # `harness_validity_report`. Run sweeps under `caffeinate -is`.
    started_at = now_iso()
    _t_cell_start = time.perf_counter()

    # Stamped onto EVERY record this function returns, including skips and
    # errors: a re-run needs the failed cell's configuration as much as the
    # ok cell's. `DEFAULT_MAX_ROWS` / `DEFAULT_COLLINEARITY_THRESHOLD` are
    # read from the module rather than hardcoded here, and
    # `test_pc_defaults_match_the_recorded_constants` pins them to `run_pc`'s
    # actual signature defaults -- Python binds default arguments at def
    # time, so a module constant reassigned later would leave `run_pc`
    # unchanged while this stamp reported the new value.
    _pc_provenance: dict[str, Any] = {
        "pc_alpha": pc_alpha,
        "pc_max_rows": _PC_CALL_DEFAULTS["max_rows"],
        "pc_collinearity_threshold": _PC_CALL_DEFAULTS["collinearity_threshold"],
        "blas_backend": _RUNTIME["blas"],
        "platform_tag": _RUNTIME["platform"],
    }

    # Pre-flight: is this agent compatible with this chamber?
    if not spec.is_compatible(chamber):
        return RunRecord(
            **_pc_provenance,
            chamber=chamber,
            configuration=configuration,
            agent_name=spec.name,
            budget_k=budget_k,
            budget_fraction=0.0,  # filled below if we can load the chamber
            seed=seed,
            status="skipped",
            started_at=started_at,
            finished_at=started_at,
            skip_reason=(
                f"agent '{spec.name}' is not compatible with chamber '{chamber}' "
                f"(spec.chambers = {spec.chambers})"
            ),
        )

    # Build the chamber adapter. Failure here is an "error" cell, not
    # a "skipped" — agent compatibility was satisfied but adapter
    # construction itself broke (e.g., disk full, network down for
    # first-time dataset download).
    # Built BEFORE the adapter: the ladder arms' `token_meter` closes over it,
    # and without that closure `as_node` attributes nothing, every node's
    # token consumption stays 0, and verify() is trivially true -- H-2 would
    # be unfalsifiable.
    counting_llm: _CountingLLM | None = None
    if spec.accepts_llm:
        counting_llm = _CountingLLM(target=llm)

    try:
        extra: dict[str, Any] = {}
        graph = None
        if spec.is_ladder_arm:
            from evaluation.chamber_pipeline.coordination import build_fan_in_graph

            c95_a, c95_b, a95, overhead = _ladder_calibration(spec, budget_k, chamber=chamber)
            graph = build_fan_in_graph(
                multiple=_PROVISION_MULTIPLE,
                k=budget_k,
                c95=c95_a,
                a95=a95,
                c95_b=c95_b,
                fixed_overhead=overhead,
            )
            meter = counting_llm
            extra = {
                "node_monitors": {n: graph.monitor_for(n) for n in _LADDER_NODES},
                # No aggregate token cap: `as_node` charges node monitors, so
                # the adapter's own `usage.tokens` stays 0 and any constraint
                # on it would be unreachable. Token budgets live on the graph,
                # where verify() can actually see them.
                "token_meter": (
                    (lambda: meter.total_input_tokens + meter.total_output_tokens)
                    if meter is not None
                    else None
                ),
            }
        # The UNCONTRACTED arm gets the menu size, not `k`. See
        # `AgentSpec.ignores_budget`: a cap of `k` here would quietly
        # re-impose the very constraint the arm exists to remove, and the
        # cell would look like a valid uncontracted run.
        cap = MENU_SIZES[chamber] if spec.ignores_budget else budget_k
        adapter = create_contracted_chamber_agent(
            chamber=chamber,
            configuration=configuration,
            intervention_budget=cap,
            **extra,
        )
        # `MENU_SIZES` is a hardcoded table, and `_budget_k_for` converts
        # every sweep's budget FRACTION into k through it -- for every arm,
        # not just the uncontracted one. If it drifts from the live menu (the
        # WT dataset changed release inside this branch, `wt_walks_v1` ->
        # `wt_validate_v1`), every contracted arm silently runs at the wrong k
        # and records a wrong `budget_fraction`, while only the uncontracted
        # arm raises. Checked unconditionally, as an equality: too small caps
        # the uncontracted arm, too large inflates every k.
        live = len(adapter.available_experiments())
        if MENU_SIZES[chamber] != live:
            raise SweepConfigurationError(
                f"MENU_SIZES[{chamber!r}]={MENU_SIZES[chamber]} disagrees with "
                f"the live menu ({live}). Every budget fraction is converted to "
                f"k through this table, so the whole sweep would run at the "
                f"wrong budget. Update the table to match the dataset."
            )
        if graph is not None:
            adapter.delegation_graph = graph  # read back by the P2 scorer
        menu_size = len(adapter.available_experiments())
        budget_fraction = _budget_fraction(budget_k, menu_size)
    except SweepConfigurationError:
        # Deliberately NOT recorded as a cell error. See the class docstring:
        # a swallowed configuration fault becomes an error rate that every
        # resume retries forever.
        raise
    except Exception as exc:
        return RunRecord(
            **_pc_provenance,
            chamber=chamber,
            configuration=configuration,
            agent_name=spec.name,
            budget_k=budget_k,
            budget_fraction=0.0,
            seed=seed,
            status="error",
            started_at=started_at,
            finished_at=now_iso(),
            error_type=type(exc).__name__,
            error_message=_truncate(str(exc), 500),
        )

    # Wrap the agent invocation in a logging handler scoped to inference,
    # so we count PC-degeneracy fallbacks per cell.
    inference_logger = logging.getLogger("evaluation.chamber_pipeline.inference")
    pc_calls_before = inference_module.PC_INVOCATIONS
    handler = _PcDegeneracyHandler()
    collinear_handler = _PcCollinearHandler()
    zerovar_handler = _PcZeroVarianceHandler()
    inference_logger.addHandler(handler)
    inference_logger.addHandler(collinear_handler)
    inference_logger.addHandler(zerovar_handler)
    # Don't let the handler-level filter override the logger level.
    prev_level = inference_logger.level
    if prev_level > logging.WARNING:
        inference_logger.setLevel(logging.WARNING)

    # Wrap the LLM (user-supplied or lazy-imported litellm.completion)
    # in a per-cell _CountingLLM so n_llm_calls + tokens + cost can be
    # populated uniformly. Non-LLM variants get None for all four.
    kwargs = _build_agent_kwargs(spec, budget_k, seed, pc_alpha, counting_llm, model=model)

    t0 = time.perf_counter()
    _setup_seconds = t0 - _t_cell_start
    try:
        predicted = _invoke_with_timeout(spec.run, adapter, kwargs, cell_timeout_seconds)
        wall = time.perf_counter() - t0

        # Score against ground truth.
        truth = adapter.ground_truth()
        cell_shd = float(shd(predicted, truth))
        cell_f1 = float(f1_edges(predicted, truth))

        # Edge counts (excluding diagonal). These are useful for
        # spotting degenerate "all-zeros" outputs without re-loading
        # the Parquet later.
        n_edges_pred = int(predicted.values.sum() - predicted.values.trace())
        n_edges_truth = int(truth.values.sum() - truth.values.trace())

        finished_at = now_iso()
        _score_seconds = time.perf_counter() - t0 - wall
        (
            n_llm_calls_for_cell,
            tokens_in,
            tokens_out,
            cost_usd,
            n_selection_fallbacks,
            n_llm_attempts,
        ) = _read_llm_metrics(counting_llm)
        model_id, reasoning_effort, providers_used = _read_llm_provenance(counting_llm)

        # Coordination + P2 columns. `getattr` on both: rungs 0 and 3 set
        # neither attribute, and a bare access raises AttributeError on every
        # reused-arm cell.
        coord = getattr(adapter, "coordination_stats", {}) or {}
        cell_graph = getattr(adapter, "delegation_graph", None)
        agg_tokens: int | None = None
        scout_a_tokens: int | None = None
        scout_b_tokens: int | None = None
        frag: int | None = None
        refuse: bool | None = None
        certified: bool | None = None
        if cell_graph is not None:
            from evaluation.chamber_pipeline.tree_accounting import (
                max_tree_fragment,
                tree_would_refuse,
            )

            agg_tokens = cell_graph.monitor_for("aggregator").usage.tokens
            # Per-scout spend, so the next calibration can check whether the
            # SCOUTS overran as well. `c95` is still budget-invariant and that
            # is unverified at k=45; without these fields the question is
            # unanswerable after the fact.
            scout_a_tokens = cell_graph.monitor_for("scout_a").usage.tokens
            scout_b_tokens = cell_graph.monitor_for("scout_b").usage.tokens
            frag = max_tree_fragment(cell_graph, "aggregator")
            # Only score a cell whose aggregator actually spent. On an
            # early-return cell (zero budget, empty menu) it spends nothing,
            # and `tree_would_refuse(..., 0)` is a hard False -- recording a
            # cell that tested nothing as positive evidence that a tree
            # encoding would have coped, which is precisely the diluted
            # refusal rate the function's None contract exists to prevent.
            refuse = (
                tree_would_refuse(cell_graph, "aggregator", agg_tokens) if agg_tokens > 0 else None
            )
            # NOT gated on aggregator spend. `verify()` is graph-wide, so a
            # cell whose scouts overran but whose aggregator reported no
            # usage would drop out of H-C's denominator instead of counting
            # as the failure it is -- biasing the reported compliance rate
            # upward. Only `tree_would_refuse` is aggregator-specific.
            if is_provisional_calibration(
                chamber, budget_k, negotiates=spec.negotiation_rounds > 0
            ):
                # Placeholder budgets produce a verify() result that describes
                # the placeholder, not the framework. Reporting it would put
                # provisioning noise into H-C, which is precisely the error
                # the single `_A95_RECONCILE` made at k=45.
                certified = None
            else:
                try:
                    cell_graph.verify()
                    certified = True
                except ConservationViolationError:
                    certified = False

        # All three PC counters gate on whether `run_pc` ACTUALLY RAN, not on
        # the arm's name. The name test covered `llm_only` and missed the seven
        # agents that return `_empty_adjacency` early -- zero budget, empty
        # menu, or no frames after every selection failed. Those cells recorded
        # 0/0/0 with an all-zeros adjacency, reading in the validity report
        # exactly like a clean PC run. None (not 0) means "not applicable".
        pc_ran = pc_calls_before < inference_module.PC_INVOCATIONS
        n_pc_degen: int | None = handler.count if pc_ran else None
        n_collinear: int | None = collinear_handler.count if pc_ran else None
        n_zerovar: int | None = zerovar_handler.count if pc_ran else None

        return RunRecord(
            **_pc_provenance,
            chamber=chamber,
            configuration=configuration,
            agent_name=spec.name,
            budget_k=budget_k,
            budget_fraction=budget_fraction,
            seed=seed,
            status="ok",
            started_at=started_at,
            finished_at=finished_at,
            shd=cell_shd,
            f1=cell_f1,
            n_edges_predicted=n_edges_pred,
            n_edges_truth=n_edges_truth,
            wall_time_seconds=wall,
            n_llm_calls=n_llm_calls_for_cell,
            n_llm_attempts=n_llm_attempts,
            n_selection_fallbacks=n_selection_fallbacks,
            model_id=model_id,
            reasoning_effort=reasoning_effort,
            providers_used=providers_used,
            overlap_frac=coord.get("overlap_frac"),
            # Falls back to the adapter's own roster, so EVERY arm reports it.
            # Only the multi-agent agents set `coordination_stats`, so the
            # reference arm -- the loop -- reported None and dropped out of any
            # analysis that grouped on this column: the baseline missing from
            # its own comparison. Derived, not duplicated: the roster is the
            # single source and this is its distinct count.
            n_experiments_distinct=coord.get(
                "n_experiments_distinct", len(set(adapter.purchased)) or None
            ),
            n_contested=coord.get("n_contested"),
            n_negotiation_failures=coord.get("n_negotiation_failures"),
            n_claim_truncated=coord.get("n_claim_truncated"),
            n_substring_conflicts=coord.get("n_substring_conflicts"),
            chosen_experiments=(",".join(adapter.purchased) or None),
            claim_pool_share=coord.get("claim_pool_share"),
            conservation_certified=certified,
            aggregator_tokens=agg_tokens,
            scout_a_tokens=scout_a_tokens,
            scout_b_tokens=scout_b_tokens,
            max_tree_fragment=frag,
            tree_would_refuse=refuse,
            n_pc_degeneracies=n_pc_degen,
            n_collinear_dropped=n_collinear,
            n_zero_variance_dropped=n_zerovar,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            extra={
                "setup_seconds": round(_setup_seconds, 2),
                "score_seconds": round(_score_seconds, 2),
                # Aggregator-ablation diagnostics (`fan_in_agg` only). Absent
                # for every other arm, which is why they live in `extra`
                # rather than becoming columns that are null 8 times in 9.
                **{k: v for k, v in coord.items() if k.startswith("agg_")},
            },
        )
    except NotImplementedError as exc:
        # Defensive: the registry already filtered this, but the agent
        # may have its own compatibility check (e.g., greedy_ig_lite's
        # menu-parse guard). Treat as a skip rather than an error so
        # the M5 figure doesn't show this as a failure.
        return RunRecord(
            **_pc_provenance,
            chamber=chamber,
            configuration=configuration,
            agent_name=spec.name,
            budget_k=budget_k,
            budget_fraction=budget_fraction,
            seed=seed,
            status="skipped",
            started_at=started_at,
            finished_at=now_iso(),
            skip_reason=_truncate(str(exc), 500),
        )
    except SweepConfigurationError:
        # The SECOND place this has to re-raise. `_provider_order_for`'s
        # unpinned-model raise fires inside `_CountingLLM.__call__`, i.e. in
        # this block and not the adapter-construction one -- so guarding only
        # the first left `--model <unlisted-id>` producing N identical error
        # rows that every resume re-attempts, the exact loop the class exists
        # to prevent.
        raise
    except Exception as exc:
        return RunRecord(
            **_pc_provenance,
            chamber=chamber,
            configuration=configuration,
            agent_name=spec.name,
            budget_k=budget_k,
            budget_fraction=budget_fraction,
            seed=seed,
            status="error",
            started_at=started_at,
            finished_at=now_iso(),
            error_type=type(exc).__name__,
            error_message=_truncate(str(exc), 500),
            extra={"traceback": _truncate(traceback.format_exc(), 2000)},
        )
    finally:
        inference_logger.removeHandler(handler)
        inference_logger.removeHandler(collinear_handler)
        inference_logger.removeHandler(zerovar_handler)
        inference_logger.setLevel(prev_level)


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------


@dataclass
class SweepSpec:
    """Parameters defining a sweep — the §6.1 cell grid for one experiment.

    The orchestrator iterates the Cartesian product of `chambers x
    budget_fractions x agent_names x seeds`, dispatching each cell
    via `run_cell`. AGENT_REGISTRY is the source of truth for which
    agents exist; `agent_names` is a filter, defaulting to "all".

    Attributes:
        chambers: Chambers to sweep. Default both LT and WT.
        budget_fractions: k/M values to test. Plan §6.1 default
            five-level: (0.10, 0.25, 0.50, 0.75, 1.00). M4 pilot
            uses three: (0.10, 0.50, 1.00).
        agent_names: Variants to include. None = all from registry.
        seeds: Range of RNG seeds. Default 30 per plan §6.1.
        configuration: Chamber configuration. "standard" per §6.1.
        pc_alpha: PC independence-test significance level.
        cell_timeout_seconds: Optional per-cell wall-clock timeout
            forwarded to `run_cell`. None = no timeout. M4b's pilot
            uses None (LLM calls are typically <30s); M5 should set
            this to ~120s to recover from rare API hangs without
            losing the surrounding sweep.
    """

    chambers: tuple[ChamberId, ...] = ("lt", "wt")
    budget_fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.00)
    agent_names: tuple[str, ...] | None = None
    seeds: tuple[int, ...] = tuple(range(30))
    configuration: ConfigId = "standard"
    pc_alpha: float = 0.05
    cell_timeout_seconds: float | None = None

    def selected_specs(self) -> list[AgentSpec]:
        """The AgentSpec list this sweep will dispatch (filtered by agent_names)."""
        if self.agent_names is None:
            return list(AGENT_REGISTRY)
        names_set = set(self.agent_names)
        return [s for s in AGENT_REGISTRY if s.name in names_set]


# Plan §3.2 menu sizes (intervention catalogs per chamber). Used to convert
# budget fraction → integer k. Centralized here so the orchestrator doesn't
# need to load the chamber dataset just to compute k.
MENU_SIZES: dict[ChamberId, int] = {"lt": 59, "wt": 28}


def _budget_k_for(chamber: ChamberId, fraction: float) -> int:
    """Convert a fractional budget to integer k for a chamber.

    Round-half-to-even for stability (Python's default int(x + 0.5) is
    actually banker's rounding via round()). Clamp to [1, menu_size].
    """
    menu = MENU_SIZES[chamber]
    k = round(fraction * menu)
    return max(1, min(menu, k))


def iter_sweep_cells(
    sweep: SweepSpec,
) -> Iterator[tuple[AgentSpec, ChamberId, int, float, int]]:
    """Iterate cells of a sweep as (spec, chamber, budget_k, fraction, seed).

    Pure: doesn't load chambers or invoke agents. Useful for sizing
    a sweep ("how many cells will I run?") and for the CLI's dry-run
    mode.
    """
    specs = sweep.selected_specs()
    for chamber in sweep.chambers:
        for fraction in sweep.budget_fractions:
            budget_k = _budget_k_for(chamber, fraction)
            for spec in specs:
                for seed in sweep.seeds:
                    yield spec, chamber, budget_k, fraction, seed


def count_cells(sweep: SweepSpec, *, exclude_skipped: bool = False) -> int:
    """How many cells `iter_sweep_cells` will yield.

    Args:
        sweep: The sweep spec.
        exclude_skipped: If True, exclude cells the registry would
            skip due to chamber-incompatibility. Useful for sizing
            "how many real runs" vs "how many cells the orchestrator
            will iterate through."
    """
    if not exclude_skipped:
        return sum(1 for _ in iter_sweep_cells(sweep))
    return sum(1 for spec, chamber, *_ in iter_sweep_cells(sweep) if spec.is_compatible(chamber))


def _pc_provenance_snapshot(pc_alpha: float) -> dict[str, Any]:
    """The PC/platform stamp for a record the PARENT synthesizes.

    A worker that died produced no cell of its own, but the row still has to
    carry provenance or it becomes the one row in the frame whose backend is
    unknown -- and register entry 10 forbids pooling across backends. These
    are the parent's values, which is correct: `pc_alpha` aside, the backend
    and platform are properties of the machine, and every worker is forked
    from this process.
    """
    return {
        # Taken from the payload, never hardcoded: `run_cell` stamps
        # `sweep.pc_alpha`, so a literal here would give the synthesized row a
        # different alpha from all the others -- and since `pc_alpha` is a
        # provenance column checked BEFORE non-ok rows are dropped, one dead
        # worker would make the entire Parquet raise MixedProvenanceError.
        "pc_alpha": pc_alpha,
        "pc_max_rows": _PC_CALL_DEFAULTS["max_rows"],
        "pc_collinearity_threshold": _PC_CALL_DEFAULTS["collinearity_threshold"],
        "blas_backend": _RUNTIME["blas"],
        "platform_tag": _RUNTIME["platform"],
    }


def _run_cell_in_worker(
    args: tuple[str, ChamberId, ConfigId, int, int, float, float | None, str | None],
) -> RunRecord:
    """Child-process entry point for the parallel sweep.

    Takes the spec NAME, not the `AgentSpec`. `AgentSpec.static_kwargs` is a
    `MappingProxyType`, which is not picklable, so shipping the dataclass to
    a worker fails at submit time. The child re-resolves through the registry
    instead -- the same object the parent would have used.
    """
    spec_name, chamber, configuration, budget_k, seed, pc_alpha, timeout, model = args
    return run_cell(
        spec=get_spec(spec_name),
        chamber=chamber,
        configuration=configuration,
        budget_k=budget_k,
        seed=seed,
        pc_alpha=pc_alpha,
        cell_timeout_seconds=timeout,
        model=model,
    )


def run_sweep(
    sweep: SweepSpec,
    llm: LLMCallable | None = None,
    on_cell: Callable[[RunRecord, int, int], None] | None = None,
    skip_keys: set[tuple[str, str, str, int, int]] | None = None,
    model: str | None = None,
    max_workers: int | None = None,
) -> list[RunRecord]:
    """Run a full sweep and return all RunRecords.

    Serial nested loop. Per-cell exceptions are captured into the
    RunRecord's `error_type` / `error_message` — the sweep itself
    never raises mid-flight.

    Args:
        sweep: The cell-grid spec.
        llm: LLM callable threaded into LLM-bearing agents. Pass
            `litellm.completion` for production sweeps; FakeLLM for
            tests; None to let agents use their own default
            (which lazy-imports litellm).
        on_cell: Optional progress callback invoked after each cell
            completes, with `(record, idx, total)`. The CLI uses
            this for tqdm-style progress bars.
        skip_keys: Optional set of `(chamber, configuration,
            agent_name, budget_k, seed)` tuples to skip. Used by the
            CLI's resume-from-checkpoint logic so cells already in
            the JSONL sidecar aren't re-run.

    Returns:
        One RunRecord per cell, in iteration order (over the
        post-filter cell list).
    """
    from .checkpoint import filter_done_cells

    raw_cells: Iterable[tuple[AgentSpec, ChamberId, int, float, int]] = iter_sweep_cells(sweep)
    if skip_keys:
        raw_cells = filter_done_cells(raw_cells, skip_keys, configuration=sweep.configuration)
    cells = list(raw_cells)
    total = len(cells)
    if max_workers is not None and max_workers > 1:
        return _run_sweep_parallel(sweep, cells, on_cell, model, max_workers, llm)

    records: list[RunRecord] = []
    for idx, (spec, chamber, budget_k, _fraction, seed) in enumerate(cells):
        record = run_cell(
            spec=spec,
            chamber=chamber,
            configuration=sweep.configuration,
            budget_k=budget_k,
            seed=seed,
            pc_alpha=sweep.pc_alpha,
            llm=llm,
            cell_timeout_seconds=sweep.cell_timeout_seconds,
            model=model,
        )
        records.append(record)
        if on_cell is not None:
            on_cell(record, idx, total)
    return records


def order_by_cell_index(completed: list[tuple[int, RunRecord]]) -> list[RunRecord]:
    """Restore cell-grid order from `(cell_index, record)` pairs.

    `as_completed` yields whichever worker finished first, so without this
    the Parquet row order would depend on scheduling -- two identical sweeps
    would produce byte-different files and a resumed sweep would interleave
    differently from a fresh one.

    Kept as a pure function rather than inline because the inline version was
    untestable: with fast uniform cells, completion order happens to equal
    submission order, so a test against a live pool passes whether or not the
    reordering exists.
    """
    return [record for _, record in sorted(completed, key=lambda pair: pair[0])]


def _run_sweep_parallel(
    sweep: SweepSpec,
    cells: list[tuple[AgentSpec, ChamberId, int, float, int]],
    on_cell: Callable[[RunRecord, int, int], None] | None,
    model: str | None,
    max_workers: int,
    llm: LLMCallable | None,
) -> list[RunRecord]:
    """Run `cells` across a process pool, returning them in CELL order.

    Processes, not threads, for two reasons that both corrupt data silently
    rather than raising:

      * `_PcDegeneracyHandler` attaches to the global
        `evaluation.chamber_pipeline.inference` logger for a cell's duration.
        Concurrent cells in one process would each receive every other cell's
        records, so `n_pc_degeneracy` would count the whole pool.
      * `_invoke_with_timeout` leaks a daemon thread on every timeout, since
        a worker stuck in openssl's `SSL_read` cannot be cancelled. In one
        process those accumulate for the whole sweep; per worker they stay
        bounded and the worker keeps serving.

    `on_cell` is invoked in the PARENT, once per completed cell. That keeps
    the CLI's checkpoint-sidecar append single-threaded, so the checkpoint
    layer needs no locking of its own.

    Returns records ordered by the cell grid, never by completion order --
    Parquet row order must not depend on which worker finished first.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool

    if llm is not None:
        raise ValueError(
            "max_workers > 1 cannot ship a custom `llm` to worker processes "
            "(callables such as FakeLLM are generally not picklable). Pass "
            "llm=None so each worker lazy-imports litellm, or run serially."
        )

    total = len(cells)
    payloads = [
        (
            spec.name,
            chamber,
            sweep.configuration,
            budget_k,
            seed,
            sweep.pc_alpha,
            sweep.cell_timeout_seconds,
            model,
        )
        for (spec, chamber, budget_k, _fraction, seed) in cells
    ]

    completed: list[tuple[int, RunRecord]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_cell_in_worker, p): i for i, p in enumerate(payloads)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                record = fut.result()
            except SweepConfigurationError:
                # A mis-configured sweep must still abort. Every remaining
                # cell would raise the same way; failing now shows the
                # operator the message instead of 450 identical error rows.
                raise
            except BrokenProcessPool as exc:
                # NOT a per-cell fault. Once the pool is broken every
                # remaining future raises the same thing, so synthesizing a
                # record each time turns a sweep that died at cell 250 into an
                # instant "completion" with 200 fabricated error rows and exit
                # 0 -- a success-shaped ending for a run that did a fraction of
                # its work. Completed cells are already in the sidecar, so
                # aborting loses nothing and resume picks up correctly.
                raise RuntimeError(
                    f"the worker pool died after {len(completed)} of {total} "
                    f"cells ({exc}). Completed cells are in the sidecar; "
                    "re-run the same command to resume. If this was an OOM, "
                    "lower --max-workers (budget ~700 MB per worker)."
                ) from exc
            except Exception as exc:
                # A genuine per-cell fault: the worker survived, this cell did
                # not. `Exception`, not `BaseException`, so KeyboardInterrupt
                # still aborts the sweep.
                # `run_cell` converts in-cell exceptions to error records, so
                # reaching here means the WORKER died: OOM-killed (the
                # docstring budgets ~700 MB/worker on an 8 GB VPS), or a
                # result that failed to pickle. Uncaught, that BrokenProcessPool
                # propagates out of the `with ProcessPoolExecutor(...)` block,
                # whose `__exit__` calls `shutdown(wait=True)` -- the same
                # wait-on-exit shape as the ThreadPoolExecutor trap this
                # project root-caused in M4b -- and every in-flight cell is
                # discarded with no sidecar line. A 20-hour sweep would die at
                # hour 12 over one worker fault. Synthesize the error record
                # the serial path would have produced and keep going.
                spec_name, chamber, configuration, budget_k, seed = payloads[i][:5]
                cell_pc_alpha = payloads[i][5]
                record = RunRecord(
                    **_pc_provenance_snapshot(cell_pc_alpha),
                    chamber=chamber,
                    configuration=configuration,
                    agent_name=spec_name,
                    budget_k=budget_k,
                    budget_fraction=0.0,
                    seed=seed,
                    status="error",
                    started_at=now_iso(),
                    finished_at=now_iso(),
                    error_type=type(exc).__name__,
                    error_message=_truncate(f"worker process died: {exc}", 500),
                )
            completed.append((i, record))
            if on_cell is not None:
                # `len(completed) - 1`, not `i`: the callback's index is a
                # progress counter, and out-of-order cell indices would make
                # the CLI's ETA jump backwards.
                on_cell(record, len(completed) - 1, total)

    return order_by_cell_index(completed)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _truncate(text: str, n: int) -> str:
    """Cap a string at n characters, appending an ellipsis suffix."""
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


def _invoke_with_timeout(
    target: Callable[..., Any],
    adapter: Any,
    kwargs: dict[str, Any],
    timeout: float | None,
) -> Any:
    """Invoke `target(adapter, **kwargs)`, optionally with a wall-clock timeout.

    With timeout=None: direct call (zero overhead — relevant for the
    common case of the M4b/M5 pilot, where most cells complete in <1s).

    With a timeout: dispatch via a daemon `threading.Thread` and
    `thread.join(timeout=...)`. We deliberately do NOT use
    `with ThreadPoolExecutor(...) as exe:` because the context manager
    calls `shutdown(wait=True)` on exit, which blocks indefinitely if
    the worker is stuck in a non-cancellable C-level call (e.g.,
    openssl's SSL_read on a hung TLS socket). This was the root cause
    of the M4b pilot hangs (2026-05-14, 2026-05-15): the worker
    couldn't return because httpx wasn't honoring our socket-level
    timeout, and the main thread couldn't escape because the context
    manager's __exit__ waited for the worker.

    daemon=True ensures Python's process-exit atexit handler doesn't
    block on the leaked thread. The leaked worker sits idle (Python
    has no thread cancellation and openssl ignores signals) until
    process exit. For the serial sweep this is acceptable; M5
    parallelism would need process-level isolation.

    On timeout, raises `TimeoutError` so `run_cell`'s outer
    `except Exception` records the cell as `status="error"` with
    `error_type="TimeoutError"`.
    """
    if timeout is None:
        return target(adapter, **kwargs)

    result_box: list[Any] = []
    error_box: list[BaseException] = []

    def _runner() -> None:
        try:
            result_box.append(target(adapter, **kwargs))
        except BaseException as exc:
            error_box.append(exc)

    worker = threading.Thread(target=_runner, daemon=True)
    worker.start()
    worker.join(timeout=timeout)

    if worker.is_alive():
        raise TimeoutError(f"cell exceeded {timeout}s wall-clock timeout")
    if error_box:
        raise error_box[0]
    return result_box[0]


def _read_llm_provenance(
    counting_llm: _CountingLLM | None,
) -> tuple[str | None, str | None, str | None]:
    """Extract (model_id, reasoning_effort, providers_used) from a wrapper.

    Each is a sorted comma-joined string when more than one distinct value was
    seen in the cell -- provider rotation makes that the normal case -- and
    None when the wrapper is absent or saw no calls.
    """
    if counting_llm is None or not counting_llm.calls:
        return None, None, None

    def joined(values: set[str]) -> str | None:
        return ",".join(sorted(values)) if values else None

    return (
        joined(counting_llm.observed_models),
        joined(counting_llm.observed_efforts),
        joined(counting_llm.observed_providers),
    )


def _read_llm_metrics(
    counting_llm: _CountingLLM | None,
) -> tuple[int | None, int | None, int | None, float | None, int | None, int | None]:
    """Extract (n_llm_calls, tokens_in, tokens_out, cost_usd, fallbacks, attempts).

    `n_llm_calls` is LOGICAL calls -- one per `__call__` -- which is what the
    name implies and what every recorded figure means (6/30/59 at the three
    M4b budgets). `n_llm_attempts` adds provider-rotation retries, and is the
    cost-attribution number. They are equal in every cell recorded before
    2026-08-29, because rotation never fired: 0 extra attempts across all 450
    loop cells.

    Returns a 6-tuple of Nones when the wrapper is None
    (non-LLM variant). When the wrapper saw at least one call, all
    four are populated — even if the wrapped target reported zero
    tokens (e.g., FakeLLM). When the wrapper saw zero calls (LLM
    variant ran a budget=0 short-circuit path), n_llm_calls=0 is
    populated but token / cost fields stay None to keep "tracked
    zero" distinguishable from "no measurement."
    """
    if counting_llm is None:
        return None, None, None, None, None, None
    n = counting_llm.n_requests
    if n == 0:
        return 0, None, None, None, 0, len(counting_llm.calls)
    return (
        n,
        counting_llm.total_input_tokens,
        counting_llm.total_output_tokens,
        counting_llm.total_cost_usd,
        counting_llm.selection_fallbacks,
        len(counting_llm.calls),
    )


__all__ = [
    "AGENT_REGISTRY",
    "MENU_SIZES",
    "AgentSpec",
    "SweepConfigurationError",
    "SweepSpec",
    "count_cells",
    "get_spec",
    "iter_sweep_cells",
    "run_cell",
    "run_sweep",
]
