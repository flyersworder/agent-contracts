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
from typing import Any

from agent_contracts.core.delegation import ConservationViolationError
from agent_contracts.integrations.causalchamber import (
    ChamberId,
    ConfigId,
    create_contracted_chamber_agent,
)

from .agents import (
    fan_in_agents,
    greedy_ig_lite_agent,
    llm_only_agent,
    llm_pc_agent,
    planner_reasoner_agents,
    random_agent,
    team_agents,
)
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
    static_kwargs: Mapping[str, Any] = field(default_factory=dict)

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
    # ---- M6 coordination ladder -------------------------------------------
    # Rungs 1, 2 and 4. Rungs 0 (`llm_pc`) and 3 (`planner_reasoner`) are
    # reused from the M4b pilot unchanged, which is why nothing above this
    # comment may move.
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
        static_kwargs={"differentiate": True},
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
)


def get_spec(name: str) -> AgentSpec:
    """Look up an AgentSpec by name. KeyError on unknown name."""
    for spec in AGENT_REGISTRY:
        if spec.name == name:
            return spec
    raise KeyError(
        f"Unknown agent name: {name!r}. Available: {sorted(s.name for s in AGENT_REGISTRY)}"
    )


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

    # OpenRouter's `provider.order` preference for `deepseek-v4-flash`.
    # Provider performance is genuinely dynamic across days: yesterday
    # (2026-05-14) Parasail was the fastest (~0.5-1.3s/call) and AtlasCloud
    # was throttling badly. Today (2026-05-15) Parasail dropped to ~9 t/s
    # while Novita and AtlasCloud both serve at ~70-80 t/s. At
    # `_ADJACENCY_MAX_TOKENS=32768` with ~95% reasoning load on DeepSeek
    # v4 Flash, a 9 t/s provider takes ~55 min per cell vs ~6 min on a
    # 70 t/s provider — every cell on Parasail would time out at 1800s.
    # Order is therefore Novita + AtlasCloud first (consistently fast today,
    # both fp8), Parasail + SiliconFlow as fallbacks. Rotation in `__call__`
    # advances on `finish_reason: 'error'` and the cell-timeout safety net
    # bounds any genuinely-slow provider at `cell_timeout_seconds`.
    # DeepInfra (fp4) remains excluded to keep the inference-precision
    # class constant for AAMAS reproducibility.
    DEFAULT_PROVIDER_ORDER: tuple[str, ...] = (
        "Novita",
        "AtlasCloud",
        "Parasail",
        "SiliconFlow",
    )

    # Per-request socket-level timeout (seconds). Without this, a single
    # litellm.completion call can BLOCK FOREVER on the underlying SSL
    # socket read when the upstream provider accepts the request but
    # stops sending bytes mid-response. Discovered via root-cause
    # systematic debugging during M4b smoke: a stuck call kept the
    # python process at 0% CPU for 12+ minutes inside `_ssl__SSLSocket_read`.
    # 30s is generous for normal completions (~1-15s with retries) and
    # short enough that a stuck call surfaces a TimeoutError that
    # `num_retries` can backoff-retry against. num_retries handles
    # *exceptions*; without timeout, hangs never raise → retries
    # never trigger → process appears wedged. Both are required.
    DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0

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

        # Provider rotation loop. OpenRouter's `allow_fallbacks: True`
        # cycles through `provider.order` when the upstream provider
        # returns an HTTP error — but **NOT** when the provider returns
        # 200 with `finish_reason: 'error'` in the response body
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
        provider_order = list(self.DEFAULT_PROVIDER_ORDER)
        max_attempts = len(provider_order)
        response: Any = None
        for attempt in range(max_attempts):
            kwargs["extra_body"] = {
                **existing_extra,
                "provider": {
                    "order": provider_order,
                    "allow_fallbacks": True,
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

_ROLE_C95: dict[str, int] = {"plain": 2205, "broad": 3003, "targeted": 10379}
_A95_RECONCILE = 8557
_C95_NEGOTIATE = 4138

_LADDER_NODES = ("scout_a", "scout_b", "aggregator")


def _ladder_calibration(spec: AgentSpec) -> tuple[int, int, int, int]:
    """`(c95_a, c95_b, a95, fixed_overhead)` for one ladder arm.

    Read off the spec rather than matched on the arm's name: per-call cost is
    driven by the role's prompt, not by `k`, and a name-keyed lookup silently
    returns plain-role figures for any arm it does not recognise.
    """
    if spec.scout_roles is None:
        raise ValueError(f"{spec.name!r} is not a ladder arm")
    role_a, role_b = spec.scout_roles
    overhead = spec.negotiation_rounds * (_PROVISION_MULTIPLE * _C95_NEGOTIATE)
    return _ROLE_C95[role_a], _ROLE_C95[role_b], _A95_RECONCILE, overhead


def _build_agent_kwargs(
    spec: AgentSpec,
    budget_k: int,
    seed: int,
    pc_alpha: float,
    llm: LLMCallable | None,
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

    return kwargs


def run_cell(
    spec: AgentSpec,
    chamber: ChamberId,
    configuration: ConfigId,
    budget_k: int,
    seed: int,
    pc_alpha: float = 0.05,
    llm: LLMCallable | None = None,
    cell_timeout_seconds: float | None = None,
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
    started_at = now_iso()

    # Pre-flight: is this agent compatible with this chamber?
    if not spec.is_compatible(chamber):
        return RunRecord(
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

            c95_a, c95_b, a95, overhead = _ladder_calibration(spec)
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
        adapter = create_contracted_chamber_agent(
            chamber=chamber,
            configuration=configuration,
            intervention_budget=budget_k,
            **extra,
        )
        if graph is not None:
            adapter.delegation_graph = graph  # read back by the P2 scorer
        menu_size = len(adapter.available_experiments())
        budget_fraction = _budget_fraction(budget_k, menu_size)
    except Exception as exc:
        return RunRecord(
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
    handler = _PcDegeneracyHandler()
    inference_logger.addHandler(handler)
    # Don't let the handler-level filter override the logger level.
    prev_level = inference_logger.level
    if prev_level > logging.WARNING:
        inference_logger.setLevel(logging.WARNING)

    # Wrap the LLM (user-supplied or lazy-imported litellm.completion)
    # in a per-cell _CountingLLM so n_llm_calls + tokens + cost can be
    # populated uniformly. Non-LLM variants get None for all four.
    kwargs = _build_agent_kwargs(spec, budget_k, seed, pc_alpha, counting_llm)

    t0 = time.perf_counter()
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
        (
            n_llm_calls_for_cell,
            tokens_in,
            tokens_out,
            cost_usd,
            n_selection_fallbacks,
        ) = _read_llm_metrics(counting_llm)
        model_id, reasoning_effort, providers_used = _read_llm_provenance(counting_llm)

        # Coordination + P2 columns. `getattr` on both: rungs 0 and 3 set
        # neither attribute, and a bare access raises AttributeError on every
        # reused-arm cell.
        coord = getattr(adapter, "coordination_stats", {}) or {}
        cell_graph = getattr(adapter, "delegation_graph", None)
        agg_tokens: int | None = None
        frag: int | None = None
        refuse: bool | None = None
        certified: bool | None = None
        if cell_graph is not None:
            from evaluation.chamber_pipeline.tree_accounting import (
                max_tree_fragment,
                tree_would_refuse,
            )

            agg_tokens = cell_graph.monitor_for("aggregator").usage.tokens
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
            if agg_tokens > 0:
                try:
                    cell_graph.verify()
                    certified = True
                except ConservationViolationError:
                    certified = False

        # PC variants populate degeneracy count; llm_only doesn't run PC.
        n_pc_degen: int | None = None if spec.name == "llm_only" else handler.count

        return RunRecord(
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
            n_selection_fallbacks=n_selection_fallbacks,
            model_id=model_id,
            reasoning_effort=reasoning_effort,
            providers_used=providers_used,
            overlap_frac=coord.get("overlap_frac"),
            n_experiments_distinct=coord.get("n_experiments_distinct"),
            n_contested=coord.get("n_contested"),
            conservation_certified=certified,
            aggregator_tokens=agg_tokens,
            max_tree_fragment=frag,
            tree_would_refuse=refuse,
            n_pc_degeneracies=n_pc_degen,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
        )
    except NotImplementedError as exc:
        # Defensive: the registry already filtered this, but the agent
        # may have its own compatibility check (e.g., greedy_ig_lite's
        # menu-parse guard). Treat as a skip rather than an error so
        # the M5 figure doesn't show this as a failure.
        return RunRecord(
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
    except Exception as exc:
        return RunRecord(
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


def run_sweep(
    sweep: SweepSpec,
    llm: LLMCallable | None = None,
    on_cell: Callable[[RunRecord, int, int], None] | None = None,
    skip_keys: set[tuple[str, str, str, int, int]] | None = None,
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
        )
        records.append(record)
        if on_cell is not None:
            on_cell(record, idx, total)
    return records


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
) -> tuple[int | None, int | None, int | None, float | None, int | None]:
    """Extract (n_llm_calls, tokens_in, tokens_out, cost_usd, fallbacks).

    Returns a 5-tuple of Nones when the wrapper is None
    (non-LLM variant). When the wrapper saw at least one call, all
    four are populated — even if the wrapped target reported zero
    tokens (e.g., FakeLLM). When the wrapper saw zero calls (LLM
    variant ran a budget=0 short-circuit path), n_llm_calls=0 is
    populated but token / cost fields stay None to keep "tracked
    zero" distinguishable from "no measurement."
    """
    if counting_llm is None:
        return None, None, None, None, None
    n = len(counting_llm.calls)
    if n == 0:
        return 0, None, None, None, 0
    return (
        n,
        counting_llm.total_input_tokens,
        counting_llm.total_output_tokens,
        counting_llm.total_cost_usd,
        counting_llm.selection_fallbacks,
    )


__all__ = [
    "AGENT_REGISTRY",
    "MENU_SIZES",
    "AgentSpec",
    "SweepSpec",
    "count_cells",
    "get_spec",
    "iter_sweep_cells",
    "run_cell",
    "run_sweep",
]
