# M7: the mechanism, the missing arms, and the two reviewer objections

**Status**: planning, opened 2026-08-29
**Predecessor**: M6 coordination ladder (`2026-08-22-m6-coordination-ladder-design.md`)
**Results of record**: `docs/chamber-results.md`
**Harness defects**: `docs/chamber-harness-validity-register.md` (17 entries — read first)

## 1. Why there is an M7

M6 answered *what*: of 24 topology-vs-loop contrasts, 10 resolve and 9 favour a
single sequential loop. It did not answer *why*, and it left the comparison
axis open at both ends.

Three gaps, in the order they hurt:

1. **We cannot explain the cause.** `team` reaches identical 30/30 distinct
   coverage against the loop and still scores −0.047. The explanation must lie
   in *which* experiments it bought — and until 2026-08-29 nothing recorded
   them. All 2,050 existing cells hold only `n_experiments_distinct`.
2. **The axis has no endpoints.** Every multi-agent rung *divides* the loop's
   running record. Nothing established what an undivided record is worth
   (`one_shot`) or what a second agent costs when it divides nothing
   (`critique`).
3. **Two objections have no answer.** "Your agents are the same model" and
   "real scientists look at results." Both are foreseeable and both are cheap
   to address.

## 2. The reframing this plan is built on

The ladder is not varying *topology*. It varies **how much of the loop's
running record survives**:

| arm | record | status |
|---|---|---|
| `one_shot` | none — whole budget in one call | built 2026-08-29, unrun |
| `llm_pc` (loop) | complete | M6 ✓ |
| `critique` | complete, plus a reviewer holding no budget | built 2026-08-29, unrun |
| `shared_blackboard` | complete, two voices alternating | **not built** |
| `planner_reasoner` (relay) | complete, one seam | M6 ✓ |
| `team` | split by agreement | M6 ✓ |
| `fan_in_spec` / `fan_in_homog` | split blind | M6 ✓ |

Stated as a claim to be tested: *multi-agent structure is free when agents
share the record of what has been done, and costly when they do not. The
measured cost is not of having several agents; it is of partitioning their
information.*

`shared_blackboard` is the sharpest test and the one most likely to collapse
into the loop by construction — two agents alternating with full shared
history IS the loop with two voices. Build it only if Phase 2 leaves the axis
ambiguous.

## 3. Naming

`planner_reasoner` is misleading: nothing is planned. It is one loop with a
seam at the midpoint and a different system prompt on each half — which is
exactly why it resolves in neither direction in all six of its contrasts.

**Rename the paper-facing label to `relay`; keep the code identifier.** The
identifier is the `agent_name` value in 450+ recorded rows and renaming it
would orphan the data. One line in `VARIANT_LABELS`.

## 4. Phases

### Phase 1 — mechanism (~$5, highest value)

**Question**: why does `team` lose at equal coverage?

Three hypotheses, currently indistinguishable because they predict the same
symptom:

- **H1 experiments ≠ variables.** 30 distinct menu entries can touch 30
  variables or 20, since one variable carries up to three entries
  (mid/strong/weak). Team may be buying depth where the loop bought breadth.
- **H2 forced allocation.** Scout pools are disjoint by construction, so each
  scout must spend exactly ⌈k/2⌉ inside its own half even if that half
  deserves less. The loop allocates all k adaptively. The cost would then be
  *committing to a division of labour before knowing where the work is* —
  which is a more interesting finding than "coordination is expensive".
- **H3 blind depth duplication.** Zero overlap at the *experiment* level is
  compatible with redundancy at the *variable* level.

**Instruments** (both shipped 2026-08-29):
`chosen_experiments` (roster, spending order, recorded at the adapter so no
agent can forget it) and `n_zero_variance_dropped` (how many variables never
moved — H1 and H3 in the algorithm's own terms).

**Sweep**: LT, `llm_pc` + `team`, k=30, n=10. ~$2.
**Analysis**: distinct *variables* touched per arm; breadth/depth mix;
family distribution across the two scout pools; padding count per arm.

**Decision rule**: if team touches materially fewer distinct variables at
equal experiment count → H1/H3. If variable coverage matches but the
family distribution is lopsided across pools → H2. If neither, the cause is
not in what was bought and the ladder needs a different instrument.

### Phase 2 — close the axis (~$8)

Run `one_shot` and `critique` on both chambers at the M6 budgets and seeds.
Both are built and unit-tested; neither has run against a live model at scale.

**Pre-registered predictions**, so the result is falsifiable:

- `one_shot` < `loop`. If it MATCHES, the running record was never
  load-bearing and the headline weakens substantially — the honest claim
  becomes "how selection is organised barely matters." We want to find that
  ourselves rather than be shown it.
- `critique` ≈ `loop` on accuracy at ~1/10th the calls. A reviewer that costs
  three flat calls and does not divide the budget is the cheapest multi-agent
  shape on the ladder; if it holds accuracy, that is a positive result and the
  only one available to us.

### Phase 3 — the two objections (~$15)

**3a. Mixed-model team.** `fan_in_homog`'s two scouts differ only by sampling
temperature — the same opinion drawn twice, which is a fair criticism.
`fan_in_spec` already answers it partly (prompted breadth-vs-depth produced
REAL divergence: overlap 0.79 → 0.32, coverage 27.6 → 38.0 of 45, and still
lost). A mixed-model team is the stronger form: genuine architectural
difference rather than sampling noise.

Requires a second provider wired into `PROVIDER_ORDER_BY_MODEL` — unpinned
models now raise by design. **Probe first**: availability, price, provider
endpoints and quantization for each candidate before committing.

**3b. Adaptive feedback — one arm, as a bound, not a branch.** Giving agents
results after each pick changes the question from *one-shot design quality* to
*closed-loop experimentation*, and would confound every topology contrast,
since feedback is exactly what fan-in cannot share. So: **loop + feedback, LT
only, one budget.** If it beats the no-feedback loop we report the headroom
and scope the paper to open-loop design. If it does not, the objection dies.

Design care needed: what feedback? The honest minimum is the *current
adjacency estimate* after each purchase — cheap to compute, and it is what a
scientist would actually look at. Not the score, which does not exist at run
time without the answer key.

### Phase 4 — rewrite

Rename `relay`; fold the mechanism into the results doc; scope the headline to
the axis rather than to "topology"; state the untested shapes (critique with
iteration, adversarial disagreement, >2 agents) as scope limits.

## 5. Sequencing and the reason for it

**Phase 1 → look → then decide 2 and 3.** Each new arm is another chance to
find a harness defect: 17 recorded so far, and three found today were in code
written the same day. The mechanism result may also reframe what is worth
running — if the cause turns out to be forced allocation, `shared_blackboard`
becomes the most interesting arm on the list rather than an optional one.

## 6. Non-goals

- Not a causal-discovery methods paper. PC on pooled interventional data stays
  fixed and mis-specified-but-uniform; improving it would change every number
  and answer a different question.
- Not >2 agents. The axis is about partitioning information, not scale.
- Not a cross-vendor model sweep. One additional model family answers the
  objection; a sweep is a different paper.
