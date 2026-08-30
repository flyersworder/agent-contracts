# M7: the mechanism, the missing arms, and the two reviewer objections

**Status**: planning, opened 2026-08-29
**Predecessor**: M6 coordination ladder (`2026-08-22-m6-coordination-ladder-design.md`)
**Results of record**: `docs/chamber-results.md`
**Harness defects**: `docs/chamber-harness-validity-register.md` (19 entries — read first)

## 1. Positioning: what this contributes to the loop-vs-graph debate

Written down because it is the reason the phases below are worth their cost,
and because it changes what Phase 4 has to say.

### The open slot

**Loop engineering** (named 2026-06-07) and **graph engineering** (~2026-07-18)
put multi-agent topology on the field's agenda. Critical reviews of both land
on the same complaint: **there is no comparative benchmark.** Both camps argue
from production anecdote, and the loudest advocates for graphs have vendor
stakes. "Someone should measure this" is the open slot.

The reason nobody has filled it is not lack of interest. **The obvious
comparison is confounded and hard to de-confound**: in every multi-agent-vs-
single comparison in the wild, the multi-agent system is allowed to spend more
— more calls, more tokens, more tool invocations. When the graph wins you
cannot tell whether topology helped or whether it simply bought more; when the
loop wins, the same problem inverted.

### What we can claim, and what backs it

Four properties, each of which a reviewer can check:

1. **Spend is held exactly constant across topologies.** Every rung buys
   exactly *k* interventions — verified from the data, not trusted from the
   code: `distinct = |A| + |B| - shared` solved against recorded overlap on all
   270 fan-in cells, residual 0.00, zero non-integer shared counts.
2. **The score is objective and external.** F1 against a physically-constructed
   ground-truth graph. No LLM judge, no rubric.
3. **Replicated across two chambers** with different graphs, menus and sampling
   regimes.
4. **Replicated across two models** separated by 3.9x in price, where the
   ordering survives and the expensive model is the *less* accurate one.

That combination is the contribution before any result is stated.

### The intellectual contribution is the reframing, not a winner

We should not publish "loops beat graphs." The grid does not say that, and it
is the less interesting claim.

The debate is framed as a question about **shape**. Our grid says shape
predicts little: `fan_in_homog` and `team` have very different shapes and
similar deficits; `relay` and `llm_pc` have different shapes, no measurable
gap, and `relay` is cheaper. What predicts performance is **how much of the
running record survives the partition** (§3).

> **Multi-agent structure is close to free when agents share the record of what
> has been done, and costly when they do not. The measured cost is not of
> having several agents; it is of partitioning their information.**

This is friendly to both camps and useful to neither's slogan: build the graph,
just do not let its nodes go blind. It is also falsifiable — it predicts a
shared-blackboard topology collapses onto the loop, and that a blind fan-out
loses in proportion to how much it partitions.

**Status: hypothesis, not finding.** The axis was recognised *after* seeing the
grid and both its endpoints are unrun. This is exactly why Phase 2 is
load-bearing rather than optional.

### The axis has a name already: bagging vs boosting

Recorded 2026-08-30 because it costs nothing and makes the claim legible to a
reader who has never thought about agent topology but has trained a hundred
gradient-boosted trees.

The reframing above maps onto ensemble learning almost exactly:

| our rung | ensemble shape | aggregation |
|---|---|---|
| `fan_in_homog` / `fan_in_spec` / `team` | bagging — independent learners | at the end |
| `llm_pc` (loop), `planner_reasoner` (relay) | boosting — sequential learners | continuous |

Our measured ordering is the one ensemble theory predicts, **for the reason it
gives**: parallel learners duplicate because nothing tells them what the others
already covered; sequential learners condition on what came before. That is the
partition-of-the-record claim in different vocabulary, and the vocabulary is
sixty years old and uncontroversial, which is worth borrowing.

**Two things this framing must not be allowed to overclaim**, or a reviewer
who knows ensembles hands it straight back:

1. **Our loop is not boosting.** Boosting conditions each learner on the
   *residual error*. Our loop conditions only on *what was already bought* —
   there is no feedback at all (the structural absence admitted above). So the
   analogy motivates Phase 3b; it does not describe what we ran. Used
   correctly it makes 3b stronger: adding the current adjacency estimate as
   feedback is precisely what makes the loop boosting-like, so 3b stops being
   "answering an objection" and becomes "testing a prediction the framing
   makes."
2. **We have never tested ensemble *aggregation*, only ensemble
   *acquisition*.** `fan_in_homog` runs independent selectors and combines
   their purchased **data**, then runs PC once. Bagging proper would run PC per
   scout and vote the edges. So "bagging lost in our grid" is NOT shown; what
   lost is parallel acquisition with pooled data. State the weaker claim.

The vote-aggregation version is an estimator change, not a topology change —
see §7 for why it stays out.

### The price, which the discourse has not measured at all

Everyone argues capability; nobody publishes the bill. We can: **+0 calls for
the relay, +1 for either fan-in rung, +5 for team — and F1 does not rise with
it. 12 of 12 fan-in points strictly dominated** across both chambers. For a
practitioner this is the most directly actionable result we hold, and it is a
free byproduct of having budget-matched arms.

### Contracts are the instrument, not adjacent decoration

The connection to make explicit in the paper: **you cannot budget-match
topologies without a mechanism that splits a budget across agents and verifies
the split held** — which is flow conservation over a delegation DAG
(`core/delegation_graph.py`). The benchmark the critics are asking for is
blocked on an enforcement problem, and we built the enforcement first, for
unrelated reasons.

Contracts made the measurement possible; the measurement is what the discourse
is missing. That is a better story than either half alone.

### The objection that can sink this, and the answer

A graph-camp reviewer will say: **your task has no context pressure and no
feedback, which is the entire reason graphs exist.** They are right on both
counts.

- **No context pressure.** The strongest argument for decomposition is that one
  agent's context saturates over a long horizon. Our loop's record is *k*
  lines. **We cannot observe the failure mode graphs are designed to solve.**
- **No feedback.** Every arm is blind to the data it bought. Much of the
  loop-engineering case is about error correction from observed results, and
  our design removes precisely that.

Do not soften either. **Scope the claim to them, which sharpens it:**

> In the regime where a single agent's context is *not* the bottleneck,
> partitioning strictly loses, and it loses in proportion to how much of the
> record it destroys.

That is bounded, defensible, and makes a prediction about where graphs should
start winning — when the running record exceeds one context. It invites the
follow-up study instead of pretending to have done it.

### What this positioning obliges the phases to do

Not decoration: it changes the plan.

- **Phase 2 is promoted to load-bearing.** Without `one_shot` and `critique`
  the axis is post-hoc and a reviewer will say so. Its pre-registered
  predictions are what convert the reframing into a test — including the one
  that kills it (`one_shot` ≈ `loop` means the record was never load-bearing).
- **Phase 3b is not optional.** The adaptive-feedback arm is the only thing
  that answers the sinking objection on its own terms. Reported as a bound,
  not a branch.
- **Phase 1 is what makes the reframing an explanation rather than a
  correlation.** We know `team` matches the loop's coverage and scores worse;
  until `chosen_experiments` is analysed we cannot say *which* picks differ.
- **Phase 4 must carry the scoping sentence verbatim**, and must state the
  untested regimes (long-horizon context pressure, >2 agents, heterogeneous
  tools) as scope limits rather than future work.

## 2. Why there is an M7

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

## 3. The reframing this plan is built on

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

## 4. Naming

`planner_reasoner` is misleading: nothing is planned. It is one loop with a
seam at the midpoint and a different system prompt on each half — which is
exactly why it resolves in neither direction in all six of its contrasts.

**Rename the paper-facing label to `relay`; keep the code identifier.** The
identifier is the `agent_name` value in 450+ recorded rows and renaming it
would orphan the data. One line in `VARIANT_LABELS`.

## 5. Phases

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

Since §1's bagging/boosting framing, this arm carries a **pre-registered
direction**: feeding back the current adjacency estimate is what turns the
loop from sequential-without-residual into something boosting-shaped, so it
should HELP. If it does not, the framing is weakened as well as the objection
answered, and both go in the write-up.

### Phase 4 — rewrite

Rename `relay`; fold the mechanism into the results doc; scope the headline to
the axis rather than to "topology"; state the untested shapes (critique with
iteration, adversarial disagreement, >2 agents) as scope limits.

Three things §1 obliges this rewrite to do, which the earlier draft did not:

1. **Carry the scoping sentence verbatim** — "in the regime where a single
   agent's context is not the bottleneck, partitioning strictly loses, and it
   loses in proportion to how much of the record it destroys." The bound is the
   claim; an unbounded version is wrong.
2. **State the two structural absences in the limitations, not in future
   work**: no context pressure (our loop's record is *k* lines, so we cannot
   observe the failure mode graphs exist to solve) and no feedback outside the
   Phase 3b arm.
3. **Make the instrument argument explicit** — budget-matching topologies is an
   enforcement problem, flow conservation solves it, and that is why this
   benchmark did not already exist.

## 6. Sequencing and the reason for it

**Phase 1 → look → then decide 2 and 3.** Each new arm is another chance to
find a harness defect: 17 recorded so far, and three found today were in code
written the same day. The mechanism result may also reframe what is worth
running — if the cause turns out to be forced allocation, `shared_blackboard`
becomes the most interesting arm on the list rather than an optional one.

## 7. Non-goals

- Not a causal-discovery methods paper. PC on pooled interventional data stays
  fixed and mis-specified-but-uniform; improving it would change every number
  and answer a different question.
- **Not bagged/stability-selected scoring**, which is the specific form the
  point above takes for the §1 ensemble framing. Bootstrap-resampling the
  purchased rows and majority-voting the edges is an ESTIMATOR change: it
  moves every number in all 2,221 recorded cells, and register entry 10
  forbids pooling rows scored differently, so adopting it forks the dataset
  rather than extending it (~$90 to re-run). **Parked as a journal-extension
  question with a $0 way in**: score the LLM-free `random` arm both ways and
  compare the sd across seeds. If PC's own instability is a large share of our
  spread, bagged scoring would shrink the MDE and more than 10 of 24 contrasts
  would resolve — worth knowing, not worth doing before the ladder reports.
- Not >2 agents. The axis is about partitioning information, not scale.
- **Not adjudicating loop engineering vs graph engineering.** We are not
  crowning a winner; we are supplying the missing measurement and replacing
  their axis (shape) with a better one (surviving record). A paper that reads
  as "loops win" invites a rebuttal from a regime we never tested.
- Not a cross-vendor model sweep. One additional model family answers the
  objection; a sweep is a different paper.
