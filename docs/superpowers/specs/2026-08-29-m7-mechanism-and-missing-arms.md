# M7: the mechanism, the missing arms, and the two reviewer objections

**Status**: Phase 1 complete 2026-08-30; `team_varsplit` running; Phase 2 next
**Opened**: 2026-08-29
**Last revised**: 2026-08-30 — Phase 1 results folded in, `shared_blackboard`
promoted into Phase 2, external citations added to §1
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

**We no longer have to assert this. Anthropic's Frontier Red Team, 2026-08-13**
(`docs/related-work/2026-08-13-anthropic-multiagent-patterns.md`, quotes
verified against the raw page):

> "the simple independent parallelized method produces 21 vulnerabilities over a
> 6.5 million token run, while the coordinating agent swarm found 266
> vulnerabilities over a 27 million token run."

A **12.7x** headline advantage for the multi-agent system — on **4.2x the token
spend**, and over a **search scope the two arms did not share** ("roughly half
of these vulnerabilities were found outside of the core directories in which the
simple independent parallel agents were told to focus"). The authors are candid
about where that leaves the comparison:

> "If we limit the swarm's outputs to only the vulnerabilities in the core
> directories, the two methods seem comparable in terms of tokens per
> vulnerability found."

**Cite this carefully and generously.** The point is *not* that they erred —
they state the caveat themselves, and their goal was systemic-risk
characterisation, not benchmarking. The point is that the most careful public
multi-agent-vs-single comparison available still cannot separate topology from
spend, because nothing in the setup holds spend constant. That is the gap, and
it is an enforcement problem before it is an experimental-design problem (see
"Contracts are the instrument" below).

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

### Two of our findings now have independent corroboration

Both from the same source, both verified verbatim.

**Low-variance conformity explains our ensemble rung.** "Individual agents are
'low variance': they often act the same in situations where different people
might take a much more diverse range of actions." Their instances are stark —
"18 out of 30 agents decided to create a git branch with the exact same branch
name, 'mvp-game-loop'"; over half of a swarm asked to build something impressive
chose ray tracers or self-hosting compilers.

This is the criticism we already levelled at our own `fan_in_homog` (two scouts
differing only by sampling temperature are the same opinion drawn twice), now
supported from outside. It also **predicts a measurement we already have**:
`team`'s negotiation resolves almost nothing (`n_contested` = 1.2 claims of 30)
because near-identical agents have little to negotiate. And it makes
`fan_in_spec`'s prompted role differentiation (overlap 0.79 -> 0.32) the right
mitigation to have tested.

**Hidden-profile is our thesis in another domain.** They distribute facts so
that shared evidence supports the wrong choice while individuals hold pivotal
private knowledge, and score whether the group recovers it (n=400 episodes per
model, against a "solo ceiling baseline" in which one agent holds all the
facts). Their own reading: "This matches the human literature where discussion
converges on what everyone already knows."

That is *the cost is partitioning the information* in an unrelated task, and it
connects our framing to the Stasser-Titus hidden-profile literature — fifty
years of grounding for an axis we otherwise introduce ourselves. Note their
solo-ceiling baseline is structurally our loop.

**Do not cite their hidden-profile or routing percentages.** Those quantities
appear only in figures; a summarising fetch fabricated plausible values for
them, and the notes file quarantines all four. Read the published figures
directly if a number is wanted.

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

**And the prediction already has a confirming instance, which we should cite
rather than avoid.** Anthropic's coordinating swarm searched 15 codebases over a
long horizon, specialised, and stayed genuinely complementary with the
independent arm ("only 12 vulnerabilities in common"). That is a search space
vastly exceeding one agent's context — the regime our bound excludes — and
there partitioning does add value. Our LT menu is 59 entries: one loop covers
all of it, so partitioning can only subtract.

Presented this way the external result is **support for the bound, not a
counterexample to the claim**, and it turns our scope limit from an admission
into a positive prediction that someone else's data already satisfies.

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
  correlation** — ~~pending~~ **done 2026-08-30**. It says which picks differ
  (4.5 fewer distinct variables), what that costs (+0.0073 F1 per variable, so
  about two-thirds of the gap), and that the negotiation protecting against it
  performs at chance. The reframing now has a mechanism under it, not a
  correlation.
- **Phase 4 must carry the scoping sentence verbatim**, and must state the
  untested regimes (long-horizon context pressure, >2 agents, heterogeneous
  tools) as scope limits rather than future work.
- **Phase 4 must cite the external work as motivation AND as boundary.**
  The vulnerability comparison is what makes the confound concrete; the same
  paper's swarm success in a context-saturated regime is what makes our bound a
  prediction rather than an excuse. Both, or neither — quoting only the half
  that flatters us is the version a reviewer will catch.

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
| `shared_blackboard` | complete, two voices alternating | **not built — Phase 2** |
| `planner_reasoner` (relay) | complete, one seam | M6 ✓ |
| `team` | split by agreement | M6 ✓ |
| `team_varsplit` | split by agreement, on the RIGHT object | built 2026-08-30, running |
| `fan_in_spec` / `fan_in_homog` | split blind | M6 ✓ |

`team_varsplit` is the arm this table did not anticipate. It sits at the same
point on the record axis as `team` — the running record is split the same way —
but partitions the WORK by variable rather than by menu entry. It therefore
separates two things the axis alone conflates: how much of the record survives,
and whether the partition is drawn where the information actually lives.

Stated as a claim to be tested: *multi-agent structure is free when agents
share the record of what has been done, and costly when they do not. The
measured cost is not of having several agents; it is of partitioning their
information.*

`shared_blackboard` is the sharpest test and the one most likely to collapse
into the loop by construction — two agents alternating with full shared history
IS the loop with two voices. **Promoted 2026-08-30 from conditional to included
in Phase 2**; see there for the three reasons that arrived together.

## 4. Naming

`planner_reasoner` is misleading: nothing is planned. It is one loop with a
seam at the midpoint and a different system prompt on each half — which is
exactly why it resolves in neither direction in all six of its contrasts.

**Rename the paper-facing label to `relay`; keep the code identifier.** The
identifier is the `agent_name` value in 450+ recorded rows and renaming it
would orphan the data. One line in `VARIANT_LABELS`.

## 5. Phases

### Phase 1 — mechanism: **COMPLETE 2026-08-30, $2**

**Question**: why does `team` lose at equal coverage? **Answered**, and the
answer moved twice before it settled. Full detail in `docs/chamber-results.md`
§"M7 PHASE 1"; register entry 20 for the confound found on the way.

**Result.** `team` buys 30 experiments but only **23.4 distinct variables**
against the loop's **27.9**, because the LT menu carries up to three entries per
variable and the two scout pools are disjoint as sets of EXPERIMENTS while a
variable can sit in both. 5.6 variables are bought twice while `overlap_frac`
reads exactly 0.0 in every cell — the safeguard was aimed at the wrong
granularity.

- **H1 (scouts buy depth) rejected.** Each scout is individually MORE
  breadth-seeking than the loop: 0.053 and 0.013 repeats per pick against 0.070.
- **H2 (lopsided allocation) rejected.** The split is even, 14.2 vs 14.8.
- **H3 (cross-scout duplication) confirmed**: 14.2 + 14.8 − 5.6 = 23.4 exactly.

**Does it cost accuracy?** Yes, ~two-thirds of it. A direct LLM-free
manipulation (`coverage_max_ms` / `coverage_min_ms`, 30 vs 15 variables at the
same budget with weak levels excluded) gives **+0.0073 F1 per distinct
variable**, so team's 4.5-variable deficit predicts **−0.033** of the measured
**−0.048**. The residual is below the contrast's own MDE.

**The negotiation is at chance on the axis that costs it.** Null model — pools
split at random, picks at random within pool, 8,000 draws: each scout beats
chance INSIDE its own pool (14.2/14.8 against 12.8) while cross-scout
duplication does not (5.6 against **4.11 ± 1.51**, z = +0.99). Every stage that
builds the pools is blind to variables: conflict detection is a set intersection
on NAMES, the leftover split a parity slice of a shuffled NAME list. **The
scouts coordinate competently over the wrong object.**

**Built in response, and it worked**: `team_varsplit` — identical topology,
budgets, four negotiation calls and A-wins-ties rule, partitioned by VARIABLE so
cross-scout duplication is structurally impossible. **90/90 cells, n=30 per arm,
2026-08-30:**

| arm | distinct vars | shared | F1 |
|---|---|---|---|
| `llm_pc` | 27.5 | — | 0.411 |
| `team` | 22.7 | 6.50 | 0.388 |
| `team_varsplit` | **28.2** | **0.00** | **0.424** |

Pre-registered **+0.0399**, observed **+0.0360** (MDE 0.0344, **RESOLVED**);
against the loop, +0.0127, below MDE. **Changing only what is partitioned brings
a two-agent arm level with the single loop.** This is the pillar's strongest
result: a manipulation confirming a mechanism, with the prediction fixed
beforehand from a slope measured on unrelated LLM-free arms.

The mock-LLM control is what makes it a real test rather than a rigged one:
under random selection the gain cancels exactly (shared 3.83 → 0.00 but
per-scout distinct 12.4 → 10.3), so the arm pays off only if scouts avoid
self-repetition — which the real ones do.

**Caveat that travels with it**: `team` − `llm_pc` came in at −0.023 here
against −0.046 (M6) and −0.048 (Phase 1); pooled n=40 gives −0.0296 at MDE
0.0298. The deficit `team_varsplit` closes is itself only marginally resolved.
Cause and fix in register entry 21 — temperature is unpinned, and it moves arm
means, not only cells. `--temperature` and `RunRecord.temperature` shipped
2026-08-30, default unset.

**Consequence for the paper's claim.** The ladder's cost is not the cost of
several agents, nor even of partitioning their information. It is the cost of
partitioning it **on the wrong object**. Drawn where the information lives, a
two-agent split is free. That is a friendlier and more useful claim than
"loops win", and it is what §1 should say.

**Two withdrawals, recorded rather than quietly re-founded**: Phase 1's first
reading ("coverage does not explain the loss") was a range-and-power artifact;
and the 25 Aug note "team's cost is genuine coordination, not redundancy" is
withdrawn, not merely re-founded.

### Phase 2 — close the axis (~$8), now with the blackboard

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

**`shared_blackboard` is promoted from conditional to included** (was: "build it
only if Phase 2 leaves the axis ambiguous"). Three independent reasons arrived
together on 2026-08-30:

1. It is the **upper endpoint of the axis** and the axis is the contribution.
   Leaving it unrun keeps the reframing post-hoc, which is the objection Phase 2
   exists to answer.
2. Anthropic's post names a **central forum** as the mitigation for exactly the
   conformity failures we now know drive our fan-in deficits — "One possible
   solution to this class of failures is to use something like a central forum
   in which agents can agree on best practices and protocols" — and their
   coordinating swarm, which had one, is the arm that worked. That is an
   external pre-registered prediction for a rung we already planned to build.
3. Phase 1 says team's loss is **duplicated work its agents could not see**. A
   shared record is the direct fix; `team_varsplit` is the structural fix. Both
   arms together separate "prevent the collision" from "see the collision",
   which is a sharper result than either alone.

**Prediction**: `shared_blackboard` ≈ `loop`, since two agents alternating with
a complete shared history IS the loop with two voices. If it does NOT collapse
onto the loop, the axis is wrong and the cost is in having several agents rather
than in partitioning them — the most informative failure available to us.

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

## 6. Sequencing — updated 2026-08-30, after looking

The original rule was **Phase 1 → look → then decide 2 and 3**, on the reasoning
that each new arm is another chance to find a harness defect and that the
mechanism result might reframe what is worth running. Both happened.

**The defect did appear**, in an arm built the same day: the first coverage
manipulation was confounded with intervention strength (register entry 20). It
was caught by tabulating every recorded attribute of the picks across arms
rather than only the manipulated one — one `groupby`. Deconfounding *doubled*
the effect rather than shrinking it, so a confounded result had been reported
as decisive in the wrong direction. The rule earned its keep; keep it.

**And the result did reframe the plan**, though not the way the spec guessed.
It anticipated that a forced-allocation answer would promote `shared_blackboard`.
The answer was cross-scout duplication instead — and that promotes the
blackboard anyway, for a different and better reason: the loss is work the
agents could not see, and a shared record is the direct remedy. Paired with
`team_varsplit` (the structural remedy) the two arms separate *preventing* the
collision from *seeing* it.

**Revised order:**

1. ~~Phase 1 (mechanism)~~ — **complete**, $2.
2. `team_varsplit` at n=30 — **running**, ~$4. Tests the redundancy account with
   a one-change control.
3. **Phase 2** — `one_shot`, `critique`, and now `shared_blackboard`. The axis
   is the contribution and this is what closes it. Highest remaining value.
4. **Phase 3** (the two objections) — unchanged, and 3b's direction is now
   pre-registered by the bagging/boosting framing (§1).
5. **Phase 4** (rewrite) — gains the external citations from §1 and the
   mechanism result from Phase 1.

Still true, and worth repeating because it keeps being the thing that saves
results: look between phases rather than queueing them.

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
- **Not swarm scale, and not swarm vocabulary.** Anthropic's post runs 10-80
  agents and asks about systemic risk; we run 2 and ask about per-topology
  efficiency at matched budget. Borrowing "swarm" invites a comparison we would
  lose, and their failure modes (collusion, turf wars, cascading conformity) are
  not observable at n=2.
- **Not the hidden-profile analogue, this paper.** Giving each scout different
  partial DATA rather than a different slice of the menu tests information
  ASYMMETRY, not task partition — a different axis with its own literature
  (Stasser-Titus). Parked as a journal extension; recorded here so it is not
  rediscovered as novel.
