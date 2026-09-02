# M7: the mechanism, the missing arms, and the two reviewer objections

**Status**: Phases 1-2 complete. **Plan revised 2026-08-31 — see §8**, which
reprioritises Phase 3 around what stands between this and an accept.
**Opened**: 2026-08-29
**Last revised**: 2026-08-31 — Phase 2 folded in (the record axis is
unsupported at 5 of 6 budgets), and §8 added: revised framing, the four ranked
threats, and a reprioritised Phase 3 led by cross-vendor replication
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

**`shared_blackboard` claims more than it delivers, and the paper must say so.**
In the classical sense (Hearsay-II and successors) a blackboard is a workspace
where agents post partial hypotheses and read each other's *reasoning*. Ours
shares only the running list of experiments already bought: the voices write a
pick and nothing else — no rationale, no advice, no "I am covering the pressure
sensors, you take the optics".

That narrowness is deliberate, not an oversight. The axis is *how much of the
loop's running record survives*, and the loop's record is exactly a list of
picks. Sharing more would make the arm differ from the loop in TWO ways at once
— shared record AND a new communication channel — and no difference could then
be attributed to either.

Either rename it (`shared_record` is honest and unglamorous) or state the
restriction on first use. Do not let a reviewer assume the richer thing and
find the thinner one.

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

### Phase 2 — **COMPLETE 2026-08-31, $11.57, 960 cells, 0 errors**

Full panels, both chambers, in `docs/chamber-results.md` §"M7 PHASE 2
COMPLETE". Scorecard against what was pre-registered below:

| prediction | outcome |
|---|---|
| `one_shot` < loop | **FALSE at 5 of 6 budgets.** Holds only at LT k=6 (−0.059). Ties at LT k=30/45 and all three WT budgets. |
| `critique` ≈ loop | **TRUE** (corrected 2026-09-01). Cell-level scoring read it as resolved worse at LT k=30/45; averaged over 9 PC subsample seeds those deficits are −0.013 and −0.015, inside a tighter MDE. |Δ| < 0.022 on both chambers at every budget. |
| `shared_blackboard` ≈ loop | **TRUE except at LT k=6** (−0.079 there; ties at the other five). |

The spec anticipated the `one_shot` outcome and named its consequence:
"the honest claim becomes 'how selection is organised barely matters'". That
is now the measured result, and it is stated in the results doc rather than
softened. Two things keep it from being a pure null:

1. **The axis test resolves at the middle budget on both chambers.**
   `shared_blackboard` vs `fan_in_spec` — same two role prompts, record shared
   vs split — gives **+0.053 (LT k=30)** and **+0.046 (WT k=14)**, both above
   MDE, with nothing resolving at the small or large budget on either. Sharing
   a record beats *splitting* one; sharing one with yourself (the loop) is
   worth nothing. Caveats in the results doc: cross-run, and WT sits on the
   MDE boundary after drift adjustment.
2. **The record claim now has a bound worth quoting.** Design-level re-scoring
   (no LLM cost) puts LT k=30's equivalence at **±0.021** against a
   loop-vs-random gap of +0.055, so it excludes "the record is worth nearly as
   much as selecting at all". `critique`'s status changed in the same pass: it
   ties the loop rather than losing to it, which is a weaker but defensible
   negative — a reviewer pass costs three flat calls and moves accuracy by
   less than 0.02 either way.

**Consequence for §1 and §3.** The ladder's rungs are ordered by
record-survival, and Phase 2 shows that ordering is not what produces the M6
effect. The M6 result stands (it replicates on WT under a 3.9x pricier model);
its *explanation* does not. Phase 4's rewrite must not argue from the record
axis except at the middle budget, where it is the best-evidenced claim we hold.

<details>
<summary>Original Phase 2 specification (pre-registration, kept verbatim)</summary>

#### Phase 2 — close the axis (~$8), now with the blackboard

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

</details>

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

## 8. Revised plan (2026-08-31): what turns this into a strong submission

> **REVISED AGAIN 2026-09-02.** Two results landed after this section was
> written and both hit its framing: the **coverage oracle** (2026-09-01) and
> the **WT `team_varsplit` non-replication** (2026-09-02). §8.1's third load
> and §8.4's change 1 both lean on `team_varsplit` as *the* positive result;
> that is now one chamber at one budget. And §8 nowhere mentions the oracle,
> which is the single most consequential thing in the corpus. §8.6 below
> supersedes §8.1 and §8.4's item 1; the rest of §8 stands.

Phases 1 and 2 are done and the corpus is large. The remaining question is no
longer "what else can we measure" but **"what is between this and an accept"**.
The honest assessment: a solid submission exists today; a strong one needs the
generality gap closed, and needs the framing rebuilt because Phase 2 falsified
the one the ladder was designed around.

### 8.1 The framing that survived

Not "a coordination ladder ordered by surviving record" — `one_shot` ties the
loop at 5 of 6 budgets, so that ordering is not what produces the M6 effect.
The framing that the data does support:

> Multi-agent evaluations almost always give more agents more budget. We hold
> the budget fixed — enforced by delegation contracts — and compare topologies
> on two real physical causal-discovery testbeds. Under matched budgets no
> fan-in topology beats a single sequential loop; the deficit is mostly
> duplicated coverage rather than coordination overhead; and changing *what*
> is partitioned (the variable space, not the task) recovers it.

Three loads this carries that the old framing did not:

1. **The matched-budget control is the field's missing measurement.** The
   motivating citation concedes the point in its own numbers (§1): a 12.7%
   multi-agent win on 4.2x the tokens, which the authors reduce to "comparable"
   once scope is matched.
2. **Contracts are the instrument, not decoration.** You cannot credibly assert
   "matched budget" across a fan-in topology without conservation-enforcing
   delegation. That is why this benchmark did not already exist, and it is what
   ties `core/delegation_graph.py`'s telescoping bound to the empirical work.
3. **There is a positive, quantitative, actionable result**: +0.0073 F1 per
   distinct variable, measured LLM-free by direct manipulation, predicting
   two-thirds of `team`'s deficit — and `team_varsplit` then recovers +0.036 as
   a pre-registered prediction. "More agents don't help; more coverage does, at
   this exchange rate, and here is the one-line change" beats any negative.

### 8.2 The four threats, ranked by what they cost to close

| # | threat | cost to close | closes it? |
|---|---|---|---|
| 1 | **One task family.** Two chambers, but both are "buy experiments, run PC". The loop-vs-graph discourse is about software and research agents. | very high (new domain) | **No — scope instead** |
| 2 | **One model family.** flash + pro is DeepSeek twice. Second question every reviewer asks. | **~$20–30** | **Yes** |
| 3 | **Equivalence claims at modest power.** The MDE is ~80% PC noise (§"WHY THE MIDDLE BUDGET"), and `one_shot`'s LT k=30 bound is ±0.051 after the pseudo-replication correction (register §24). | ~$5 + compute | **Mostly** |
| 4 | **Post-hoc assembly.** Two of three Phase 2 pre-registrations failed; the surviving story was assembled after. | $0 | **Yes, by disclosure** |

Threat 1 is not closable at this budget and must be **scoped in the title and
abstract**, not deferred to §7. Threat 4 is closable for free and the defence is
unusually strong: predictions are timestamped in git, both failures are
reported, and three earlier conclusions were publicly retracted (the walks
external-validity claim, "team's cost is not redundancy", and the flat-variance
reading). Lead with that rather than burying it.

### 8.3 Phase 3, reprioritised

**3a. Cross-vendor replication — PROMOTED to the highest-value remaining
work.** This was previously filed under non-goals as "not a cross-vendor
sweep", which conflated two different things: a *sweep* across many vendors
(still a non-goal, a different paper) and a *replication* of the key contrasts
on one second family (the thing that answers threat 2).

Scope it to the contrasts the paper actually leads with, not the full grid:

| contrast | budgets | why it must replicate |
|---|---|---|
| loop vs `team` | mid only | the headline topology negative |
| loop vs `team_varsplit` | mid only | the positive result and the fix |
| loop vs `one_shot` | mid + high | the record claim |
| loop vs random | mid only | proves the instrument discriminates |

One chamber (LT), n=30, mid budget = k=30, plus k=45 for `one_shot`. ~$20–30.
**Probe first** — availability, price, provider endpoints and quantization per
candidate — as `PROVIDER_ORDER_BY_MODEL` raises on unpinned models by design.
Done for `glm-5.3-flash` on 2026-08-31 (register §25): endpoints are usable
once `reasoning.effort` is pinned, which the agents already do; `Relace` is fp4
and excluded.

**Do NOT reuse the DeepSeek MDEs — added 2026-09-01, register §26.** Per-cell
variance is driven by a chaotic fork early in the reasoning trace, so an arm
that reasons longer has larger spread from that mechanism alone. If GLM's
traces differ in length, its MDEs differ. Compute each arm's sd from the
replication's own cells; otherwise a power difference will read as a failure to
replicate, which is the one outcome this phase must not get wrong.

A replication that holds is worth more to this paper than any additional
topology. If it does NOT hold, that is also publishable and reframes the
contribution as model-dependence of coordination benefit — but we need to know
before the rewrite, not after.

**3b. Selection diversity for single-call arms — NEW, small, blocking for the
record claim.** `one_shot` re-picks the same design (6 distinct at LT k=30), so
its equivalence bound cannot be tightened by seeds. **Shuffle the menu order
per seed** inside the single call, re-run `one_shot` on both chambers (~180
cells, ~$4), and report distinct-selection counts for every arm as a standard
column. Without this, the LT k=30 half of the record claim stays at ±0.051.

**Menu-order shuffling is the only lever — corrected 2026-09-01.** An earlier
version of this item offered "a pinned non-zero temperature" as the
alternative. Measurement (register §21) removes it: **temperature 0.0 is
already nondeterministic on this endpoint** — six distinct picks in nine draws
— and unset, 1.0 and 0.0 show indistinguishable diversity. Temperature is not
the knob. The reconciliation with §24 is that a single *pick* is variable while
a 30-pick *set* is nearly canonical, so the diversity has to be injected into
the prompt the set is chosen from, not into the sampler.

**3c. Bound tightening by multi-seed scoring — NEW, no LLM cost.** Averaging a
cell over m PC subsample seeds shrinks its noise by sqrt(m); the probe shows
noise is most of our per-cell spread. **Only valid after clustering by
selection** (register §24 records the near-miss where it was not). Applies to
M7 files only — the M6 ladders predate `chosen_experiments`.

**3d. Adaptive feedback — DEMOTED to optional.** Still the right design (loop +
current adjacency estimate, LT, one budget, as a bound rather than a branch),
and its direction is pre-registered by the bagging/boosting framing. But it
answers a reviewer objection about scope, while 3a answers one about validity.
Run it only if 3a lands early.

**3e. Mixed-model team — DEMOTED to optional.** A genuinely different pair of
architectures inside one team is interesting, but it adds an arm to a benchmark
whose problem is generality, not coverage. `fan_in_spec` already shows prompted
divergence is real (overlap 0.79 -> 0.32) and still loses.

### 8.4 Phase 4 rewrite — three changes Phase 2 forces

1. **The old scoping sentence is falsified and must not be reused.** It read:
   "partitioning strictly loses, and it loses in proportion to how much of the
   record it destroys." `one_shot` destroys the entire record and loses nothing
   at 5 of 6 budgets. Replacement, which the data does carry:

   > Under matched budgets, partitioning the *task* between agents loses;
   > partitioning the *variable space* does not. The loss tracks duplicated
   > coverage, not how much of the record survives.

2. **Add a discrimination table, early, before the negatives.** The compressed
   reviewer objection is "nothing you tried mattered — your task doesn't
   discriminate." The answer is a table of what *does* resolve: loop vs random
   (+0.047 to +0.055, 4 of 7 LT budgets), `team_varsplit` vs `team` (+0.036),
   shared vs split record (+0.053 LT / +0.046 WT), contracted vs uncontracted
   (+0.058 WT), coverage (+0.0073/variable). The instrument discriminates; the
   topologies genuinely do not differ.

3. **State the reproducibility claim at the level it holds.** Neither the
   seed, nor temperature, nor a pinned provider makes a *cell* reproducible
   (register §10 for BLAS, §21 for sampling). Reproducibility in this pillar
   lives at the level of **arm means over n seeds** — claim that, archive the
   resolved environment including the linear-algebra backend, and run every arm
   of a comparison on one machine. Stated positively this is a methods
   contribution; left implicit it reads as a hole.

   **Add: a pinned model id does not pin the computation** (register §32,
   added 2026-09-02). Twice now, DeepSeek has changed how much it reasons per
   call under an unchanged model string — 4.35x tokens on 2026-08-13, 2.4x on
   2026-09-02 — with `n_llm_calls` fixed by the arm and our code byte-identical.
   Three obligations follow, and all three are cheap: **record `n_llm_calls`
   and `tokens_out` per cell** so the question is answerable after the fact;
   **never schedule arms in blocks of time**, or provider drift lands on one
   arm (this is what forced the interleaving fix mid-run); and **run every arm
   of a contrast concurrently** wherever the schedule allows. Stated
   positively, this is the second half of the methods contribution: the paper
   can show that a 1.7x swing in reasoning moved F1 by 0.004, which is a
   measured robustness bound rather than a hope.

4. **Report power, not just significance.** State the noise-only MDE floor
   beside every equivalence (0.031 at LT k=30, 0.029 at WT k=21) and the seeds
   a 0.02 effect would need (n≈75 LT, n≈110 WT). An equivalence without a bound
   reads as a null; with one it is a result.

### 8.5 Sequencing

1. **Probe the second model family** (availability, price, endpoints,
   quantization). Half a day, no sweep.
2. **3a cross-vendor replication**, ~$20–30. Highest value remaining.
3. **3b selection diversity** re-run of `one_shot`, ~$4. Can run alongside 3a.
4. **3c bound tightening**, no LLM cost, after 3b so it clusters correctly.
5. **Phase 4 rewrite** with 8.4's three changes.
6. Optional, only if time: 3d feedback, 3e mixed-model team.

**Not on this list, deliberately**: more topologies, more budgets, >2 agents,
bagged scoring, the hidden-profile analogue, the rationale-passing blackboard.
All are recorded in §7 with reasons; none of them moves an accept/reject
threat.

### 8.6 The framing after the oracle (2026-09-02) — supersedes 8.1

**Lead with the reference policy, not with topology.** `coverage_max` is a
ten-line LLM-free rule (round-robin over distinct variables). Re-scored on one
BLAS backend at 9 PC seeds — the earlier table was cross-backend, register §31
— it **ties every LLM arm at four of six budgets, loses at LT k=6, and BEATS
every LLM arm at WT k=21**. Under core-20 scoring it ties at every LT budget,
including k=6.

> On a task with a computable near-optimum, we measure agent topologies as
> distance-from-optimum under contract-enforced matched budgets. LLM selection
> beats a ten-line coverage rule only where the budget is too tight for
> coverage to bind — and on the non-trivial subgraph, not even there. Above
> that, no topology we built beats the rule and several lose to it. No fan-in
> ever beats a single sequential loop.

Why this is the right lead:

1. **A computable near-optimal reference policy is rare in agent benchmarks.**
   It converts every arm from "better or worse than another arm" into
   "distance from a known ceiling", which is what makes the negatives
   interpretable rather than merely disappointing.
2. **It keeps a scoped positive.** The LLM's contribution is real and located:
   tight budget, where a coverage heuristic does not help. That is a finding
   with an actionable shape.
3. **It absorbs the `one_shot` result instead of being embarrassed by it.**
   Every arm converging above k/M ≈ 0.5 is them converging on the coverage
   optimum; that is the explanation Phase 2 was missing.

**The positive result is now the two-factor model, not `team_varsplit` itself.**
`team_varsplit` gains +0.043 on LT k=30 (resolved) and nothing detectable on
WT — but the non-replication is **predicted** by LLM-free measurements:

> predicted gain = coverage exchange rate × variables recovered by partitioning

3/3 on the verdict, and WT k=21 nearly exact (+0.015 predicted, +0.017
measured). The exchange rate is **higher** on WT (0.0111 vs 0.0061), so the
moderator is not "WT is a worse chamber" — it is **headroom in the action
space** (menu entries per variable: LT 1.97, WT 1.33). This is the version
that transfers to the loop-vs-graph discourse: partitioning by role pays in
proportion to the duplication the action space affords, and both terms are
measurable before running a model. `analyze_headroom.py`.

**Replacement for §8.4 item 1's scoping sentence:**

> Under matched budgets, partitioning the task between agents loses;
> partitioning the variable space recovers it, in proportion to how much
> duplication the action space affords — which is why the effect resolves on
> the redundant menu and is predicted to be undetectable on the sparse one.

**The threat this framing creates, and the only honest response.** "Your task
is coverage-shaped, so a coverage rule winning is a benchmark artifact." It is
correct: §29's ground truth is bipartite, depth 1, zero mediators, and §28
notes 18 of 38 nodes are pure apparatus sources. **Scope it in the title and
abstract.** The closable version is that the chambers' depth is TEMPORAL and
our pooled-i.i.d. reduction discards it; the authors' own WT case study meets
the same autocorrelation with PCMCI+ rather than a different dataset.

### 8.7 Sequencing, revised 2026-09-02

Reprioritised because the oracle changed what the marginal dollar buys.

| # | work | cost | why now |
|---|---|---|---|
| 1 | **Re-score the whole corpus on one backend**, quote `f1_rescored` throughout | $0 | register §31 moved two verdicts; until this is done "no LLM arm beats the rule" is not established |
| 2 | **Core-20 as a standard reported column** | $0 | pre-empts the coverage-shape objection instead of conceding it |
| 3 | **WT k=21 varsplit at n≈132** | ~250 WT cells | the model's own pre-registered confirmatory test; converts a non-replication into a law or falsifies it |
| 4 | **3b `one_shot` menu-order shuffle** | ~$4 | unchanged; still blocking the record claim's bound |
| 5 | **3a cross-vendor replication** | ~$20–30 | **value DOWNGRADED**: if an LLM-free rule ties every LLM arm, a second vendor mostly confirms that a second vendor also ties the rule. Still answers a reviewer reflex, no longer the central threat |
| 6 | **Lagged-estimator variant** (coverage does not bind) | engineering, $0 API | the only work that answers the top-ranked threat; the difference between a solid paper and one that is hard to reject |

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
- **Not the rationale-passing blackboard, this paper — but it is the most
  natural next arm.** Recorded 2026-08-30. `shared_blackboard` shares the bare
  record; let each voice post a one-line rationale alongside its pick and the
  next voice reads both. Concretely: extend the already-chosen block from
  `name` to `name — why`, and ask for `NAME | one-line reason` in the reply.

  Why it is worth running later, in order of value:

  1. **It is a second point on the SAME axis, not a different experiment.**
     The axis measures how much of the record survives; this raises the
     richness of the record rather than changing the topology. Bare record vs
     record-plus-reasoning is exactly the gradation the ladder currently skips.
  2. **It is the honest version of the name** (see §4), so running it would let
     the paper use the classical term without the caveat.
  3. **It is the first arm where agents could actually divide labour in
     words** — "I am covering the pressure sensors" — which is what a reader
     imagines multi-agent coordination to be, and which no rung on the current
     ladder does.

  Two design cautions to carry, both learned the hard way this milestone:

  - **It must be compared against `shared_blackboard`, not against the loop.**
    Against the loop it varies two things (two voices AND rationale) and
    resolves nothing, which is the confound `team_varsplit` was built to avoid.
  - **The rationale must not become extra budget.** It rides inside the
    existing selection call, so call count is unchanged — but output tokens
    rise, and the cost-frontier claim is denominated in CALLS precisely so
    that this kind of change stays comparable. State the token delta rather
    than hiding it.

  A plausible failure worth pre-registering: the rationales are near-identical
  across voices (low-variance conformity, §1), the channel carries no
  information, and the arm lands on `shared_blackboard` exactly. That would be
  a clean negative result about LLM-to-LLM coordination channels, not a null.
