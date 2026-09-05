# Notes: "Teamwork: When AI Becomes a Research Partner" (Google Antigravity, 2026-08-27)

**URL**: <https://antigravity.google/blog/teamwork-when-ai-becomes-a-research-partner>
**Authors**: "The Antigravity Team" (no named authors)
**Read**: 2026-09-05, from the **raw page** — the URL returns gzip-compressed
HTML that `file` reports as `gzip compressed data` despite the `.html`
extension; `gunzip -c` then tag-stripping yields ~16 KB of prose. Every quote
below was matched against that text.

**Note on the summariser**: two WebFetch passes were run first, and on this
article *both returned figures that the raw text confirms*. That is worth
recording alongside the Anthropic note's opposite result, because it shows the
failure is not deterministic — a summariser that happened to be right once is
not evidence it can be trusted. It missed a *qualifier* rather than inventing a
number: the seven problems are **Gemini 3.1 Pro** results of which three
*reproduce* on Flash, which the summary flattened into a joint claim. Read the
source.

## Why we care

This is the **second independent instance within a month** of the confound M7
§1 is built on — a multi-agent result reported without a matched-budget
single-agent control — and it is a *cleaner* instance than the Anthropic one.
It also contains, stated qualitatively and without measurement, the exact
moderator our headroom model quantifies. Positioning value is high; there is
nothing here that threatens a result of ours.

## Verified quotes and figures

### 1. The only comparative number, and it moves two variables at once

> "Using **Gemini 3.7 Flash** together with **3.1 Pro**, the Long Proof pattern
> achieves **71% on TCSBench** — up from the **67.7%** reported in the TCSBench
> paper with **Gemini 3.6 Flash** and 3.1 Pro, and the highest score in our
> internal testing."

The entire post rests one comparative claim on this sentence, and the two
conditions differ in **both** the model (3.6 Flash → 3.7 Flash) **and** the
orchestration (the TCSBench paper's harness → Long Proof). A +3.3 point delta
is attributed to topology while the model underneath it also changed.

No n, no variance, no seeds, no per-condition compute. TCSBench is
"independently developed **within Google**", so the baseline is not externally
checkable either.

**Use**: this is precisely the confound M6 spec §10.3 removed from our own
ladder when it dropped `llm_only` in favour of `llm_pc` — because comparing a
different inference procedure across rungs measures inference, not topology. We
made that correction internally, at the cost of a headline. Cite this as the
uncorrected version of the same error, from an independent lab.

**Do not overstate.** This is a product blog, not a paper; it does not claim a
controlled experiment. The point is that the field's most visible multi-agent
claims are structurally uncontrolled, not that these authors erred against
their own stated standard.

### 2. The thesis is topology-as-substitute-for-scale — asserted, never tested

> "Teamwork Long Proof is designed to get the most out of Flash models by
> **coordinating many agents rather than relying on a single, larger model**."

> "three of them (Problems 1, 3, 4) are reproduced with Gemini 3.7 Flash — the
> first time a Flash-tier model has produced such PhD-level mathematical
> research with the right orchestration framework."

This is the hypothesis our pillar exists to test, stated by a major lab as a
design principle. The missing arm is the obvious one: **3.7 Flash in a single
long loop at the same total compute**. Without it, "coordinating many agents
beats one larger model" is untested against "spending the same tokens in one
agent".

**Use**: our answer, on two chambers and across a 3.9x model price range, is
that no fan-in topology beats a single sequential loop where the comparison
resolves. Their sentence is the claim; ours is the measurement. We should quote
it in §1 as the position we test.

### 3. Compute is decided at runtime — the budget is not merely unmatched, it is unstateable

> "Critically, patterns are **adaptive at runtime**. The framework dynamically
> decides how many agents to spawn based on task requirements, not a preset
> number. Agent count and team structure can shift mid-run as the problem
> reveals itself — making each campaign a living process, not a fixed pipeline."

> "Some of these results used **higher parallelism than the default**. The
> version available on Antigravity balances cost and capability."

This is a **stronger** version of the Anthropic confound. There, the budgets
were unmatched but stated (27M vs 6.5M tokens), so a reader could do the
division. Here the agent count is endogenous to the topology and varies within
a run, so no per-condition budget can be quoted even in principle — and the
"higher parallelism than the default" sentence tells us the reported results
are not the shipped configuration, without saying by how much.

**Use**: this is the sharpest available argument for why a contracting
framework belongs in the multi-agent evaluation stack at all. If the topology
chooses its own budget, topology and budget cannot be separated by any
post-hoc analysis. Enforced per-node budgets are not bookkeeping; they are what
makes the comparison exist. **This is the strongest single sentence in the
article for our §1.**

### 4. They condition topology on decomposability — our headroom result, unmeasured

> "**Iterative Coding** for non-decomposable problems solved through tight
> agent–test–refine loops; **Distributed Coding** for decomposable engineering
> tasks that fan out across parallel workers with critic review"

> "Different challenges demand different team structures"

The taxonomy is explicit: **loop for non-decomposable, fan-out for
decomposable.** That is our finding in qualitative form. What is absent is any
statement of *how to tell which you have before running*, or how large the
effect is.

**Use**: this is the best positioning point in the article. We have
`predicted gain = coverage exchange rate × variables recovered`, exchange rates
regressed on LLM-free arms (LT 0.0061 ± 0.0005, WT 0.0111 ± 0.0006), an
a-priori headroom statistic computable from the menu before any run, and a
pre-registered point prediction that landed at **+0.0149 predicted / +0.0139
measured**. They select patterns by runtime heuristic and report that selection
matters; we predict the sign and the magnitude. Frame the contribution that
way: *the field has the taxonomy and lacks the predictor.*

### 5. The record-survival axis, built into their design

> "Refuted routes remain in the process with their objections attached — a
> broken route may still contain a useful idea."

> "**It learns across rounds.** Failed drafts remain available to the next
> attempt, while verifier findings are distilled into an answer-agnostic
> pitfall registry. A **shared knowledge directory** records proved results,
> useful observations, failed approaches, and relevant references for later
> use."

Their Long Proof pattern is a shared blackboard with a surviving record —
exactly the axis our M6 ladder was built on, and the one **M7 Phase 2 found is
not load-bearing** in our task (`one_shot`, a single call with no record at
all, ties the loop at LT k=30/45 and at all three WT budgets).

**State this carefully, with our own scope limit attached.** The regime
difference is item 6 and it is decisive: their record accumulates *verifier
findings*. Ours accumulates only which experiments were already bought. A
record whose entries are refutations is worth more than a record whose entries
are a shopping list, and our result is evidence about the second only.

### 6. Every headline task is verifier-rich; ours is not

> "Many candidate strategies are generated in parallel, **each paired with a
> falsifier whose sole job is to break it**."

> "the 40-page proof was formally **verified in Lean**"

> "Validated against BOOM hardware execution ground truth, the Teamwork
> Simulator achieved an average cycle alignment error of **0.71%** on unseen
> test workloads."

> "Teamwork's solution addresses the execution gap by maintaining continuous
> **lockstep co-simulation** against the air-gapped Spike reference simulator."

> "2× throughput on initial inserts with 64 threads" / "1.5× throughput overall
> with a single thread" / "25% less memory per element" (ParlayHash)

Lean checks a proof. Spike checks a cycle count. A benchmark checks a hash
table. **Every** result they report sits on a cheap, automatic, per-candidate
verifier — which is what makes generate-and-falsify pay, and it is stated as
the motivation:

> "For open problems, many promising approaches eventually fail — and the flaw
> stays invisible until deep into the attempt."

Our chamber has no verifier in the loop, and a selection's quality has no
hidden depth of that kind.

**Use**: this converts the dossier's top open threat ("your task is
coverage-shaped, so of course partitioning does not help") from a concession
into a **stated scope condition with a named foil**. Partitioning pays when the
action space has headroom — which we measure and predict — *or* when a cheap
verifier makes parallel generate-and-falsify affordable, which is their regime
and not ours. That is a defensible boundary, not a retreat.

### 7. Corroborates the orchestration-failure motivation

> "on hard research and engineering problems, multi-agent systems frequently
> encounter orchestration issues. Loosely organized agents quickly go off
> track, **agreeing with other agents' early mistakes and building confidently
> on flawed ideas**."

The same conformity failure the Anthropic note documents ("low variance"), from
a second source, and again offered as the motivation for a shared forum.
Supports our `fan_in_homog` reading.

## Other verified detail, recorded for completeness

- Published **Aug 27, 2026**; "11 min read"; shipped as `/teamwork-preview` in
  Antigravity on all paid plans.
- Seven problems: Coresets for Lp Subspace Approximation (FOCS 2025), Sparse
  Convex Optimization (JMLR 2021), Maximal Inner Product Embeddings, Provable
  Hadamard Quantization ("reducing the leading constant by ~5.93×"), Erdős Unit
  Distance ("Independently reproduced"), Prefix-Matrix Factorizations, Knuth's
  Cycles Conjecture. Five papers on arXiv.
- "reviewed and confirmed correct by human experts with the exception of the
  Knuth's Cycles result — where the 40-page proof was formally verified in
  Lean."
- Five shipped patterns: Iterative Coding, Distributed Coding, Long Proof,
  Self-Verification, Document Review. "Gemini analyzes your prompt and
  automatically selects the appropriate pattern."
- "orchestration logic is decoupled from agent descriptions"; "A pattern is a
  specification rather than an executable program."
- Self-Verification is "inspired by the Aletheia agent".
- Eigen: a GeMV fast path with "SIMD operations with 4-way accumulator
  unrolling", merged upstream.

## QUARANTINE: nothing

Unusually for this directory, **every figure the summariser produced is present
verbatim in the source text** and no number here is unverified. The section is
kept, empty, so its absence is not read as an omission.

The one thing the summariser got *wrong* was not a number but a **scope
qualifier**: it reported "71% ... using Gemini 3.7 Flash combined with 3.1 Pro
models" alongside the seven problems as if one condition produced both, where
the text says the problems are 3.1 Pro results and only three reproduce on
Flash. Cheap to miss, and it would have made our §1 claim about Flash-tier
capability stronger than the source supports.

## What to do with this

1. **Add item 3 to M7 §1 as the primary citation.** "Agent count and team
   structure can shift mid-run" is a better motivating quote than the Anthropic
   token counts, because it makes budget-matching *impossible* rather than
   merely *absent*. Highest value; free.
2. **Quote item 2 as the hypothesis under test.** Their design principle,
   stated in their words, is what our two-chamber result answers.
3. **Reframe the coverage-shaped-task threat using item 6.** Name the verifier
   as the moderator we do not have, cite their tasks as the regime where it
   holds, and state our scope explicitly rather than defensively. This should
   go in the results doc's threat list, not only in §1.
4. **Position the headroom model against item 4** — the field has the
   decomposable/non-decomposable taxonomy and no predictor for it.
5. **Attach the item 5 caveat wherever Phase 2's `one_shot` result appears.**
   Our record holds a shopping list; theirs holds refutations. Do not let the
   "the running record is not load-bearing" sentence travel without it.

**Do not** claim they should have run our experiment. It is a product
announcement about capability, and its results are real engineering artifacts
merged into upstream libraries. The argument is about what such announcements
can and cannot establish, and that argument is stronger made politely.
