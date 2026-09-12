# Kim et al., "Towards a Science of Scaling Agent Systems" (Google Research / MIT / DeepMind; arXiv 2512.08296, v1 9 Dec 2025, v3 8 Apr 2026; blog 28 Jan 2026)

Read 2026-09-12 from the Google Research blog post and the arXiv v3 HTML.
Every figure below was checked against the source text; figures the blog
and the paper disagree on are quarantined at the end.

## What they did (verified against the paper text)

- Five architectures: single-agent (SAS) and four multi-agent (independent,
  centralized hub-and-spoke, decentralized peer mesh, hybrid), on six
  agentic benchmarks (Finance Agent, BrowseComp-Plus, PlanCraft, Workbench,
  SWE-bench Verified, Terminal-Bench), three model families (OpenAI, Google,
  Anthropic); 260 configurations in v3; 50–100 instances per configuration
  (20 for the two Docker benchmarks).
- **"Fixed computational budgets (matched total tokens across MAS and
  SAS)"** — the comparison is budget-matched by design, with identical
  tools, prompts and truncation policies.
- Headline: +80.8 % over SAS on decomposable financial reasoning
  (centralized), −39.0 % to −70.0 % on sequential planning (PlanCraft) for
  every multi-agent variant. Mechanism stated as budget fragmentation:
  "intra-agent reasoning for constraint verification and state tracking
  consumes most available tokens before communication can occur; subsequent
  inter-agent messages then compress reasoning quality."
- Error amplification A = E_MAS / E_SAS: independent systems up to 17.2×,
  centralized 4.4× ("validation bottleneck").
- Three scaling patterns: tool-coordination trade-off (β = −0.096, p =
  0.002), capability saturation (coordination stops paying once the
  single-agent baseline is high), centralized verification contains error
  propagation. A domain-complexity score D ∈ [0, 1] per benchmark
  (Appendix C; PlanCraft 0.419, BrowseComp-Plus 0.839, Workbench 0.000).
- A mixed-effects regression on coordination metrics (efficiency, error
  amplification, message density, redundancy): cross-validated R² = 0.373
  (0.413 with a task-grounded capability metric); picks the best
  architecture for 87 % of held-out configurations.
- Limitations they state: agent count up to nine only; homogeneous base
  models; tool-heavy environments as the primary failure mode.

## What it means for our pillar

**Same sign, same axis, independently.** Their "sequential penalty" — every
multi-agent variant below the single agent on a task with sequential
dependencies, under matched tokens — is our M6/M7 headline on a
sequential experiment-selection task: of 24 topology-vs-loop contrasts, 10
resolve and 9 favour the loop; no fan-in topology beats the sequential loop
where the comparison resolves. Two groups, different tasks, different
models, same direction. Cite it as the closest contemporaneous result and
as the field-level context our single-task depth complements.

**Where ours is sharper, and the paper should say so:**

1. *Budget matching is enforced, not assumed.* They match total tokens by
   design parameter. Our matching is a contract with a certified
   conservation check, and H-C shows the design-time forecast FAILS in
   35 % of graph cells even when matched by design (register, WT `team`
   67 % certified). "Matched by construction" is a claim their method
   cannot verify after the fact; ours can, and the failures are reported.
2. *Their mechanism is not ours.* They attribute the sequential penalty to
   cognitive-budget fragmentation (communication eats the token budget).
   On our task the record is not load-bearing (`one_shot`, one call and no
   record, ties the loop at every budget on both chambers; register §35),
   the loop saturates at F1 ≈ 0.42 while random catches up, and two-thirds
   of `team`'s deficit is variable-level redundancy (M7 Phase 1). So a
   sequential task can punish multi-agent topologies WITHOUT budget
   fragmentation: the loss is duplicated work and a coverage plateau, not
   compressed reasoning. Their story and ours are compatible but distinct;
   a paper that only cites theirs would import a mechanism we measured not
   to be the one operating.
3. *Their per-configuration numbers carry unquantified run-to-run
   variance.* E.4 describes instance subsets and bootstrap CIs over
   instances; no repeated runs per configuration are described. Register
   §21/§26: the same seed, same config gives F1 0.33 and 0.48 on our task;
   temperature 0 is itself nondeterministic. Bootstrap over instances does
   not see this. Our MDE discipline (n = 30–50 per arm, 9-seed re-scoring,
   design clustering, two caps) is the methodological contribution to set
   against a 260-configuration single-pass sweep.
4. *Predict-before-run.* Their D score is assigned per benchmark
   (Appendix C, "Complexity Score Assignment") and their regression is
   fitted on run outcomes. Our `a_priori_headroom` is computed from the
   menu before any cell runs and ranks the chambers correctly (LT vs WT
   exchange rates 0.0061 vs 0.0111). Same move — a measurable task property
   that predicts whether coordination pays — but ours is derivable from the
   task specification alone, at a 1.5× under-prediction.
5. *A verifier and an oracle.* Their "centralized verification contains
   errors" needs a verifier; the Antigravity note already scopes our top
   threat as "partitioning pays with headroom OR with a cheap verifier", and
   our task has none. We do have a ground-truth oracle scale on purchases,
   which they do not; it is what let us show that no LLM arm's purchases
   beat a random draw on LT — and today that even a truthful effect
   feedback pushes them BELOW random (results doc, 2026-09-12).

**Where they are ahead:** six benchmarks, three vendors, nine models and a
cross-validated selector. We have two chambers, one vendor (cross-vendor
replication still unrun, ~$25) and one task family. Their breadth is the
reason to run the cross-vendor replication before submission; our depth is
the reason it is worth reporting at all.

**One sentence for the related-work paragraph:** Kim et al. (2026) find,
under matched tokens across six benchmarks, that multi-agent coordination
helps decomposable tasks and degrades sequential ones by 39–70 %; we
reproduce the sequential-task sign on a physical testbed with
contract-certified budgets, and show that the degradation there is
redundancy and a coverage plateau, not the budget fragmentation they
propose — and that it persists under three different feedback signals.

## Quarantined: blog vs paper disagreements

The blog post (28 Jan 2026) says **180 configurations**, **four
benchmarks** and **R² = 0.513**; the arXiv v3 (8 Apr 2026) says **260
configurations**, **six benchmarks** and **R² = 0.373 / 0.413**. The blog
reflects an earlier version; quote the paper. The summarising fetch also
reported the arXiv id and the R² correctly but omitted that the 87 %
figure is "in a restricted sense" (relative architecture selection, not
absolute performance prediction) — the paper's own qualifier.
