# Yang et al., "CausaLab: A Scalable Environment for Interactive Causal Discovery Toward AI Scientists" (arXiv 2605.26029, May 2026)

Found 2026-09-13 via a citation in CausalDS (arXiv 2607.08093, Leban & Sun),
which calls it "concurrent work" and summarises it as: the agent intervenes
under a budget, predicts a held-out target, and is scored on both task
success and the fidelity of the recovered mechanism. Abstract verified
against the arXiv page; **body not read.**

## What they claim (abstract, verified)

- A scalable environment for interactive causal discovery by LLM agents that
  scores "both whether an agent can solve a problem using causal evidence
  and whether its answer is grounded in a faithful recovered causal
  mechanism".
- Each episode: prior measurement records, the agent intervenes on a
  "manipulator crystal", predicts the resonance frequency of a held-out
  "reactor crystal" governed by the same mechanism. Hidden DGP is a
  randomly sampled SCM, so success needs graph + structural equations, not
  recall.
- Headline gap: in the observational 6-node setting GPT-5.2-high reaches
  92 % task accuracy but only 0.471 all-edge F1.

## Why it matters to us

**Closest published task to ours**: an LLM agent spends an intervention
budget and is scored on the recovered graph. Differences that the paper
must state in one sentence each:

1. Synthetic SCMs sampled per episode vs two physical chambers with one
   released ground truth and a fixed menu (our menu is the thing that makes
   coverage countable and the rule computable).
2. They evaluate the *agent's* causal competence (its own answer and its
   recovered mechanism); we hold the estimator fixed and evaluate what a
   *topology* does with the budget. Ours is a coordination benchmark that
   happens to use causal discovery; theirs is a causal-discovery benchmark
   that happens to use an agent.
3. Their prediction-vs-mechanism gap (92 % vs 0.471 F1) is a cousin of our
   core-20 vs full-graph gap: a score can be high for reasons that are not
   structure recovery. Cite when arguing for reporting core-20 beside F1.

Also surfaced in the same search, cited in §2 as the "interventional
agents" reference: Roy & Parbhoo, "Why LLMs Fail at Causal Discovery and How
Interventional Agents Escape" (arXiv 2605.27567) — 17 LLMs near random on
discovery from correlational statements; abstract-level only.

## Quarantined (not verified)

- Budget sizes, number of nodes beyond the 6-node setting, and whether any
  multi-agent configuration is evaluated.
- Whether the environment releases a fixed intervention menu (if it does,
  the coverage rule could be run on it: a cheap external replication of
  "no agent beats round-robin coverage").
