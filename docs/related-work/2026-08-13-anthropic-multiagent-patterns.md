# Notes: "Patterns and problems in emerging multiagent systems" (Anthropic, 2026-08-13)

**URL**: <https://www.anthropic.com/research/multiagent-systems>
**Authors**: Anthropic Frontier Red Team
**Read**: 2026-08-30, from the raw page (Next.js payload parsed out of the HTML,
Sanity portable-text spans concatenated in document order) — **not** from a
summariser. That distinction matters and is the reason for the quarantine
section at the bottom.

## Why we care

Its framing is systemic safety, not benchmarking, but four of its results bear
directly on the M7 loop-vs-graph positioning. One of them supplies the
motivating example our §1 currently asserts without a citation.

## Verified quotes and figures

Every line in this section was matched against the article text. Quotes are
verbatim.

### 1. The vulnerability swarm — our unmatched-budget confound, in the wild

> "we initiated 45 different agents and gave each one its own virtual machine, a
> shared forum on which they could coordinate, and an identical prompt that
> asked them to find vulnerabilities in a set of 15 open-source software
> projects."

> "For Mythos Preview, the simple independent parallelized method produces 21
> vulnerabilities over a 6.5 million token run, while the coordinating agent
> swarm found 266 vulnerabilities over a 27 million token run."

> "However, roughly half of these vulnerabilities were found outside of the core
> directories in which the simple independent parallel agents (stars in the
> above plot) were told to focus."

> "If we limit the swarm's outputs to only the vulnerabilities in the core
> directories, the two methods seem comparable in terms of tokens per
> vulnerability found."

> "The two methods are largely complementary: there were only 12 vulnerabilities
> in common between them."

**Use**: a 12.7x headline advantage (266 vs 21) that comes with **4.2x the token
spend** (27M vs 6.5M) and an **unmatched search scope**, and which the authors
themselves reduce to "comparable" once scope is held constant. This is exactly
the confound M7 §1 claims the literature does not control for, now citable from
a careful, recent, high-profile source rather than asserted.

**Do not overstate it.** They are explicit about the caveat themselves; the
point is not that they erred, it is that even a careful study cannot make the
comparison cleanly without budget matching — which is our contribution.

### 2. Low-variance conformity — the mechanism behind `fan_in_homog`

> "Individual agents are 'low variance': they often act the same in situations
> where different people might take a much more diverse range of actions. All
> that differentiates one agent from another is its context, its scaffolding,
> and the model that underlies it."

Instances, all verbatim:

- "18 out of 30 agents decided to create a git branch with the exact same branch
  name, 'mvp-game-loop.'"
- "multiple agents in multiple runs titled their first submission 'The
  Cartographer's Last Commission'. The agents were given zero guidance on the
  subject matter."
- "over half of the agents decided to build either ray tracers or self-hosting
  compilers."
- "In an iterated prisoner's dilemma game with communication, agents all settle
  upon the same strategy and they all defect at the same time."
- Job queues, finite bandwidth: agents "flooded the system with high-frequency
  (30 times per second) polling daemons… In one run there were 2.4 million job
  requests and only 117 jobs accepted."

**Use**: independent evidence for the criticism we already levelled at our own
`fan_in_homog` — two scouts differing only by sampling temperature are the same
opinion drawn twice. It also predicts our measured result that `team`'s
negotiation resolves almost nothing (`n_contested` = 1.2 of 30): near-identical
agents have little to negotiate. And it makes `fan_in_spec`'s prompted role
differentiation (overlap 0.79 -> 0.32) the right mitigation to have tested.

### 3. Siloing is the default, even when sharing is available

> "Only Sonnet 5 is able to maintain both a high merge fraction while directly
> collaborating and sharing code with other agents"

> "Sonnet 4.6 and Opus 4.6 do a terrible job of merging PRs compared to newer
> models that are able to merge most of the PRs that they open."

Design: 12-hour "build a fantasy game" runs, swarm sizes varied, three prompt
types (metrics "averaged over the three different prompt types for varying
simulation size").

**Use**: agents partition themselves by default even when a shared record is
available. Our arms have the partition imposed; theirs choose it. Same
direction.

### 4. Hidden-profile — our thesis in a different domain

> "we distribute facts across a group of agents, such that the evidence they
> share between them supports a wrong choice, but individual agents hold unique
> knowledge that should be decisive for the right one."

> "we find that performance scales with model intelligence but does not saturate
> even at the top of our range. This matches the human literature where
> discussion converges on what everyone already knows"

Design, from the figure caption: "Groups of four agents decide between two
options in scenarios like hiring, investment, or property buying… the percentage
of episodes where the hidden-best option received the majority of the group's
votes, with **n=400 episodes per model**. In the solo ceiling baseline, one
agent has all the facts and decides unilaterally."

**Use**: the same claim as ours — *the cost is partitioning the information* —
in an unrelated task, and it connects our framing to the Stasser–Titus
hidden-profile literature, which gives it fifty years of grounding. Note they
report a **solo ceiling baseline**, i.e. a single agent with the whole record,
which is structurally our loop.

### 5. Conflict, and capability != coordination

Turf war: three same-model agents each told to migrate one Python backend to a
different target language, four hours, "each agent was initially unaware of the
presence of the others". Resolution categories scored "Across **n=120 episodes
per model**" as force / passivity / truce / not settled.

> "we find that this ability is not strictly better in Mythos-class models,
> which often successfully lock out other agents before resolving conflicts
> productively. This orthogonality between prosociality and other capabilities
> further necessitates strong multiagent alignment. Models more capable in
> execution are not necessarily more coordinated"

**Use**: their qualitative version of our quantitative finding — each of our
scouts individually beats a random baseline inside its own pool (14.2 / 14.8
distinct variables against 12.8) while cross-scout coordination sits at chance
(5.6 duplicated against a null of 4.11 +- 1.51). Individual competence does not
produce coordination.

### 6. The line that cuts against us, and sharpens the scope

Their coordinated swarm specialised, stayed complementary (12 in common) and
"found new vulnerabilities" at a sustained rate over a long run; ours never
benefits from partitioning.

We should read this as a **regime difference, and cite it as support for our
scoping sentence rather than hide from it**: their search space (15 codebases,
12+ hours) vastly exceeds one agent's context, so partitioning is the only way
to cover it. Our LT menu is 59 entries — one loop covers everything, so
partitioning can only subtract. That is precisely the boundary M7 §1 draws:
*in the regime where a single agent's context is not the bottleneck,
partitioning strictly loses.* Their result is the other side of that line, from
an independent source.

## QUARANTINE: figures that are NOT in the article text

A WebFetch summariser produced these on first read. **None of them appears in
the prose.** They may be correct readings of chart axes — the article does plot
these quantities — but we could not confirm them, so **none may be cited**:

| claimed | status |
|---|---|
| hidden profile: solo ~100%, groups 17–36%, Mythos 5 85% | **unverified** — prose gives no percentages, only "scales with model intelligence but does not saturate" |
| routing accuracy: Mythos ~0.85, Sonnet ~0.62 | **unverified** — no such figures in the text |
| turf war: "Mythos 5: 98% settled via truce" | **unverified** — no "98%" in the text |
| "Opus 4.8 and Mythos typically within 2 hours" | **unverified** |

To cite any of these, read the figures in the published page directly.

**Lesson, and it is the same one the harness register keeps recording**: a
summarising fetch will supply plausible numbers for quantities that exist only
as chart axes. Verify against the source text before a number enters a
document, and keep the unverified ones visibly quarantined rather than dropped,
so nobody re-derives them later and assumes they were checked.

## What to do with this

1. **Rewrite M7 §1's confound paragraph around item 1.** Free; converts our
   motivation from assertion to citation. Highest value.
2. **Add items 2 and 4 to related work** — low-variance conformity explains our
   ensemble result, hidden-profile corroborates the thesis and supplies the
   psychology lineage.
3. **Promote `shared_blackboard`.** Their "shared forum" is that arm and they
   report it as the mitigation for conformity failures ("One possible solution
   to this class of failures is to use something like a central forum").
4. **Park a hidden-profile chamber analogue** — give each scout different
   partial *data* rather than a different slice of the menu. Different axis
   (information asymmetry, not task partition); journal extension, not M7.

**Do not** adopt their "swarm" vocabulary: their agent counts are 10–80 and
their question is systemic risk; ours is 2 agents and per-topology efficiency.
Borrowing the framing invites a comparison we would lose.
