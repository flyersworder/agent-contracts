# Tran & Kiela, "Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets" (arXiv 2604.02460, v1 2 Apr 2026, v2 11 Apr 2026)

Found 2026-09-13 by a web search for matched-budget single-vs-multi-agent
comparisons while fixing the AAMAS draft's citations. Read from the arXiv
abstract page only (title, authors, dates, abstract verified against the page
and the arXiv BibTeX endpoint). **The body of the paper has not been read.**

## What they claim (abstract, verified)

- Reported multi-agent gains "are often confounded by increased test-time
  computation"; when computation is normalised, single-agent systems (SAS)
  "can match or outperform" multi-agent systems (MAS).
- An information-theoretic argument from the Data Processing Inequality:
  under a fixed reasoning-token budget and perfect context utilisation, a
  single agent is more information-efficient; MAS become competitive when a
  single agent's effective context utilisation degrades or when more compute
  is spent.
- Controlled study across three model families (Qwen3,
  DeepSeek-R1-Distill-Llama, Gemini 2.5), SAS vs "multiple MAS
  architectures under matched budgets"; SAS "consistently match or
  outperform MAS on multi-hop reasoning tasks when reasoning tokens are held
  constant".
- "Significant artifacts in API-based budget control (particularly in
  Gemini 2.5) and in standard benchmarks, both of which can inflate apparent
  gains from MAS."

## Why it matters to us

**Second independent matched-budget result with our sign**, after Kim et
al. (`2026-09-12-google-scaling-agent-systems.md`). Their budget is
*thinking tokens*; Kim et al.'s is *total tokens*; ours is *experiments*,
certified by contract. Three different denominators, one sign. That is the
sentence for §1/§2 of the paper.

Their mechanism (context utilisation, DPI) is the third distinct mechanism
offered for the same sign: Kim et al. say budget fragmentation, we say
duplicated coverage. On our task theirs and Kim's are both ruled out by the
same fact: `one_shot`, a single call with no context accumulation and no
fragmentation, ties the loop. Say that rather than adjudicating their
mechanism on their task.

Their "artifacts in API-based budget control" is register §25/§32 from the
other side: a pinned parameter does not pin the computation. Cite it beside
ours when the reviewer asks whether token budgets can be matched at all.

## Quarantined (not verified)

- Which benchmarks were used (a secondary source names "multi-hop QA";
  the abstract says only "multi-hop reasoning tasks").
- Whether "Stanford" is the affiliation (a secondary source says so; the
  arXiv page shows none).
- The exact budget-matching procedure and the MAS architectures compared.

Read the body before quoting any number from it in the paper.
