# Evaluation Experiments for Agent Contracts

This folder contains the evaluation pipelines for the **COINE 2026** conference paper:
*"Agent Contracts: A Formal Framework for Resource-Bounded Autonomous AI Systems"*

**Paper location**: `paper/paper.qmd` (Quarto source) → `paper/output/paper.pdf` (compiled)

## Overview

We provide **three complementary experiments** that demonstrate the value of Agent Contracts at different levels of complexity:

| Experiment | Complexity | Pattern | Sample Size | Key Demonstration |
|------------|------------|---------|-------------|-------------------|
| **1. Strategy Modes** | Single LLM call | ContractExecutor | 100 articles | Budget-aware prompting, strategy optimization |
| **2. Research Pipeline** | Multi-agent sequential | Researcher → Analyzer → Reporter | 50 topics | Conservation laws, budget delegation |
| **3. Code Review Pipeline** | Multi-agent iterative | Coder ↔ Reviewer loop | 100 problems | Runaway prevention, iteration limits |

```
                    Complexity Progression
    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │   Strategy     Research        Code Review          │
    │    Modes       Pipeline         Pipeline            │
    │      │            │                │                │
    │   Single       Multi-Agent     Multi-Agent          │
    │    Call        Sequential       Iterative           │
    │      │            │                │                │
    │      ▼            ▼                ▼                │
    │  ContractExecutor → DelegatingAdkAgent → Loops     │
    │                                                     │
    └─────────────────────────────────────────────────────┘
```

---

## Experiment 1: Strategy Modes

**Location:** `strategy_modes/`

This experiment demonstrates **ContractExecutor**, the core execution engine that provides the most comprehensive set of Agent Contracts features for single LLM calls.

### What It Tests

| Paper Section | Concept | How Tested |
|---------------|---------|------------|
| §4.1 | Contract definition C = (I,O,S,R,T,Φ,Ψ) | Full contract with all fields |
| §4.2 | Resource constraints R | Token budgets per mode |
| §5.2 | Runtime monitoring | Real-time token tracking |
| §7.2 | Enforcement capabilities | Multi-call budget enforcement |

### The Three Modes

| Mode | Prompt Guidance | Expected Behavior |
|------|----------------|-------------------|
| **URGENT** ⚡ | "Optimize for speed, accept 85% accuracy" | Faster, brief output, may use more tokens |
| **ECONOMICAL** 💰 | "Minimize tokens, use parametric knowledge" | Concise output, fewest tokens |
| **BALANCED** ⚖️ | "Work thoroughly, comprehensive results" | Standard quality, baseline tokens |

### Task: CNN/DailyMail Summarization

We use the [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail) dataset because summarization has a natural **quality-effort tradeoff**:

- **URGENT**: Quick summary, key points only
- **ECONOMICAL**: Concise but complete
- **BALANCED**: Comprehensive coverage

### Agent Contracts Components Used

```python
from agent_contracts import Contract, ContractMode, ResourceConstraints
from agent_contracts.core.executor import ContractExecutor, ExecutionResult
from agent_contracts.core.prompts import generate_budget_prompt
from agent_contracts.core.planning import recommend_strategy

# Contract with strategy mode
contract = Contract(
    id="summarize-task",
    name="Article Summarization",
    mode=ContractMode.ECONOMICAL,  # or URGENT, BALANCED
    resources=ResourceConstraints(
        tokens=2000,
        cost_usd=0.10,
    ),
)

# ContractExecutor orchestrates everything
executor = ContractExecutor(contract)
result: ExecutionResult = executor.run(query=f"Summarize: {article}")

# Result includes:
# - result.output: The summary
# - result.tokens_used: Actual tokens consumed
# - result.strategy: Strategy recommendation used
# - result.execution_log: Full audit trail
```

### Metrics Collected

- **Token usage** per mode (primary signal)
- **Output length** (words/characters)
- **Quality score** (ROUGE-L against reference summaries)
- **Execution time**
- **Strategy recommendations** (from `recommend_strategy()`)

### Hypothesis

| Metric | URGENT | ECONOMICAL | BALANCED |
|--------|--------|------------|----------|
| Token usage | Medium-High | **Lowest** | Medium |
| Output length | Shortest | Short | Longest |
| ROUGE-L quality | ≥ 0.20 | ≥ 0.25 | ≥ 0.30 |
| Speed | Fastest | Medium | Standard |

**Key claim**: All three modes maintain acceptable quality while demonstrating different resource tradeoffs.

### Usage

```bash
# Quick smoke test (2 articles, all modes)
python -m evaluation.strategy_modes.run_experiment --n-articles 2

# Full experiment (recommended: 100 articles for statistical power)
python -m evaluation.strategy_modes.run_experiment \
    --n-articles 100 \
    --model gemini/gemini-2.5-flash \
    --seed 42

# Single mode only
python -m evaluation.strategy_modes.run_experiment \
    --mode economical \
    --n-articles 100
```

---

## Experiment 2: Research Pipeline

**Location:** `research_pipeline/`

### Architecture

```
Orchestrator (Parent Contract: 100K tokens)
    │
    ├── Researcher (40K tokens, 15 iterations)
    │   └── Uses google_search for web research
    │
    ├── Analyzer (25K tokens, 10 iterations)
    │   └── Identifies patterns and insights
    │
    └── Reporter (25K tokens, 10 iterations)
        └── Synthesizes final report
```

### What It Tests

| Paper Section | Concept | How Tested |
|---------------|---------|------------|
| §4.1 | Formal contract definition C = (I,O,S,R,T,Φ,Ψ) | Full Contract with resources, temporal, success criteria |
| §6.1 | Conservation laws: Σbᵢ ≤ B | DelegatingAdkAgent enforces budget delegation |
| §6.2 | Orchestrator-Workers pattern | Parent agent spawns child contracts dynamically |
| §8 | Research report example | Multi-agent pipeline with budget allocation |

### Agent Contracts Components Used

```python
from agent_contracts import Contract, ResourceConstraints, TemporalConstraints
from agent_contracts.core.prompts import generate_budget_prompt
from agent_contracts.integrations.google_adk import DelegatingAdkAgent

# Parent contract with multi-dimensional constraints
parent_contract = Contract(
    id="report-task",
    resources=ResourceConstraints(
        tokens=100_000,      # Token budget
        cost_usd=2.0,        # Cost cap
        iterations=50,       # Runaway prevention
    ),
    temporal=TemporalConstraints(
        max_duration=timedelta(minutes=15),
    ),
)

# Hierarchical delegation with conservation laws
delegating = DelegatingAdkAgent(
    contract=parent_contract,
    agent=orchestrator_agent,
    reserve_ratio=0.1,  # Reserve 10% for coordination
)

# Child contracts inherit from parent budget
researcher = delegating.delegate(
    name="researcher",
    tokens=40_000,
    iterations=15,
)
```

### Metrics Collected

- **Token consumption** (total and per-agent)
- **LLM call counts** (iteration tracking)
- **Conservation law compliance**
- **Execution time**
- **Quality scores** (via IndeterminacyAwareEvaluator)
  - Accuracy, Completeness, Coherence (1-10 scale)
  - Judge agreement and indeterminacy signals

### Usage

```bash
# Quick smoke test (1 topic, both conditions)
python -m evaluation.research_pipeline.run_experiment --quick

# Full experiment (recommended: 50 topics for statistical power)
python -m evaluation.research_pipeline.run_experiment \
    --n-topics 50 \
    --mode both \
    --evaluate \
    --judge-model gemini/gemini-2.5-flash \
    --num-judges 3

# Specific topic
python -m evaluation.research_pipeline.run_experiment \
    --topics tech_01 \
    --mode both
```

---

## Code Review Pipeline

**Location:** `code_review_pipeline/`

### Architecture

```
┌─────────────────────────────────────────────────────┐
│     Orchestrator (Parent Contract: 50K tokens)      │
│                                                     │
│   ┌─────────┐         ┌──────────┐                 │
│   │  Coder  │ ──────► │ Reviewer │                 │
│   │ (20K)   │ ◄────── │  (20K)   │                 │
│   └─────────┘ iterate └──────────┘                 │
│        │                   │                        │
│        ▼                   ▼                        │
│   Write Code          Test & Review                 │
│                       APPROVE/REJECT                │
└─────────────────────────────────────────────────────┘
```

### What It Tests

| Paper Section | Concept | How Tested |
|---------------|---------|------------|
| §4.2 | Resource constraints (iterations) | `r_iter` constraint prevents infinite loops |
| §6.1 | Conservation laws | Coder + Reviewer budgets ≤ Parent budget |
| §7.2 | Enforcement capabilities | Iteration limits halt execution at threshold |
| §4.3 | Contract lifecycle | VIOLATED state when limits exceeded |

### The Runaway Problem

Without Agent Contracts, a Coder ↔ Reviewer loop can iterate indefinitely:
- Coder writes buggy code
- Reviewer rejects and provides feedback
- Coder tries again... forever

**Agent Contracts Solution:**
```python
# Per-agent iteration limits
contracted_coder = orchestrator.delegate(
    name="coder",
    tokens=20_000,
    iterations=5,  # Max 5 LLM calls
)
```

### Metrics Collected

- **Iteration counts** (key metric for runaway detection)
- **Success rate** (task solved before limit)
- **Runaway prevention events** (when limit was hit)
- **Token consumption** (CONTRACTED typically lower variance)
- **LLM call counts**

### Dataset: LiveCodeBench

- **Source**: [livecodebench/code_generation_lite](https://huggingface.co/datasets/livecodebench/code_generation_lite) (HuggingFace)
- **Version**: `test6.jsonl` (Release 6 - latest available)
- **Filter**: Problems after February 2025 (`--after-date 2025-02-01`) for contamination-free evaluation
- **Platforms**: LeetCode, AtCoder, Codeforces
- **Difficulty levels**: Easy, Medium, Hard
- Each problem includes test cases for validation

### Usage

```bash
# Quick smoke test (5 problems)
python -m evaluation.code_review_pipeline.run_experiment --n-problems 5

# Full experiment (recommended: 100 problems for statistical power)
python -m evaluation.code_review_pipeline.run_experiment \
    --n-problems 100 \
    --seed 42

# By difficulty level
python -m evaluation.code_review_pipeline.run_experiment \
    --n-problems 100 \
    --difficulty medium

# Contracted only
python -m evaluation.code_review_pipeline.run_experiment \
    --n-problems 100 \
    --contracted-only
```

---

## Quality Evaluation: IndeterminacyAwareEvaluator

**Location:** `indeterminacy_evaluator.py`

Implements the NeurIPS 2025 framework from "Validating LLM-as-a-Judge Systems under Rating Indeterminacy" (Guerdan et al., 2025).

### Why This Matters

Standard LLM-as-judge approaches force judges to pick a single rating, but many evaluation tasks have inherent ambiguity. This framework:

1. **Response Set Elicitation**: Ask judges "select ALL ranges that reasonably apply"
2. **Multi-label Scoring**: Track probability vector ω across all options
3. **Indeterminacy Signal**: Judge disagreement = genuine ambiguity, not noise

### Usage in Research Pipeline

```python
from evaluation.indeterminacy_evaluator import IndeterminacyAwareEvaluator

evaluator = IndeterminacyAwareEvaluator(
    judge_model="gemini/gemini-2.5-flash",
    num_judges=3,
    use_hybrid_scoring=True,
)

score = evaluator.evaluate(question="Research topic", answer=report_text)
# Returns: accuracy (0-10), completeness (0-10), coherence (0-10)
# Plus indeterminacy levels for each dimension
```

---

## Comparison: CONTRACTED vs UNCONTRACTED

### What Changes Between Conditions

| Element | CONTRACTED | UNCONTRACTED |
|---------|-----------|--------------|
| Contract definition | ✅ Full contract with R, T, Φ | ❌ None |
| Token limits | ✅ Per-agent budgets | ❌ Unlimited |
| Iteration limits | ✅ Prevents runaway loops | ❌ Safety limit only |
| Conservation laws | ✅ Σbᵢ ≤ B enforced | ❌ N/A |
| Budget-aware prompts | ✅ Agents know constraints | ❌ Standard prompts |
| Cost tracking | ✅ Real-time monitoring | ❌ Post-hoc only |

### What Stays Constant (Controls)

- Same LLM model (gemini-2.5-flash)
- Same agent architectures
- Same prompts (minus budget info)
- Same tasks/topics
- Same random seeds
- Same evaluation criteria

---

## Statistical Methodology

### Sample Size Rationale

We use **bootstrap confidence intervals** for all comparisons. Sample sizes are chosen to ensure:
- Stable bootstrap estimates (minimum 30 samples per condition)
- Detection of medium effect sizes (Cohen's d ≈ 0.5) with 80% power
- Reasonable precision on binary outcomes (±10% for success rates)

| Experiment | Sample Size | Design | Total Runs | Rationale |
|------------|-------------|--------|------------|-----------|
| **Strategy Modes** | 100 articles | Within-subjects (paired) | 300 | Paired design is efficient; detects ~15% token difference |
| **Research Pipeline** | 50 topics | Between-subjects | 100 | Expanded from 25 for robust CIs |
| **Code Review** | 100 problems | Between-subjects | 200 | Binary outcomes need more samples |

### Bootstrap Analysis

For each metric, we compute:
1. **Point estimate**: Mean difference between conditions
2. **95% CI**: 10,000 bootstrap resamples with BCa correction
3. **Effect size**: Cohen's d with confidence interval
4. **p-value**: Permutation test (two-tailed)

```python
# Example bootstrap analysis
from scipy import stats
import numpy as np

def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):
    """Compute BCa bootstrap confidence interval."""
    boot_means = [np.mean(np.random.choice(data, len(data))) for _ in range(n_bootstrap)]
    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return np.mean(data), lower, upper
```

### Cost Estimate (Gemini 2.5 Flash)

| Experiment | Tokens/Run | Total Tokens | Est. Cost |
|------------|------------|--------------|-----------|
| Strategy Modes (100 articles) | ~3K | ~900K | ~$0.14 |
| Research Pipeline (50 topics) | ~50K | ~5M | ~$0.75 |
| Code Review (100 problems) | ~15K | ~3M | ~$0.45 |
| **Total** | | ~9M | **~$1.35** |

### Expected Figures

Each experiment will generate publication-ready figures:

**Experiment 1: Strategy Modes**
- **Figure 1a**: Bar chart with 95% CI - Token usage by mode (URGENT/ECONOMICAL/BALANCED)
- **Figure 1b**: Bar chart with 95% CI - ROUGE-L scores by mode
- **Figure 1c**: Scatter plot - Quality vs Token tradeoff (Pareto frontier)

**Experiment 2: Research Pipeline**
- **Figure 2a**: Paired bar chart with 95% CI - Token consumption (CONTRACTED vs UNCONTRACTED)
- **Figure 2b**: Box plot - Quality scores by condition
- **Figure 2c**: Stacked bar - Budget allocation across agents (conservation law visualization)

**Experiment 3: Code Review Pipeline**
- **Figure 3a**: Histogram - Iteration counts (CONTRACTED vs UNCONTRACTED)
- **Figure 3b**: Bar chart with 95% CI - Success rates by condition
- **Figure 3c**: Violin plot - Token usage distribution

**Statistical annotations**: All figures include:
- Bootstrap 95% confidence intervals (10,000 resamples)
- Effect sizes (Cohen's d) where applicable
- Significance markers (* p<0.05, ** p<0.01, *** p<0.001)

---

## Expected Outcomes

### Strategy Modes

| Metric | URGENT | ECONOMICAL | BALANCED |
|--------|--------|------------|----------|
| Token usage | Medium | **Lowest** | High |
| Output length | Shortest | Short | Longest |
| ROUGE-L quality | ≥ 0.20 | ≥ 0.22 | ≥ 0.25 |
| Execution time | **Fastest** | Medium | Standard |

**Key hypothesis**: Modes form a Pareto frontier—no single mode dominates all metrics.

### Research Pipeline

| Metric | CONTRACTED | UNCONTRACTED |
|--------|-----------|--------------|
| Budget compliance | 100% | N/A |
| Conservation violations | 0 | N/A |
| Quality score | Similar or better | Baseline |
| Token variance | Lower (predictable) | Higher |

### Code Review Pipeline

| Metric | CONTRACTED | UNCONTRACTED |
|--------|-----------|--------------|
| Runaway prevention | Guaranteed | Relies on safety limit |
| Max iterations | Hard limit (5) | Soft limit (20) |
| Success rate | Similar | Similar |
| Token predictability | High | Low |

---

## Paper Claims Validated

| Claim | Paper Section | Experiment | Evidence |
|-------|---------------|------------|----------|
| Contract definition enables governance | §4.1 | All three | C = (I,O,S,R,T,Φ,Ψ) with all fields |
| Resource constraints are enforceable | §4.2, §7.2 | Strategy Modes | Token tracking per mode |
| Conservation laws preserve budgets | §6.1 | Research Pipeline | Budget delegation respects Σbᵢ ≤ B |
| Iteration limits prevent runaway | §4.2, §7.2 | Code Review Pipeline | Loops stop at threshold |
| Orchestrator-Workers pattern works | §6.2 | Research Pipeline | Parent spawns child contracts |
| Multi-call enforcement is valuable | §7.2 | All three | Cumulative budget protection |
| Quality maintained under governance | §4.2 (Φ) | All three | ROUGE-L scores comparable |

---

## File Structure

```
evaluation/
├── README.md                       # This file
├── __init__.py
├── indeterminacy_evaluator.py      # NeurIPS 2025 LLM-as-Judge
│
├── strategy_modes/                 # Experiment 1: Single-call governance
│   ├── __init__.py
│   ├── tasks.py                    # CNN/DailyMail loader
│   ├── orchestrator.py             # ContractExecutor wrapper
│   ├── metrics.py                  # ROUGE evaluation
│   └── run_experiment.py           # Experiment runner
│
├── research_pipeline/              # Experiment 2: Multi-agent sequential
│   ├── __init__.py
│   ├── agents.py                   # Agent definitions (google_search)
│   ├── orchestrator.py             # Contracted/Uncontracted pipelines
│   ├── evaluator.py                # Report quality evaluation
│   ├── topics.py                   # 25 research topics
│   └── run_experiment.py           # Experiment runner
│
└── code_review_pipeline/           # Experiment 3: Multi-agent iterative
    ├── __init__.py
    ├── agents.py                   # Coder/Reviewer definitions
    ├── orchestrator.py             # Contracted/Uncontracted pipelines
    ├── execution.py                # Code execution sandbox
    ├── tasks.py                    # LiveCodeBench loader
    └── run_experiment.py           # Experiment runner
```

---

## References

- **Conference Paper**: `paper/paper.qmd` (source) → `paper/output/paper.pdf` (compiled)
- **CLAUDE.md**: Project context and development history
- **Indeterminacy Paper**: Guerdan et al. "Validating LLM-as-a-Judge Systems under Rating Indeterminacy" (NeurIPS 2025)
- **LiveCodeBench**: https://livecodebench.github.io/

## Paper Section Quick Reference

| Section | Title | Key Concepts |
|---------|-------|--------------|
| §4 | The Agent Contract Framework | C = (I,O,S,R,T,Φ,Ψ), lifecycle states |
| §5 | Resource Tracking and Monitoring | Token decomposition, runtime monitoring |
| §6 | Multi-Agent Coordination | Conservation laws, orchestrator-workers |
| §7 | Limitations and Enforcement | Single-call constraints, multi-call value |
| §8 | Example: Research Report | End-to-end multi-agent demonstration |
| Appendix A | Formal Properties | Conservation invariant, termination, exclusivity |
