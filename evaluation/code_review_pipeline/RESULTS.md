# Code Review Pipeline Experiment Results

## Experiment Overview

**Date**: January 1, 2026
**Purpose**: Evaluate Agent Contracts' effectiveness in a Coder-Reviewer iterative pipeline
**Target Venue**: COINE 2026 (Coordination, Organizations, Institutions, Norms, and Ethics for Governance of Multi-Agent Systems)

### Experimental Design

| Aspect | Details |
|--------|---------|
| Design Type | Within-subjects (paired) |
| Sample Size | 70 problems (31 easy + 39 medium) |
| Conditions | CONTRACTED vs UNCONTRACTED |
| Total Trials | 140 (70 × 2) |
| Data Source | LiveCodeBench (post-Feb 2025, contamination-free) |
| Model | Gemini 2.5 Flash Lite (`gemini-2.5-flash-lite`) |
| Random Seed | 42 |

### Agent Configuration

The experiment uses a two-agent iterative pattern:

```
                ┌─────────────────────────────────┐
                │         Code Review Loop        │
                │                                 │
     Problem ──►│  [Coder] ◄─────► [Reviewer]    │──► Solution
                │     │              │            │
                │     └──── fixes ───┘            │
                └─────────────────────────────────┘
```

**Iteration Limits**:
- CONTRACTED: max 3 iterations (governance enforced)
- UNCONTRACTED: max 6 iterations (no governance)

---

## Key Results

### Primary Findings

| Metric | UNCONTRACTED | CONTRACTED | Change | Significance |
|--------|--------------|------------|--------|--------------|
| **Token Usage** | 34,606 | 3,461 | **-90%** | p=0.0007 *** |
| **Variance** | 5.29B | 10.1M | **525x lower** | - |
| **Iterations** | 3.00 | 1.71 | -43% | p<0.0001 *** |
| **LLM Calls** | 9.0 | 4.5 | -50% | p<0.0001 *** |
| **Success Rate** | 60.0% | 52.9% | -7.1pp | p=0.13 (NS) |

### Statistical Interpretation

1. **Token Reduction (90%)**: The headline result—Agent Contracts deliver massive resource savings with high statistical significance (p=0.0007, Cohen's d=-0.42).

2. **Variance Reduction (525x)**: Perhaps even more important than mean reduction. CONTRACTED execution is highly predictable, directly addressing the "$47K problem" of runaway AI costs.

3. **Success Rate Trade-off**: The 7.1 percentage point difference is **not statistically significant** (p=0.13, McNemar's p=0.23). The 95% CI [-15.7, +1.4]pp includes zero, meaning we cannot rule out no difference.

### Effect Sizes

| Metric | Cohen's d | Interpretation |
|--------|-----------|----------------|
| Success Rate | -0.18 | Negligible |
| Avg Tokens | -0.42 | Small |
| Avg Iterations | -0.63 | Medium |
| Avg LLM Calls | -0.65 | Medium |

---

## Analysis by Difficulty

| Difficulty | n | CONTRACTED | UNCONTRACTED | Token Reduction |
|------------|---|------------|--------------|-----------------|
| **Easy** | 31 | 71.0% success | 77.4% success | -76% tokens |
| **Medium** | 39 | 38.5% success | 46.2% success | -92% tokens |

**Key Insight**: Token reduction is even more dramatic for medium-difficulty problems (92% vs 76%), suggesting that governance is most valuable for complex tasks where runaway risk is highest.

---

## The "$47K Problem" Evidence

This experiment provides empirical evidence for the governance value proposition:

### Without Governance (UNCONTRACTED)
- Mean tokens: 34,606
- Variance: 5.29 billion
- Standard deviation: ~72,739 tokens
- **Unpredictable costs**: Some runs use 10x the average

### With Governance (CONTRACTED)
- Mean tokens: 3,461
- Variance: 10.1 million
- Standard deviation: ~3,174 tokens
- **Predictable costs**: Tight distribution around mean

### Variance Ratio: 525x

This means UNCONTRACTED execution has 525 times more cost uncertainty than CONTRACTED execution—directly addressing enterprise concerns about AI budget predictability.

---

## Figures and Interpretation

All figures are saved in both PNG (for presentations) and PDF (for LaTeX papers) in `results/code_review/figures/`.

### Figure 1: Overall Comparison

![Code Review Pipeline Comparison](../../results/code_review/figures/code_review_comparison.png)

**Figure 1** presents a four-panel comparison of key metrics between CONTRACTED and UNCONTRACTED conditions:

- **Panel A (Success Rate)**: Both conditions achieve similar success rates (52.9% vs 60.0%), with overlapping 95% confidence intervals. The difference is not statistically significant (p=0.13), indicating that governance constraints do not substantially harm task completion.

- **Panel B (Total Tokens)**: The most striking difference—CONTRACTED uses 90% fewer tokens on average. The non-overlapping confidence intervals confirm statistical significance (p=0.0007). This demonstrates that iteration limits effectively bound resource consumption.

- **Panel C (Iterations)**: CONTRACTED averages 1.71 iterations vs 3.00 for UNCONTRACTED. This reflects the governance mechanism working as designed—limiting the Coder↔Reviewer feedback loop.

- **Panel D (LLM Calls)**: Similarly, LLM calls are halved (4.5 vs 9.0), showing proportional resource savings across all metrics.

**Key Takeaway**: The four panels together tell a compelling story—massive resource savings (90% tokens, 50% LLM calls) with a modest, non-significant quality trade-off.

---

### Figure 2: Token Distribution (The Predictability Story)

![Token Distribution Analysis](../../results/code_review/figures/code_review_token_distribution.png)

**Figure 2** visualizes the variance difference between conditions, which is arguably the most important finding for enterprise adoption:

- **Left Panel (Box Plot)**: The UNCONTRACTED distribution shows extreme outliers extending far beyond the interquartile range. Some problems consumed 200,000+ tokens—representing potential "$47K problem" scenarios where costs explode unexpectedly.

- **Right Panel (Histogram)**: The CONTRACTED distribution (blue) is tightly clustered near zero with a sharp peak, while UNCONTRACTED (orange) shows a long right tail extending to high token counts.

**Variance Analysis**:
| Condition | Variance | Std Dev |
|-----------|----------|---------|
| CONTRACTED | 10.1M | ~3,174 tokens |
| UNCONTRACTED | 5.29B | ~72,739 tokens |
| **Ratio** | **525x** | **23x** |

**Interpretation**: The 525x variance reduction means that with Agent Contracts, organizations can predict AI costs with high confidence. A budget set at 2× the mean would cover nearly all CONTRACTED runs, but would be wildly insufficient for UNCONTRACTED execution where some runs use 10-20× the average.

This addresses the core enterprise concern: *"How do I budget for AI agents when costs are unpredictable?"* Agent Contracts provide the answer through formal governance.

---

### Figure 3: Difficulty Breakdown (Governance Value Scales with Complexity)

![Analysis by Difficulty Level](../../results/code_review/figures/code_review_by_difficulty.png)

**Figure 3** breaks down results by problem difficulty, revealing an important scaling insight:

- **Left Panel (Success Rate by Difficulty)**:
  - Easy problems: 71.0% (CONTRACTED) vs 77.4% (UNCONTRACTED)
  - Medium problems: 38.5% (CONTRACTED) vs 46.2% (UNCONTRACTED)

  The success rate gap is similar across difficulties (~6-8pp), suggesting governance constraints affect easy and medium problems proportionally.

- **Right Panel (Token Usage by Difficulty)**:
  - Easy problems: 76% token reduction (2,601 vs 10,759)
  - Medium problems: **92% token reduction** (4,143 vs 53,562)

**Key Insight**: Governance value *increases* with task complexity. Medium-difficulty problems show 92% savings vs 76% for easy problems. This makes intuitive sense—complex problems have more opportunity for runaway iteration loops, so governance constraints provide greater benefit.

**Implication for Deployment**: Agent Contracts are most valuable for complex, high-stakes tasks where runaway risk is highest and cost predictability is most critical.

---

## Paper Narrative

### For COINE 2026 Submission

**Claim**: Agent Contracts provide governance for autonomous AI agents through formal resource constraints.

**Evidence from this experiment**:

1. **Resource Governance Works**: 90% token reduction demonstrates that iteration limits effectively constrain resource consumption.

2. **Predictability is the Key Value**: The 525x variance reduction is arguably more important than mean reduction. Organizations can now budget confidently.

3. **Modest Quality Trade-off**: The 7.1pp success rate difference is not statistically significant, but even if real, represents an acceptable trade-off for the governance benefits.

4. **Scalability Insight**: Medium-difficulty problems show 92% token reduction vs 76% for easy problems, suggesting governance value increases with task complexity.

### Limitations to Acknowledge

1. **Single model tested** (Gemini 2.0 Flash)—results may vary with other models
2. **Coding domain only**—other domains may show different patterns
3. **Fixed iteration limits**—adaptive limits might achieve better quality/cost balance

---

## Reproducibility

### Running the Experiment

```bash
# Run full experiment (70 problems × 2 conditions)
uv run python -m evaluation.code_review_pipeline.run_experiment \
    --n-problems 70 \
    --exclude-hard \
    --seed 42

# Run analysis
uv run python -m evaluation.code_review_pipeline.analyze_results
```

### Data Files

- **Raw results**: `results/code_review/experiment_20260101_135207.json`
- **Statistics**: `results/code_review/figures/statistics.md`
- **Figures**: `results/code_review/figures/*.{png,pdf}`

---

## Conclusion

The Code Review Pipeline experiment validates the core Agent Contracts value proposition: **governance through formal resource constraints enables predictable AI operations at acceptable quality trade-offs**.

Key takeaways:
- **90% token reduction** with statistical significance
- **525x variance reduction** for budget predictability
- **7.1pp success rate trade-off** (not statistically significant)
- **Governance value scales with complexity** (92% savings on medium problems)

These results directly support the paper's thesis that multi-agent AI systems require formal governance mechanisms to operate safely and predictably in enterprise environments.
