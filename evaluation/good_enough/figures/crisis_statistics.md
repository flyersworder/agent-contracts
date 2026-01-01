## Statistical Summary (n=24 crisis scenarios)

| Metric | UNCONSTRAINED | CONTRACTED | Difference | Effect Size (Cohen's d) | 95% CI (CONTRACTED) |
|--------|---------------|------------|------------|-------------------------|---------------------|
| Iterations | 1.29 ± 0.46 | 1.04 ± 0.20 | -19.4% | 0.70 (medium) | [1.00, 1.12] |
| Tokens | 697 ± 337 | 538 ± 189 | -22.9% | 0.58 (medium) | [474, 620] |
| Quality | 0.828 ± 0.181 | 0.864 ± 0.041 | +0.036 | -0.28 (small) | [0.848, 0.880] |

### Statistical Tests

- **Iterations**: Paired t-test t=2.769, p=0.0109 (**significant**)
- **Tokens**: Paired t-test t=3.102, p=0.0050 (**significant**)
- **Quality**: Paired t-test t=-1.012, p=0.3220 (not significant)

### Non-parametric Tests (Wilcoxon Signed-Rank)

- **Iterations**: W=0.0, p=0.0143 (**significant**)
- **Tokens**: W=48.0, p=0.0036 (**significant**)
- **Quality**: W=121.5, p=0.6157 (not significant)

## Analysis by Urgency Level

| Urgency | n | UNCONSTRAINED Tokens | CONTRACTED Tokens | Token Reduction | UNCONSTRAINED Quality | CONTRACTED Quality |
|---------|---|---------------------|-------------------|-----------------|----------------------|-------------------|
| CRITICAL | 15 | 681 | 545 | -20.1% | 0.863 | 0.876 |
| HIGH | 9 | 723 | 526 | -27.3% | 0.768 | 0.842 |

**Critical scenarios** (n=15): max_iterations=2, highest time pressure
**High scenarios** (n=9): max_iterations=3, moderate time pressure
