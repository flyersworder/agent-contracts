## Statistical Summary (n=70 problems, within-subjects)

| Metric | UNCONTRACTED | CONTRACTED | Difference | Effect Size (d) | 95% CI (diff) |
|--------|--------------|------------|------------|-----------------|---------------|
| Success Rate | 60.0% | 52.9% | -7.1pp | -0.18 (negligible) | [-15.7, +1.4]pp |
| Avg Tokens | 34,606 | 3,461 | -90% | -0.42 (small) | [-49,416, -15,617] |
| Avg Iterations | 3.00 | 1.71 | -1.29 | -0.63 (medium) | [-1.77, -0.81] |
| Avg LLM Calls | 9.0 | 4.5 | -4.46 | -0.65 (medium) | [-6.09, -2.87] |
| Runaway Prevented | 0 | 0 | -0.09 | -0.18 (negligible) | [-0.20, +0.01] |

### Variance Analysis (Predictability)

- CONTRACTED token variance: 10,074,390
- UNCONTRACTED token variance: 5,290,273,210
- **Variance ratio: 525x** (UNCONTRACTED / CONTRACTED)

### Statistical Tests (Paired)

- **Success Rate**: Paired t-test t=-1.522, p=0.1327 (not significant)
- **Avg Tokens**: Paired t-test t=-3.555, p=0.0007 (**significant**)
- **Avg Iterations**: Paired t-test t=-5.245, p=0.0000 (**significant**)
- **Avg LLM Calls**: Paired t-test t=-5.461, p=0.0000 (**significant**)
- **Runaway Prevented**: Paired t-test t=-1.514, p=0.1347 (not significant)

### McNemar's Test (Success Rate)

- Both succeed: 34, CONTRACTED only: 3, UNCONTRACTED only: 8, Both fail: 25
- McNemar exact test: p=0.2266 (not significant)

## Analysis by Difficulty Level

| Difficulty | n | CONTRACTED Success | UNCONTRACTED Success | CONTRACTED Tokens | UNCONTRACTED Tokens | Token Reduction |
|------------|---|-------------------|---------------------|-------------------|---------------------|-----------------|
| EASY | 31 | 71.0% | 77.4% | 2,601 | 10,759 | -76% |
| MEDIUM | 39 | 38.5% | 46.2% | 4,143 | 53,562 | -92% |
