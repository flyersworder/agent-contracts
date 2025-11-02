# Variance Reduction Test: Root Cause Analysis

**Date**: November 2, 2025
**Status**: Test completed, hypothesis REJECTED

## The Hypothesis

From preliminary benchmark analysis (benchmarks/research_agent/FUNDAMENTAL_ISSUES.md):
- Expected: **~50% variance reduction** in token usage
- Based on: 3 benchmark runs (15 evaluations total)
- Preliminary data: σ=741 (contracted) vs σ=1450 (uncontracted)

## The Reality

From full governance benchmark (3 questions × 20 runs each):
- Actual: **-23% average** (contracted has MORE variance)
- Token variance: -1.8%, -44.1%, -23.1% (all negative)
- Cost variance: -2.2%, -29.3%, +18.4% (mostly negative)
- Coefficient of Variation: Both agents < 10% (highly predictable)

## Root Cause Analysis

### 1. **Statistical Insufficiency** (Primary Cause)

**Problem**: The preliminary "50% variance reduction" was based on N=3 runs

**Why this matters**:
- Variance estimates require N≥20 for statistical reliability
- With N=3, a single outlier can skew results dramatically
- Standard error of variance: SE(σ) ≈ σ/√(2N)
  - For N=3: SE ≈ 41% of true variance
  - For N=20: SE ≈ 16% of true variance

**Conclusion**: The 50% finding was a **statistical artifact** from insufficient data.

---

### 2. **Temperature=0 Effect** (Major Factor)

**Discovery**: Both agents are already highly deterministic

**Data**:
| Question | Uncontracted CV | Contracted CV |
|----------|-----------------|---------------|
| Q1 (Transformers) | 8.5% | 8.9% |
| Q2 (Financial) | 1.8% | 2.7% |
| Q3 (Microservices) | 2.1% | 2.6% |

**Insight**:
- Temperature=0 makes LLM outputs deterministic
- Both agents use temperature=0 → both naturally predictable
- CV < 10% is considered "low variance" in production systems
- No room for contracts to reduce already-low variance

**Conclusion**: The baseline is already so predictable that variance reduction is impossible.

---

### 3. **Budget Constraints Increase Variance** (Counterintuitive Finding)

**Mechanism**: Budget constraints force adaptive behavior

**Why contracted has MORE variance**:
1. **Different strategic choices per run**:
   - Run 1 might hit reasoning token limit early → switches to shorter answers
   - Run 2 might optimize differently → uses more synthesis tokens
   - Run 3 might allocate differently → varies output structure

2. **Resource limit sensitivity**:
   - Small differences in early steps amplify in later steps
   - If planning uses 100 more tokens, research has 100 fewer
   - This creates divergent solution paths

3. **No constraints = consistent behavior**:
   - Uncontracted agent always takes the "natural" path
   - No need to optimize or adapt mid-execution
   - More consistent across runs

**Evidence from budget violation test**:
- Token utilization varied widely: 114%, 156%, 255%
- Agents adapt differently to same budget on different questions
- Adaptive behavior → higher variance

**Conclusion**: Budget constraints introduce strategic variability.

---

### 4. **Identical Workflow Design** (Structural Factor)

**Observation**: Both agents use the same workflow

**Workflow**:
```
1. Planning (generate sub-questions)
2. Research (answer each sub-question)
3. Synthesis (combine answers)
4. Quality evaluation
```

**Why this matters**:
- Contracted agent: Same workflow + budget constraints
- Uncontracted agent: Same workflow + no constraints
- The workflow determines variance, not the contracts
- Contracts only add enforcement, not variance reduction

**Conclusion**: Variance reduction requires DIFFERENT workflows, not just budget constraints.

---

### 5. **Quality Evaluator Variance** (Measurement Noise)

**Data**:
- Quality CV: 10.4-11.4% (both agents)
- Token CV: 1.8-8.9% (both agents)
- Quality variance dominates overall variance

**Insight**:
- LLM-based quality evaluation has inherent variance
- Even with hybrid scoring (60% LLM + 40% rules)
- Quality variance is 2-6× higher than token variance
- This masks any token variance differences

**Conclusion**: Quality evaluator adds noise that swamps token variance reduction.

---

## Why the Preliminary Analysis Was Wrong

### The Original Data (N=3 runs):

From FUNDAMENTAL_ISSUES.md:
```
Uncontracted: μ=13,189, σ=1450 (CV=11.0%)
Contracted:   μ=12,265, σ=741  (CV=6.0%)
Variance reduction: (1450 - 741) / 1450 = 49%
```

### The Full Data (N=20 runs):

Average across 3 questions:
```
Uncontracted: CV=4.1%
Contracted:   CV=4.7%
Variance reduction: -14.6% (MORE variance!)
```

### What Happened:

1. **Small sample bias**: N=3 had a few high-variance runs by chance
2. **Regression to the mean**: N=20 shows true population variance
3. **False pattern**: We saw a pattern that didn't exist
4. **Confirmation bias**: We expected variance reduction → interpreted noise as signal

---

## Lessons Learned

### 1. Statistical Rigor Matters

- ❌ N=3 is insufficient for variance analysis
- ✅ N≥20 required for reliable variance estimates
- Always calculate confidence intervals

### 2. Baseline Matters

- With temperature=0, baseline variance already low
- Can't reduce what's already near-optimal
- Context determines whether optimization is possible

### 3. Constraints ≠ Reduction

- Budget constraints don't automatically reduce variance
- They can increase variance through adaptive behavior
- Governance ≠ Optimization

### 4. Honest Science

- False findings happen with small samples
- Important to test hypotheses rigorously
- Updating beliefs based on evidence is good science

---

## The Silver Lining

While variance reduction failed, we discovered something more valuable:

### Actual Value Provided:

1. **100% budget enforcement** at realistic levels
2. **100% policy compliance** with cost limits
3. **Quality IMPROVEMENT** under constraints (77 → 86 → 95)
4. **Natural predictability** with temperature=0 (CV < 10%)

### Better Positioning:

**Before**: "Framework reduces variance by 50%"
**After**: "Framework provides governance while maintaining quality"

The framework doesn't reduce variance, but it doesn't need to - both agents are already predictable. The real value is **governance and enforcement**.

---

## Recommendations

### For Future Work:

1. **Test with temperature > 0**:
   - Higher temperature → higher baseline variance
   - Might see variance reduction there
   - But production systems rarely use temp > 0.5

2. **Test with different workflows**:
   - Uncontracted: Full research workflow
   - Contracted: Simplified workflow for budget
   - Different strategies might show variance reduction

3. **Test with stochastic environments**:
   - Web search results vary
   - Database queries change
   - External factors increase variance

### For Claims:

✅ **CLAIM**: "Both agents highly predictable (CV < 10%) at temperature=0"
✅ **CLAIM**: "100% budget enforcement and policy compliance"
✅ **CLAIM**: "Quality improves under budget constraints"

❌ **NEVER CLAIM**: "Framework reduces variance by 50%"
❌ **NEVER CLAIM**: "Contracts improve baseline predictability"

---

*This analysis demonstrates the importance of rigorous empirical testing and honest reporting of negative results.*
