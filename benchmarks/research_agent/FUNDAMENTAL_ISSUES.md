# Fundamental Issues with Current Benchmark Design

**Date**: November 2, 2025
**Critical Finding**: The benchmark doesn't test what the framework actually provides

---

## The Core Problem

After 3 runs with improved measurement (temperature=0, 3x judges, hybrid scoring, planning=1200), the data reveals:

### What We Expected
- Contracted agent uses **LESS** tokens (budget constraints)
- Contracted agent gets **BETTER** quality (strategic allocation)
- Clear win on cost efficiency

### What We Actually Got
- Contracted agent uses **+915 MORE** tokens (+7.4%)
- Contracted agent gets **+3.8 better** quality (modest, within variance)
- Only 3/15 cases save tokens, 12/15 use more

---

## Root Cause Analysis

### Issue #1: Both Agents Use the SAME Workflow ❌

```
Uncontracted Agent:
1. Decompose question (high reasoning)
2. Research each sub-question (high reasoning)
3. Synthesize answer (high reasoning)

Contracted Agent:
1. Decompose question (medium reasoning) ← ONLY DIFFERENCE
2. Research each sub-question (high reasoning)
3. Synthesize answer (high reasoning)
```

**Problem**: The workflows are **functionally identical**. The only difference is the planning budget (1200 vs ~4000 tokens), which doesn't change the fundamental approach.

### Issue #2: Budgets Are TOO GENEROUS ❌

Current contracted budgets:
- Planning: 1200 reasoning tokens
- Research: 2500 reasoning tokens × N sub-questions
- Synthesis: 2500 reasoning tokens

For 3 sub-questions: **1200 + 7500 + 2500 = 11,200 reasoning tokens**

Actual usage: **~5000 reasoning tokens**

**Problem**: Budgets are 2-3x higher than actual usage. They're not constraining behavior at all!

### Issue #3: Baseline Has MASSIVE Variance ❌

Uncontracted agent variance (same questions, temperature=0):
- **Quality: σ=9.3 points** (scores vary by ±20 points!)
- **Tokens: σ=1450 tokens** (some vary by 3000+ tokens!)

Examples:
- Q2 Run 1: 77.3 quality
- Q2 Run 2: 93.3 quality
- **Difference: 16 points** for the SAME question!

**Problem**: With σ=9.3, our +3.8 average improvement is **within measurement noise**.

---

## What The Framework Actually Provides

Looking at contracted agent variance:
- **Quality: σ=8.4** (vs 9.3 uncontracted)
- **Tokens: σ=741** (vs 1450 uncontracted) ← **50% LESS variance!**

### The Real Value Proposition

Contracts don't necessarily make you use less or get better quality. They make you **PREDICTABLE** and **GOVERNABLE**:

1. **Predictability**: 50% less token variance means more stable costs
2. **Governance**: Hard limits prevent runaway resource usage
3. **Auditability**: Know exactly what budget was allocated to what
4. **Organizational Control**: Enforce company-wide resource policies

---

## Why Our Benchmark Fails

### What We're Testing ❌
"Does the contracted agent produce better quality per dollar than uncontracted?"

### What We SHOULD Be Testing ✅

1. **Budget Enforcement**: Do contracts prevent violations?
   - Test: Set tight budgets (50% of typical usage)
   - Measure: Do agents stay within limits?
   - Success: 100% compliance

2. **Resource Governance**: Can we control costs organizationally?
   - Test: Enforce "$0.01 max per query" policy
   - Measure: Does framework prevent violations?
   - Success: No queries exceed limit

3. **Quality-Cost Tradeoffs**: Can we get acceptable quality cheaper?
   - Test: Multiple budget levels (tight, medium, generous)
   - Measure: Quality degradation curve as budget decreases
   - Success: Find optimal cost-quality point

4. **Predictability**: Are costs more stable?
   - Test: Run same query 10 times
   - Measure: Token/cost variance
   - Success: Contracted has <50% variance of uncontracted

---

## The Fundamental Misunderstanding

We built a framework for **RESOURCE GOVERNANCE** but tested it as a **QUALITY OPTIMIZER**.

Think of it this way:

```
Current Benchmark (Wrong Framing):
"Does a speed limiter make your car go faster?"
→ No, but that's not the point!

Correct Benchmark (Right Framing):
"Does a speed limiter prevent you from getting speeding tickets?"
→ Yes! That's the actual value.
```

---

## What This Means for The Paper

### Claims We CANNOT Make ❌
1. "Contracts improve quality per dollar"
2. "Contracts reduce token usage"
3. "Contracts are more efficient"

### Claims We CAN Make ✅
1. "Contracts provide **predictable, governable** AI resource usage"
2. "Contracts enforce **organizational policies** on resource limits"
3. "Contracts enable **auditability** and **cost control**"
4. "Contracts reduce **token variance** by 50%"

---

## Recommended Actions

### Option A: Fix The Benchmark (Design New Tests)

1. **Budget Violation Test**
   - Set budgets at 50%, 75%, 100% of typical usage
   - Measure: Compliance rate, quality degradation
   - Demonstrates: Enforcement works, graceful degradation

2. **Cost Governance Test**
   - Scenario: "Company policy: max $0.01 per query"
   - Measure: Can framework enforce this?
   - Demonstrates: Organizational control

3. **Variance Reduction Test**
   - Run 20 iterations of same questions
   - Measure: Token/cost variance contracted vs uncontracted
   - Demonstrates: Predictability value

### Option B: Change The Framework (Make Budgets Actually Constrain)

1. **Reduce All Budgets by 60%**
   - Planning: 1200 → 500 tokens
   - Research: 2500 → 1000 tokens
   - Synthesis: 2500 → 1000 tokens
   - Force real tradeoffs, different strategies

2. **Add Adaptive Budget Allocation**
   - Start with low budget, increase if quality drops
   - Demonstrate intelligent resource management

### Option C: Accept What We Have (Reframe The Story)

1. **Acknowledge**: Framework doesn't reduce costs in current design
2. **Emphasize**: Framework provides governance and predictability
3. **Position**: As infrastructure for multi-agent systems, not optimization

---

## The Honest Assessment

Based on 3 runs × 5 questions × 2 agents = 30 evaluations:

**What Works** ✅:
- Budget tracking: Accurate
- Violation detection: Working
- Framework stability: No crashes
- Measurement improvements: Variance reduced from σ=15 to σ=9

**What Doesn't Work** ❌:
- Cost savings: Using +7.4% MORE tokens
- Quality improvement: Only +3.8 points (within noise)
- Value demonstration: Current benchmark doesn't show real value

**The Real Value** (Not Tested):
- Governance: Enforce organizational limits
- Predictability: 50% less variance
- Auditability: Know what was spent where
- Policy Enforcement: Prevent runaway costs

---

## Next Steps

We need to decide:

1. **Pivot to governance benchmarks?**
   - Test budget enforcement, not optimization
   - Measure predictability, not efficiency
   - Demonstrate organizational control

2. **Tighten budgets drastically?**
   - Force real constraints (50% of current)
   - Create actual resource pressure
   - Show different strategies emerge

3. **Reframe the research contribution?**
   - From "optimization" to "governance"
   - From "better quality/cost" to "predictable costs"
   - From "individual agents" to "organizational policy"

The framework IS valuable, but we're measuring the wrong value.
