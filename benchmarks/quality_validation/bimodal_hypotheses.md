# Bimodal Behavior at High Quality: Hypotheses

**Date**: November 4, 2025
**Observation**: Sample 2 (excellent quality, 96.0) shows bimodal scoring (78.0 vs 96.0), while Sample 1 (good quality, 77.3) shows perfect consistency.

**Core Question**: Why does evaluation variance increase at high quality levels?

---

## Hypothesis 1: Ceiling Effect (Objective Constraint)

**Claim**: High-quality answers hit a scoring ceiling where small differences become subjective.

**Mechanism**:
- **Low-quality answers (0-70)**: Clear, objective flaws (missing information, errors, poor structure)
- **Mid-quality answers (70-85)**: Some flaws are present and identifiable
- **High-quality answers (85-100)**: No obvious flaws; differences are subtle (depth, elegance, insight)

**Why bimodal at high end?**
- Evaluator must decide: "Is this merely complete (8/10) or truly exceptional (10/10)?"
- Binary decision → bimodal distribution
- At lower quality: Gradual scale with clear anchors

**Testable prediction**:
- Answers in 70-85 range should show low variance (clear flaws to identify)
- Answers in 90+ range should show high variance (subjective excellence)

**Evidence needed**:
- Test samples at 70-75, 80-85, 85-90, 95-100
- Map variance as a function of quality level

---

## Hypothesis 2: Reasoning Token Stochasticity (Technical Mechanism)

**Claim**: Gemini's `reasoning_effort="medium"` introduces non-determinism even at `temperature=0`.

**Mechanism**:
- **Reasoning tokens**: Internal chain-of-thought before generating output
- Even at temp=0, reasoning process may explore different paths
- For complex answers, different reasoning paths → different conclusions

**Why bimodal at high end?**
- **Simple answers**: Obvious conclusion, single reasoning path
- **Complex answers**: Multiple valid reasoning paths
  - Path A: "Focuses on thoroughness, no major flaws" → 8/10
  - Path B: "Recognizes exceptional depth and insight" → 10/10

**Testable prediction**:
- Testing with `reasoning_effort="low"` should reduce variance
- Testing with `reasoning_effort="high"` might increase variance
- Non-reasoning models (GPT-4o) should show different behavior

**Evidence needed**:
- Re-run Sample 2 with `reasoning_effort="low"` (N=10)
- Compare with GPT-4o (no reasoning tokens)

---

## Hypothesis 3: Answer Complexity and Evaluation Surface Area

**Claim**: Longer, more sophisticated answers have more "attack surface" for finding issues.

**Mechanism**:
- **Sample 1** (1043 words, good): Limited scope, evaluator has clear boundaries
- **Sample 2** (1081 words, excellent): More content, more nuance, more places to disagree

**Why bimodal at high end?**
- Evaluator must weigh: "Is the extra depth valuable or excessive?"
- Sometimes sees depth as completeness (10/10)
- Sometimes sees depth as verbose without proportional value (8/10)

**Testable prediction**:
- Short excellent answers (<500 words) should be more stable
- Long excellent answers (>1000 words) should show more variance

**Evidence needed**:
- Compare variance across answer lengths at same quality level

---

## Hypothesis 4: Hybrid Scoring Instability at Boundaries

**Claim**: The 60% LLM + 40% rule-based scoring creates instability near scoring boundaries.

**Mechanism**:
- **Rule-based component (40%)**: Deterministic (word count, structure, technical terms)
- **LLM component (60%)**: Stochastic (subjective judgment)
- At high quality, small LLM swings get amplified by the weighting

**Why bimodal at high end?**
- **Low quality**: Both components agree (clearly flawed)
- **High quality**: Rule-based says "excellent", LLM sometimes disagrees
- 60/40 weighting creates discrete jumps

**Example calculation**:
```
Scenario 1 (Low LLM):
  LLM: (7, 6, 7) × 0.6 = (4.2, 3.6, 4.2)
  Rules: (10, 9, 10) × 0.4 = (4.0, 3.6, 4.0)
  Total: (8.2, 7.2, 8.2) → 78.0

Scenario 2 (High LLM):
  LLM: (10, 9, 10) × 0.6 = (6.0, 5.4, 6.0)
  Rules: (10, 9, 10) × 0.4 = (4.0, 3.6, 4.0)
  Total: (10.0, 9.0, 10.0) → 96.7
```

**Testable prediction**:
- Pure LLM (100%, no rules) should show smoother variance
- Pure rules (100%, no LLM) should show zero variance
- Different weightings (50/50, 70/30) should change bimodal gap

**Evidence needed**:
- Re-run with different hybrid weightings
- Compare pure LLM vs pure rules vs hybrid

---

## Hypothesis 5: Median Amplification Effect (Statistical Artifact)

**Claim**: Using median of 3 judges amplifies disagreements into discrete states.

**Mechanism**:
- **3 judges with small disagreements**: (8, 9, 10) → median = 9
- **3 judges with larger disagreements**: (8, 8, 10) → median = 8
- Median is discontinuous → creates jumps rather than smooth variance

**Why bimodal at high end?**
- High-quality answers cause judge disagreement:
  - 2 judges: "Good enough" (8/10)
  - 1 judge: "Exceptional" (10/10)
  - Median: 8 (low state)
- When 2 agree on "exceptional": Median: 10 (high state)

**Testable prediction**:
- Using mean instead of median should show smoother, unimodal distribution
- Using 5 judges instead of 3 should reduce discrete jumps
- Single judge (N=1) should show continuous variance if it exists

**Evidence needed**:
- Re-calculate statistics using mean instead of median
- Test with 5 judges or single judge

---

## Hypothesis 6: Prompt Specification Ambiguity

**Claim**: The evaluation rubric doesn't clearly define "excellent" (10/10) vs "very good" (8/10).

**Mechanism**:
- **Rubric says**: "Accuracy (0-10): Are the facts correct?"
- **Ambiguity**: What's the difference between 8/10 and 10/10 if facts are correct?
- Evaluator fills in the gap inconsistently

**Why bimodal at high end?**
- **Low quality**: Clear guidance ("are there errors?") → consistent
- **High quality**: Vague guidance ("how excellent?") → inconsistent
- Evaluator creates two internal standards:
  - Conservative: "10/10 requires perfection"
  - Generous: "10/10 means excellent execution"

**Testable prediction**:
- Adding calibration examples ("This is an 8/10", "This is a 10/10") should reduce variance
- More detailed rubric with specific criteria for 8 vs 10 should stabilize scores

**Evidence needed**:
- Test with enhanced prompt including calibration examples
- Compare variance before/after prompt clarification

---

## Hypothesis 7: Answer-Type Interaction Effect

**Claim**: Both samples answer the same question (Raft vs Paxos), but one answer style is easier to evaluate.

**Mechanism**:
- **Sample 1**: Straightforward comparison, clear structure
- **Sample 2**: More sophisticated analysis with deeper insights
- Evaluator has clearer heuristics for "good" answers than "excellent" answers

**Why bimodal at high end?**
- "Good" answer matches evaluator's template → consistent scoring
- "Excellent" answer exceeds template → uncertain whether to reward or penalize deviation

**Testable prediction**:
- Different questions (not Raft/Paxos) might show different patterns
- Sample 2 on a different question might be stable
- Sample 1 quality answer on different question might also be stable

**Evidence needed**:
- Test same samples on different questions
- Test different answer types (technical vs conceptual) at same quality level

---

## Hypothesis 8: LLM Attention and Context Length

**Claim**: Evaluator attention degrades over long answers, causing inconsistent focus.

**Mechanism**:
- **Short answers**: Evaluator reads completely, consistent judgment
- **Long answers**: Evaluator might focus on different parts in different runs
- Gemini's attention mechanism might non-deterministically emphasize different sections

**Why bimodal at high end?**
- When evaluator focuses on excellent sections: 10/10
- When evaluator focuses on "merely good" sections: 8/10
- Same answer, different focal points → bimodal

**Testable prediction**:
- Shorter versions of Sample 2 (same content, condensed) should be more stable
- Highlighting key sections might reduce variance

**Evidence needed**:
- Test condensed version of Sample 2
- Test with answer in different orders (move best part to beginning)

---

## Hypothesis 9: Temperature=0 is Not Truly Deterministic

**Claim**: Despite `temperature=0`, Gemini 2.5 Flash has inherent stochasticity.

**Mechanism**:
- **Theoretical temp=0**: Should always pick highest probability token
- **Reality**: API-level variations, reasoning tokens, or internal randomness

**Why bimodal at high end?**
- Stochasticity is bounded (not random walk)
- Model has two strong "attractors" in opinion space:
  - "This is good" (8/10)
  - "This is excellent" (10/10)
- Small internal variations push evaluation to one attractor or the other

**Testable prediction**:
- Different models (GPT-4o, Claude) might show different patterns
- Same model but different API endpoints might show variation
- Running 100+ times might reveal other states beyond bimodal

**Evidence needed**:
- Test with GPT-4o and Claude 3.5 Sonnet
- Run N=50 to see if other states emerge

---

## Most Likely Hypotheses (Ranked)

Based on the evidence we have:

### 1. **Hypothesis 2: Reasoning Token Stochasticity** ⭐⭐⭐
**Why**: Gemini 2.5 Flash uses reasoning tokens at `medium` effort. This is a known source of non-determinism.
**Evidence**: Sample 1 (similar complexity) is stable, suggesting it's not just complexity.
**Test**: Re-run with `reasoning_effort="low"`

### 2. **Hypothesis 5: Median Amplification Effect** ⭐⭐⭐
**Why**: Using median of 3 creates discrete states. Small judge disagreements become jumps.
**Evidence**: The bimodal pattern (78 vs 96) suggests 2-1 judge split.
**Test**: Re-calculate with mean instead of median

### 3. **Hypothesis 1: Ceiling Effect** ⭐⭐
**Why**: High-quality discrimination is inherently harder (subjective).
**Evidence**: Literature shows evaluators struggle at high quality.
**Test**: Test samples at 80-85 range

### 4. **Hypothesis 4: Hybrid Scoring Instability** ⭐⭐
**Why**: 60/40 weighting could create amplification.
**Evidence**: Need to check if rule-based scores vary.
**Test**: Re-run with pure LLM (no rules)

### 5. **Hypothesis 6: Prompt Ambiguity** ⭐
**Why**: Rubric might not define excellence clearly.
**Evidence**: Possible but would expect more variance at mid-range too.
**Test**: Add calibration examples

---

## Recommended Next Steps

1. **Quick Analysis** (No API calls):
   - Re-calculate Sample 2 stats using **mean** instead of median
   - Check if rule-based scores are constant across runs

2. **Targeted Experiment** ($0.06, 10 runs):
   - Test Sample 2 with `reasoning_effort="low"`
   - See if variance drops

3. **If still bimodal**: Test Hypothesis 5 by examining individual judge scores

Which hypothesis should we test first?
