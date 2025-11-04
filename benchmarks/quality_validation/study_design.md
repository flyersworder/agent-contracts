# Quality Validation Study: Experimental Design

**Date**: November 4, 2025
**Purpose**: Validate current quality evaluator reliability and cost-effectiveness
**Approach**: Progressive validation with cost optimization

---

## Cost Analysis

### Current Evaluator Configuration
- **Model**: `gemini/gemini-2.5-flash-preview-09-2025`
- **Reasoning effort**: `"medium"`
- **Temperature**: 0

### Cost Estimates (Gemini 2.5 Flash)

**Per Evaluation Breakdown**:
- Input tokens (prompt): ~600-800 tokens (question + answer + rubric)
- Output tokens (scores): ~150-200 tokens
- Reasoning tokens (medium): ~1,000-2,000 tokens
- **Total per evaluation**: ~1,800-3,000 tokens

**Pricing** (Gemini 2.5 Flash with reasoning):
- Input: $0.075 per 1M tokens
- Output: $0.30 per 1M tokens
- Reasoning: $0.30 per 1M tokens (charged as output)

**Cost per evaluation**: ~$0.0006-0.0010 (approximately **$0.001**)

### Full Study Cost Estimates

| Phase | Evaluations | Estimated Cost | Value |
|-------|-------------|----------------|-------|
| Phase 1: Test-Retest (Original) | 50 (5 answers × 10 runs) | $0.05-0.08 | Baseline CV |
| Phase 2: Inter-Rater (Original) | 45 (3 models × 5 answers × 3 runs) | $0.50-1.00 | Multi-model agreement |
| Phase 3: Human Study | 20 evaluations | $0 (human time) | Ground truth |
| **Total (Original Plan)** | **95 evaluations** | **$0.55-1.08** | **Full validation** |

### Cost-Optimized Alternative

| Phase | Evaluations | Estimated Cost | Value |
|-------|-------------|----------------|-------|
| Phase 1: Test-Retest (Reduced) | 25 (5 answers × 5 runs) | $0.025-0.040 | Baseline CV |
| Phase 2: Inter-Rater (Gemini-only) | 15 (3 Gemini models × 5 answers × 1 run) | $0.015-0.025 | Same-family agreement |
| Phase 3: Human Subset | 10 evaluations | $0 (human time) | Ground truth sample |
| **Total (Optimized)** | **50 evaluations** | **$0.04-0.065** | **Core validation** |

**Savings**: ~94% cost reduction ($1.08 → $0.065) while maintaining statistical validity

---

## Experimental Design: Progressive Validation

### Design Principle: **Start Small, Scale If Needed**

Instead of running all experiments upfront, use a **progressive validation** approach:

1. **Quick pilot** (N=3) → Check feasibility
2. **Core validation** (N=5) → Establish baseline
3. **Extended validation** (if needed) → Deep analysis

This approach minimizes wasted cost if early results are conclusive.

---

## Phase 1: Test-Retest Reliability

### Research Question
**"Is our current evaluator's variance (CV: 10-11%) due to evaluator inconsistency or actual agent variance?"**

### Hypotheses
- **H0**: Evaluator CV on fixed answers is <5% (evaluator is reliable)
- **H1**: Evaluator CV on fixed answers is >10% (evaluator is unreliable)

### Sample Selection Strategy

**Goal**: 5 diverse answers spanning quality and difficulty

**Selection criteria**:
1. **Quality distribution**: 1 excellent (90+), 2 good (75-85), 1 fair (60-70), 1 poor (<60)
2. **Length distribution**: 1 short (<300 words), 2 medium (300-600), 2 long (>600)
3. **Question difficulty**: Mix of easy (q5_microservices) and hard (q1_transformers)
4. **Existing data**: Use answers from completed benchmarks (no new API calls for answers)

**Why 5 answers?**
- Literature: N≥5 samples adequate for reliability studies
- Statistical power: 5 × 5 runs = 25 data points (sufficient for CV estimation)
- Diversity: Captures variation across quality levels

### Experimental Protocol

#### **Pilot Run** (N=3 runs per answer, 5 answers = 15 evaluations)
**Purpose**: Check feasibility, estimate actual CV
**Cost**: ~$0.015
**Time**: ~5 minutes
**Decision rule**:
- If CV < 5%: Evaluator highly reliable, stop here ✅
- If CV 5-8%: Good reliability, run N=2 more for confirmation
- If CV > 8%: Need full N=10 for deeper analysis

#### **Core Run** (N=5 runs per answer, 5 answers = 25 evaluations)
**Purpose**: Establish reliable baseline CV
**Cost**: ~$0.025
**Time**: ~10 minutes
**Statistical power**:
- Standard error of CV: SE(CV) ≈ CV/√(2N) = CV/√10 ≈ 0.32·CV
- For CV=10%, SE ≈ 3.2% (acceptable precision)

#### **Extended Run** (Optional: N=10 runs per answer)
**Only if**: Pilot shows unexpected results or high variance
**Cost**: Additional ~$0.025
**Time**: Additional ~10 minutes

### Metrics to Calculate

1. **Coefficient of Variation (CV)** per answer:
   - CV = (std_dev / mean) × 100%
   - Target: CV < 5% (reliable), 5-8% (acceptable), >8% (concerning)

2. **Intraclass Correlation Coefficient (ICC)**:
   - Measures consistency of repeated measurements
   - ICC(2,1): Single rater, absolute agreement
   - Target: ICC > 0.85 (good), > 0.90 (excellent)

3. **95% Confidence Intervals**:
   - For each quality score
   - Width indicates measurement precision
   - Narrow CI (±2-3 points) = reliable, Wide CI (±5+ points) = unreliable

4. **Range Analysis**:
   - Max - Min score for each answer
   - Target: Range < 10 points (reliable)

### Success Criteria

✅ **PASS**:
- Average CV < 8% across all answers
- ICC > 0.85
- 95% CI width < ±3 points
- **Conclusion**: Current evaluator is reliable

⚠️ **MARGINAL**:
- Average CV 8-12%
- ICC 0.75-0.85
- **Conclusion**: Adequate but could improve

❌ **FAIL**:
- Average CV > 12%
- ICC < 0.75
- **Conclusion**: Need improvements (Phase 2 required)

---

## Phase 2: Inter-Rater Reliability (Conditional)

### Trigger Condition
**Only run if Phase 1 shows CV > 8% OR if we want to compare models**

### Cost-Optimized Design

#### **Option A: Same-Family Comparison** (Cheapest)
**Models**:
- Gemini 2.5 Flash (current)
- Gemini 2.0 Flash Exp
- Gemini 1.5 Pro

**Cost**: ~$0.015 (all Gemini models similarly priced)
**Purpose**: Check if reasoning_effort level matters more than model version

#### **Option B: Cross-Family Comparison** (More expensive but valuable)
**Models**:
- Gemini 2.5 Flash: $0.0006/eval
- GPT-4o-mini: $0.0015/eval (2.5× more)
- Claude 3.5 Haiku: $0.0008/eval

**Cost**: ~$0.045 (3 models × 5 answers × 3 runs)
**Purpose**: Check agreement across different model families

### Sample Size Optimization

**Original plan**: 3 runs per model × 5 answers = 45 evaluations
**Optimized**: 1 run per model × 5 answers × 3 repetitions = 15 evaluations for Option A

**Rationale**:
- We already know single-model variance from Phase 1
- Inter-rater reliability needs fewer samples (checking agreement, not variance)
- Can always scale up if initial results are inconclusive

### Metrics

1. **Krippendorff's Alpha (α)**:
   - Measures agreement across multiple raters
   - α > 0.80: Acceptable
   - α > 0.90: Excellent

2. **Pairwise Correlations**:
   - Gemini ↔ GPT-4o-mini
   - Gemini ↔ Claude
   - GPT-4o-mini ↔ Claude
   - Target: r > 0.85

3. **Systematic Bias Detection**:
   - Does any judge consistently score higher/lower?
   - Bland-Altman plot for bias visualization

---

## Phase 3: Human Correlation Study (Optional)

### Trigger Condition
**Only run if**:
- Phase 1 or 2 shows unexpected results
- We need external validation for publication/credibility
- CV is acceptable but we want to know if it's measuring the right thing

### Cost-Optimized Design

**Original plan**: 20 answers (4 questions × 5 answers)
**Optimized**: 10 answers (2 questions × 5 answers)

**Rationale**:
- Human time is the real cost (2-3 hours of expert review)
- N=10 sufficient for correlation study (r calculation requires N≥8)
- Focus on most representative questions

### Sample Selection
- **Question 1**: Hard question (q1_transformers) - 5 answers spanning quality
- **Question 2**: Medium question (q3_financial) - 5 answers spanning quality

### Human Evaluation Protocol

**Rubric** (same dimensions as LLM judge):
1. Accuracy (0-10): Factual correctness
2. Completeness (0-10): Coverage of aspects
3. Coherence (0-10): Structure and clarity
4. Total: (sum / 30) × 100

**Expert qualifications**:
- PhD or 5+ years experience in relevant domain
- Familiar with evaluation rubrics
- Not involved in agent development (blind evaluation)

### Metrics

1. **Pearson Correlation (r)**:
   - Between human total scores and LLM total scores
   - Target: r > 0.70 (SOTA level)

2. **Per-Dimension Correlation**:
   - Accuracy, Completeness, Coherence separately
   - Identifies which dimensions align best with human judgment

3. **Mean Absolute Error (MAE)**:
   - |Human score - LLM score|
   - Target: MAE < 10 points (on 0-100 scale)

---

## Progressive Validation Decision Tree

```
START
  ↓
Phase 1: Test-Retest (Pilot, N=3)
  ↓
  Is CV < 5%? ──YES→ ✅ STOP (Evaluator highly reliable)
  ↓ NO
  Is CV < 8%? ──YES→ Run N=2 more for confirmation → ✅ STOP (Acceptable)
  ↓ NO
  CV > 8%? ──YES→ Run full N=5 for baseline
  ↓
  Is CV < 10%? ──YES→ Acceptable, document as limitation
  ↓ NO
  CV > 10%? ──YES→ Phase 2: Inter-Rater Reliability
  ↓
Phase 2: Inter-Rater (Option A: Gemini family)
  ↓
  Is α > 0.80? ──YES→ Problem is model variance, not our implementation
  ↓ NO
  α < 0.80? ──YES→ Phase 2B: Try cross-family (GPT-4, Claude)
  ↓
Phase 2B: Cross-Family Comparison
  ↓
  Is α > 0.80? ──YES→ ✅ Multi-model ensemble recommended
  ↓ NO
  α < 0.80? ──YES→ ⚠️ LLM-as-judge may not be reliable for this task
  ↓
Phase 3: Human Correlation Study
  ↓
  Is r > 0.70? ──YES→ ✅ LLM judges align with humans
  ↓ NO
  r < 0.70? ──YES→ ❌ Need fundamental rethinking of evaluation approach
```

---

## Implementation Plan

### Step 1: Sample Selection (10 minutes)
- Extract 5 diverse answers from existing benchmark results
- No new API calls needed
- Cost: $0

### Step 2: Pilot Test-Retest (5 minutes)
- 3 runs × 5 answers = 15 evaluations
- Cost: ~$0.015
- Decision point: Proceed or stop?

### Step 3: Core Test-Retest (10 minutes, if needed)
- 2 additional runs × 5 answers = 10 evaluations
- Cost: ~$0.010
- Total Phase 1: 25 evaluations, ~$0.025

### Step 4: Analysis (15 minutes)
- Calculate CV, ICC, confidence intervals
- Visualize results
- Decision: Pass/Marginal/Fail?

### Step 5: Phase 2 (conditional, 15 minutes)
- Only if Phase 1 shows issues
- Start with Option A (Gemini family): ~$0.015
- Escalate to Option B if needed: ~$0.030 additional

### Step 6: Documentation (30 minutes)
- Write up findings
- Recommendations for improvements
- Update quality framework docs

**Total estimated time**: 1.5-2 hours
**Total estimated cost**: $0.025-0.070 (depending on path taken)
**Maximum cost** (if all phases needed): ~$0.10

---

## Data Collection & Storage

### File Structure
```
benchmarks/quality_validation/
├── study_design.md                    # This file
├── samples/
│   ├── sample_1_excellent.json        # High quality answer
│   ├── sample_2_good.json             # Good answer
│   ├── sample_3_good.json             # Good answer
│   ├── sample_4_fair.json             # Fair answer
│   └── sample_5_poor.json             # Poor answer
├── results/
│   ├── phase1_pilot_results.json      # Pilot test-retest (N=3)
│   ├── phase1_core_results.json       # Core test-retest (N=5)
│   ├── phase2_interrater_results.json # Inter-rater (if needed)
│   └── phase3_human_results.json      # Human correlation (if needed)
├── analysis/
│   ├── test_retest_analysis.py        # CV, ICC calculations
│   ├── interrater_analysis.py         # Krippendorff's alpha
│   └── visualization.py               # Plots and charts
└── report.md                          # Final validation report
```

### Data Format (JSON)

```json
{
  "study_phase": "phase1_pilot",
  "timestamp": "2025-11-04T12:00:00Z",
  "sample_id": "sample_1_excellent",
  "question": "What is...",
  "answer": "The answer is...",
  "runs": [
    {
      "run_id": 1,
      "accuracy": 9.5,
      "completeness": 9.0,
      "coherence": 9.5,
      "total": 93.3,
      "explanation": "...",
      "model": "gemini/gemini-2.5-flash-preview-09-2025",
      "reasoning_effort": "medium",
      "tokens": {"input": 650, "output": 180, "reasoning": 1200}
    }
  ],
  "statistics": {
    "mean": 93.2,
    "std_dev": 1.5,
    "cv": 1.6,
    "min": 91.0,
    "max": 95.0,
    "ci_95_lower": 91.5,
    "ci_95_upper": 94.9
  }
}
```

---

## Risk Mitigation

### Risk 1: API Rate Limits
**Mitigation**: Add 1-second delay between evaluations

### Risk 2: Cost Overrun
**Mitigation**: Progressive approach with decision points, hard cap at $0.20

### Risk 3: Inconclusive Results
**Mitigation**: Clear success criteria and decision tree

### Risk 4: Time Overrun
**Mitigation**: Automated scripts, parallel where possible

---

## Expected Outcomes

### Best Case (CV < 5%)
- **Conclusion**: Current evaluator is highly reliable
- **Action**: Document reliability, proceed to Phase 2B (production features)
- **Cost**: ~$0.015
- **Time**: ~30 minutes

### Good Case (CV 5-8%)
- **Conclusion**: Current evaluator is acceptably reliable
- **Action**: Add confidence intervals, document limitation
- **Cost**: ~$0.025
- **Time**: ~1 hour

### Concerning Case (CV > 8%)
- **Conclusion**: Need improvements
- **Action**: Run Phase 2 (inter-rater), implement enhancements
- **Cost**: ~$0.040-0.070
- **Time**: ~2 hours

### Worst Case (CV > 12%, α < 0.75)
- **Conclusion**: LLM-as-judge may not be reliable for this task
- **Action**: Fundamental rethinking (rule-based? human-in-loop? different models?)
- **Cost**: ~$0.10 (full validation)
- **Time**: ~3 hours + design time

---

## Next Steps

1. **Review this design** - Get feedback on protocol
2. **Select samples** - Extract 5 diverse answers from existing data
3. **Implement test harness** - Script for automated evaluation
4. **Run pilot** - 15 evaluations (~5 minutes)
5. **Analyze & decide** - Proceed or stop?

**Ready to proceed?**
