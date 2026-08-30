# "Good Enough" Experiment Design

**Purpose:** Demonstrate that Agent Contracts enable agents to recognize "good enough" and stop voluntarily, optimizing for human benefit rather than engagement.

**Inspiration:** Ethan Flory's observation that AI should "tell you to stop after 5 minutes and send the email, that it's good enough and future improvements won't make a material difference."

---

## Core Thesis

**Current AI Problem:**
- AI keeps iterating, polishing, and "helping" indefinitely
- Optimizes for **engagement** (keep user engaged)
- User must manually decide when to stop

**Agent Contracts Solution:**
- Define **Q_min** (minimum quality threshold) in the contract
- Agent self-evaluates after each iteration
- Agent **voluntarily stops** when Q >= Q_min
- Optimizes for **human benefit** (save user time)

---

## Experiment Design

### Task: Professional Email Drafting

**Why email?**
1. Directly matches Ethan Flory's example
2. Universally relatable task
3. Quality is subjectively measurable
4. Clear "good enough" point exists
5. Iterations are visible and countable

### Email Scenarios

We'll create 20 diverse email scenarios across categories:

| Category | Example Scenario |
|----------|-----------------|
| **Meeting** | Reschedule a team meeting due to conflict |
| **Request** | Ask colleague for project status update |
| **Apology** | Apologize for missing a deadline |
| **Introduction** | Introduce yourself to new team member |
| **Follow-up** | Follow up on unanswered proposal |
| **Decline** | Politely decline a meeting invitation |
| **Thank You** | Thank someone for their help on a project |
| **Clarification** | Ask for clarification on requirements |

### Quality Criteria (Φ)

Each criterion is evaluated as binary (met/not met):

| Criterion | Weight | Description |
|-----------|--------|-------------|
| `clear_purpose` | 0.25 | Email clearly states its purpose in first 2 sentences |
| `professional_tone` | 0.20 | Language is professional, respectful, appropriate |
| `key_info_complete` | 0.25 | All necessary information is included |
| `appropriate_length` | 0.15 | Not too short (<50 words) or too long (>300 words) |
| `actionable` | 0.15 | Clear next step or call-to-action for recipient |

**Quality Threshold (θ):** 0.80 (must meet 80% of weighted criteria)

### Experiment Conditions

#### Condition 1: UNCONSTRACTED (Baseline)
- Agent drafts email
- User provides generic feedback: "Can you improve this?"
- Agent keeps refining
- Stops only when user says "That's fine" OR max iterations (10)
- **Simulates current AI behavior**

#### Condition 2: CONTRACTED (Q_min)
- Agent drafts email
- Agent self-evaluates against quality criteria
- If Q >= 0.80 → Agent responds: "This email meets quality standards. Ready to send?"
- If Q < 0.80 → Agent identifies gaps and refines
- Stops when Q >= Q_min OR max iterations (10)
- **Demonstrates Agent Contracts value**

### Metrics

| Metric | Description | Expected Difference |
|--------|-------------|-------------------|
| **Iterations to Stop** | Number of refinement rounds | CONTRACTED: 2-4, UNCONSTRACTED: 6-10 |
| **Tokens Used** | Total tokens consumed | CONTRACTED: 40-60% less |
| **Time to Completion** | Wall clock time | CONTRACTED: faster |
| **Final Quality Score** | Q at stopping point | Similar (both should be good) |
| **Early Stop Rate** | % that stopped before max | CONTRACTED: higher |
| **User Time Saved** | Simulated user interaction time | CONTRACTED: significant |

### Key Hypothesis

> **H1:** Contracted agents stop at significantly fewer iterations (p < 0.05) while achieving equivalent final quality.

> **H2:** The quality at stopping point is similar between conditions (both >= Q_min), but CONTRACTED reaches it faster.

---

## Implementation Architecture

```
evaluation/good_enough/
├── __init__.py
├── DESIGN.md                    # This file
├── scenarios.py                 # Email scenarios dataset
├── evaluator.py                 # EmailQualityEvaluator (LLM-as-judge)
├── agents.py                    # IterativeEmailAgent (with/without Q_min)
├── orchestrator.py              # Experiment orchestrator
├── run_experiment.py            # Main entry point
└── analyze_results.py           # Statistical analysis
```

### Component Details

#### 1. `scenarios.py` - Email Scenarios
```python
@dataclass
class EmailScenario:
    id: str
    category: str  # meeting, request, apology, etc.
    context: str  # Background situation
    recipient: str  # Who the email is to
    goal: str  # What the email should achieve
    key_info: list[str]  # Required information points
```

#### 2. `evaluator.py` - Quality Evaluation
```python
@dataclass
class EmailQualityCriteria:
    clear_purpose: bool  # Purpose stated in first 2 sentences
    professional_tone: bool  # Appropriate language
    key_info_complete: bool  # All required info present
    appropriate_length: bool  # 50-300 words
    actionable: bool  # Clear next step


class EmailQualityEvaluator:
    """LLM-as-judge for email quality."""

    def evaluate(self, email: str, scenario: EmailScenario) -> EmailQualityCriteria:
        """Evaluate email against quality criteria."""

    def compute_quality_score(self, criteria: EmailQualityCriteria) -> float:
        """Weighted sum of criteria (0-1 scale)."""

    def meets_threshold(self, criteria: EmailQualityCriteria, threshold: float) -> bool:
        """Check if Q >= threshold."""
```

#### 3. `agents.py` - Iterative Agents
```python
class IterativeEmailAgent:
    """Agent that iteratively refines emails."""

    def draft(self, scenario: EmailScenario) -> str:
        """Generate initial draft."""

    def refine(self, current_draft: str, feedback: str) -> str:
        """Refine draft based on feedback."""


class ContractedEmailAgent(IterativeEmailAgent):
    """Agent with Q_min stopping behavior."""

    def __init__(self, quality_threshold: float = 0.80):
        self.threshold = quality_threshold
        self.evaluator = EmailQualityEvaluator()

    def run_with_contract(self, scenario: EmailScenario) -> ContractedResult:
        """Run with self-evaluation and early stopping."""
        for iteration in range(max_iterations):
            draft = self.draft_or_refine(...)
            quality = self.evaluator.evaluate(draft, scenario)

            if quality.score >= self.threshold:
                return ContractedResult(
                    draft=draft,
                    iterations=iteration + 1,
                    stopped_early=True,
                    reason="Quality threshold met",
                )

        return ContractedResult(stopped_early=False, reason="Max iterations")
```

#### 4. `orchestrator.py` - Experiment Runner
```python
class GoodEnoughExperiment:
    """Main experiment orchestrator."""

    def run_unconstracted(self, scenario: EmailScenario) -> TrialResult:
        """Run without Q_min stopping (baseline)."""

    def run_contracted(self, scenario: EmailScenario) -> TrialResult:
        """Run with Q_min stopping (treatment)."""

    def run_experiment(self, n_scenarios: int = 20) -> ExperimentResults:
        """Run full experiment with both conditions."""
```

---

## Statistical Analysis

### Primary Analysis
- **Mann-Whitney U test**: Compare iterations between conditions
- **Effect size**: Cohen's d for iteration difference
- **Bootstrap CI**: 95% confidence intervals for all metrics

### Secondary Analysis
- **Quality equivalence test**: Confirm final quality is similar
- **Cost-benefit analysis**: Tokens saved vs quality tradeoff
- **Per-category analysis**: Do some email types benefit more?

---

## Expected Results

Based on the thesis, we expect:

| Metric | UNCONSTRACTED | CONTRACTED | Difference |
|--------|---------------|------------|------------|
| Avg Iterations | 6-8 | 2-4 | **50-60% fewer** |
| Avg Tokens | ~8000 | ~3500 | **~55% less** |
| Final Quality | 0.85-0.90 | 0.82-0.88 | Similar |
| Early Stop % | 20% | 80% | **4x higher** |

### Key Finding for Paper

> "Agent Contracts enable autonomous agents to recognize 'good enough' and stop voluntarily. In our email drafting experiment, contracted agents required 50-60% fewer iterations while achieving equivalent quality, demonstrating optimization for human benefit rather than engagement."

---

## Integration with COINE 2026 Paper

This experiment directly supports:

1. **Simon's Satisficing Principle (§2)**: Agents work within defined quality thresholds
2. **Success Criteria Φ (§3)**: Q_min operationalizes the paper's formal success criteria
3. **Termination Conditions Ψ (§3)**: "Quality threshold met" as explicit termination event
4. **Governance Value (§5)**: Contracts align agent behavior with human interests

### Narrative for Paper

> "Current LLM agents are designed to maximize helpfulness, leading to unbounded iteration that consumes user time without proportional benefit. Agent Contracts address this by defining explicit quality thresholds (Q_min) that agents evaluate against, enabling them to recognize when 'good enough' is reached and voluntarily terminate. This operationalizes Simon's satisficing principle: agents achieve acceptable quality within resource bounds, optimizing for actual human benefit rather than engagement metrics."

---

## Implementation Timeline

1. **Day 1**: Create scenarios dataset + quality evaluator
2. **Day 2**: Implement iterative agents (both conditions)
3. **Day 3**: Build experiment orchestrator + runner
4. **Day 4**: Run experiments (n=20 scenarios × 2 conditions)
5. **Day 5**: Analyze results + create visualizations

---

*Design Date: December 31, 2025*
*Target: COINE 2026*
