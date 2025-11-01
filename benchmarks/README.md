# Agent Contracts Benchmarks

This directory contains comprehensive benchmarks and performance validation for the Agent Contracts framework.

## Phase 1 Benchmarks

### 1. Infrastructure Benchmark (`benchmark_phase1.py`)

Tests the contract enforcement infrastructure:

```bash
# Ensure you have your GOOGLE_API_KEY in .env
uv run python benchmarks/benchmark_phase1.py
```

### 2. Value Demonstration Benchmark (`benchmark_phase1_qa.py`)

Demonstrates efficiency gains through strategic prioritization:

```bash
uv run python benchmarks/benchmark_phase1_qa.py
```

### What It Tests

The Phase 1 benchmark provides a comprehensive performance comparison:

1. **Baseline Test** - Raw litellm calls without contract wrapper (establishes baseline)
2. **Contracted LLM - No Limits** - Measures pure wrapper overhead
3. **Strict Enforcement** - Validates hard budget limits with enforcement
4. **Lenient Monitoring** - Validates soft limits with warnings only

### Metrics Tracked

**Performance Metrics:**
- Total execution time
- Average time per API call
- Individual call latencies
- Contract enforcement overhead

**Accuracy Metrics:**
- Token counting accuracy
- API call tracking
- Cost estimation

**Functional Metrics:**
- Budget enforcement correctness
- Event callback performance
- Contract state transitions
- Violation detection

### Benchmark Results (Gemini 2.5 Flash Preview 09-2025)

**Performance Overhead:** ~-12.1% average (within network variance, actually faster)
- Baseline: 1.044s per call
- Contracted (no limits): 0.887s per call (-15.0%)
- Strict enforcement: 0.972s per call (-6.9%)
- Lenient monitoring: 0.895s per call (-14.3%)

**Token Tracking Accuracy:** 100% (322 tokens tracked correctly)

**Enforcement Validation:**
- ✅ Strict mode: Stopped at exactly 2 calls (limit: 2)
- ✅ Lenient mode: Detected violations (1), allowed continuation
- ✅ Event callbacks: 4-7 events per test, all firing correctly
- ✅ API key loading: python-dotenv successfully loads .env

**Reasoning Token Support:**
- ✅ Framework supports separate reasoning/text token tracking
- ✅ Handles both lumpsum (total only) and separate budgets
- ℹ️ Gemini 2.5 Flash didn't return reasoning token breakdown for these simple queries (may require more complex reasoning tasks)
- ✅ Framework gracefully handles models with and without reasoning token details
- ✅ Infrastructure ready for o1, Claude extended thinking, and future reasoning models

**Key Finding:** Contract overhead is **negligible** and within normal network variance. The framework adds comprehensive monitoring and enforcement with effectively **zero performance penalty**.

### Requirements

- Python 3.12+
- GOOGLE_API_KEY environment variable (for Gemini access)
- Dependencies installed via `uv sync`

### Output

The benchmark provides:
- Comprehensive performance comparison
- Detailed latency breakdown per API call
- Token tracking accuracy validation
- Budget enforcement validation
- Event callback performance metrics
- Side-by-side comparison of all modes

## Phase 1 QA Benchmark Results

### Key Learnings

**Reasoning Model Support**:
- ✅ Successfully handles Gemini 2.5 Flash's reasoning tokens
- ✅ Framework separates internal reasoning from text output tokens
- ✅ Properly tracks both reasoning_tokens and text_tokens
- ⚠️ Reasoning models need generous max_tokens for both thinking + output phases

**Benchmark Design Insights**:
- Task difficulty must be calibrated to model capabilities
- Current QA task (15 questions, quantum computing) is easily solved with 2000 tokens
- Both baseline and budget-aware strategies achieve 100% accuracy
- Need tighter constraints or more complex tasks to demonstrate ≥30% efficiency gains

**Infrastructure Validation**:
- ✅ ContractedLLM wrapper works correctly with reasoning models
- ✅ Answer parsing and evaluation pipeline functional
- ✅ Weighted accuracy metrics implemented correctly
- ✅ Quality-per-token efficiency calculation works

**Future Improvements**:
- Design tasks that force trade-offs (e.g., summarization length vs. detail)
- Test with non-reasoning models (Claude, GPT-4) for comparison
- Implement dynamic budget allocation based on question importance
- Add multi-turn conversation scenarios

## Future Benchmarks

Additional benchmarks will be added as we progress through the roadmap:

- **Phase 2**: Quality metrics benchmarks, skill verification tests
- **Phase 3**: Framework integration benchmarks (LangChain, AutoGen, CrewAI)
- **Phase 4**: Scalability tests, production performance benchmarks
