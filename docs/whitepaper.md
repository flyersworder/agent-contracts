# Agent Contracts: A Resource-Bounded Optimization Framework for Autonomous AI Systems

**Abstract**

We introduce *Agent Contracts*, a formal framework for governing autonomous AI agents through explicit resource constraints and temporal boundaries. Drawing from contract theory, optimization theory, and resource allocation principles, we propose a multi-dimensional constraint system that transforms agent behavior from open-ended execution to bounded optimization. By defining contracts across computational resources (tokens, API calls), temporal constraints (deadlines, duration), and output specifications, we enable predictable, governable, and cost-effective agentic systems. We demonstrate how time-resource tradeoffs create a rich design space for agent specialization and show how contract-based coordination enables efficient multi-agent collaboration. This framework addresses critical gaps in current agentic systems: lifecycle management, resource governance, and formal verification of agent behavior.

---

## 1. Introduction

### 1.1 The Governance Gap in Agentic AI

The emergence of Large Language Model (LLM)-based agents has introduced unprecedented autonomy in AI systems. Agents can now:
- Use tools and APIs through function calling
- Search the web and retrieve information
- Generate structured outputs
- Engage in multi-step reasoning and planning
- Coordinate with other agents in multi-agent systems

However, this autonomy creates significant challenges for production deployment:

**Resource Management**: Agents can consume unbounded computational resources (tokens, API calls, compute time) without explicit constraints, leading to unpredictable costs.

**Lifecycle Ambiguity**: Unlike traditional software with clear start/stop conditions, agents often lack explicit termination criteria, leading to resource leaks and unclear accountability.

**Governance and Compliance**: Organizations struggle to audit agent behavior, ensure compliance, and attribute costs when agent boundaries are implicit rather than explicit.

**Coordination Complexity**: Multi-agent systems lack formal mechanisms for resource allocation, task distribution, and conflict resolution.

### 1.2 From Implicit Constraints to Explicit Contracts

Current approaches to agent definition typically include:
- **System prompts**: Natural language instructions
- **Tool specifications**: Available functions and their schemas
- **Output formats**: Structured output schemas (JSON, XML)
- **Skills/capabilities**: Declarative descriptions of agent abilities

While these define *what* an agent can do, they fail to formalize:
- *How much* of each resource the agent may consume
- *How long* the agent has to complete its task
- *What tradeoffs* the agent should make under resource pressure
- *When* the agent's contract expires and resources should be released

**Agent Contracts** bridge this gap by providing a formal, executable specification that transforms agent operation from unbounded exploration to constrained optimization.

### 1.3 Theoretical Foundations

Agent Contracts draw from multiple theoretical traditions:

**Contract Theory** (Economics): Formal agreements that align incentives and define obligations, rights, and remedies between parties.

**Resource Allocation Theory** (Operations Research): Optimization under constraints, where scarce resources must be distributed among competing demands.

**Real-Time Systems** (Computer Science): Systems with explicit timing constraints where correctness depends not just on logical results but on meeting deadlines.

**Linear/Dynamic Programming** (Optimization Theory): Finding optimal solutions within feasible regions defined by constraint boundaries.

---

## 2. Formal Framework

### 2.1 Agent Contract Definition

An **Agent Contract** $C$ is a tuple:

$$C = (I, O, S, R, T, \Phi, \Psi)$$

Where:

- $I$: **Input specification** - Schema and constraints for acceptable inputs
- $O$: **Output specification** - Schema and quality criteria for outputs
- $S$: **Skills** - Set of capabilities (tools, functions, knowledge domains)
- $R$: **Resource constraints** - Multi-dimensional resource budget
- $T$: **Temporal constraints** - Time-related boundaries and deadlines
- $\Phi$: **Success criteria** - Measurable conditions for contract fulfillment
- $\Psi$: **Termination conditions** - Events that end the contract

### 2.2 Resource Constraint Space

The resource constraint $R$ defines a multi-dimensional budget:

$$R = \{r_1, r_2, ..., r_n\}$$

Common resource dimensions include:

| Resource | Symbol | Unit | Example Constraint |
|----------|--------|------|-------------------|
| LLM Tokens | $r_{tok}$ | tokens | 100,000 tokens |
| API Calls | $r_{api}$ | calls | 50 calls |
| Web Searches | $r_{web}$ | queries | 10 queries |
| Tool Invocations | $r_{tool}$ | invocations | 30 invocations |
| Memory/Storage | $r_{mem}$ | MB | 500 MB |
| Compute Time | $r_{cpu}$ | CPU-seconds | 300 seconds |
| External Cost | $r_{cost}$ | USD | $5.00 |

Each resource $r_i$ has:
- **Budget** $b_i$: Total allocation
- **Consumption** $c_i(t)$: Amount used at time $t$
- **Rate limit** $\dot{r}_i$: Maximum consumption per time unit

**Constraint satisfaction** requires:
$$\forall i: c_i(t) \leq b_i \text{ and } \frac{dc_i}{dt} \leq \dot{r}_i$$

### 2.3 Temporal Constraint Space

The temporal constraint $T$ defines time-related boundaries:

$$T = (t_{start}, t_{deadline}, \Delta t_{max}, \tau)$$

Where:
- $t_{start}$: Contract activation time
- $t_{deadline}$: Hard deadline for completion
- $\Delta t_{max}$: Maximum elapsed time (wall-clock)
- $\tau$: Contract duration/expiration (lifecycle limit)

**Temporal constraint types:**

**Hard Deadline**: Task must complete by $t_{deadline}$
$$t_{completion} \leq t_{deadline}$$

**Soft Deadline**: Task should complete by $t_{soft}$ with quality degradation after
$$Q(t) = \begin{cases}
Q_{max} & \text{if } t \leq t_{soft} \\
Q_{max} \cdot e^{-\lambda(t-t_{soft})} & \text{if } t > t_{soft}
\end{cases}$$

**Time Budget**: Maximum computation time regardless of wall-clock
$$\int_{t_{start}}^{t_{completion}} \mathbb{1}_{computing}(t) \, dt \leq \Delta t_{max}$$

**Contract Expiration**: Agent lifecycle terminates after duration $\tau$
$$t_{current} - t_{start} \geq \tau \implies \text{terminate contract}$$

### 2.4 The Time-Resource Tradeoff Surface

A fundamental property of Agent Contracts is the **time-resource tradeoff**, which creates a multi-dimensional feasible region for optimization.

**Tradeoff Function**: Given task complexity $\mathcal{C}$ and quality requirement $Q_{target}$, there exists a tradeoff surface:

$$f(t, r_1, r_2, ..., r_n) \geq \mathcal{C} \cdot Q_{target}$$

Where $t$ is time and $r_i$ are resource allocations.

**Key Insights:**

1. **Time-Token Tradeoff**: More time allows for sequential processing with lower token usage
   - Parallel processing: High tokens/API calls, low time
   - Sequential processing: Low tokens/API calls, high time

2. **Quality-Resource Tradeoff**: Higher quality requires more resources or time
   - Quick approximation: Low resources, low time, acceptable quality
   - Thorough analysis: High resources, high time, high quality

3. **Urgency Modes**:

**Urgent Mode** ($t$ constrained, $R$ relaxed):
```
max Q(t, R)
subject to: t ≤ t_deadline
            R ≤ R_max (relaxed)
```

**Economical Mode** ($R$ constrained, $t$ relaxed):
```
max Q(t, R)
subject to: R ≤ R_budget (strict)
            t ≤ t_max (relaxed)
```

**Balanced Mode** (both constrained):
```
max Q(t, R)
subject to: t ≤ t_deadline
            R ≤ R_budget
            α·t/t_deadline + β·R/R_budget ≤ 1
```

### 2.5 Agent Optimization Problem

Under a contract $C$, an agent solves:

$$\max_{a \in A} U(a, C)$$

Subject to:
$$\begin{align}
&c_i(t) \leq b_i & \forall i \in R \\
&t_{completion} \leq t_{deadline} \\
&O(a) \in \Phi \\
&a \text{ uses only skills in } S
\end{align}$$

Where:
- $A$: Set of possible action sequences
- $U(a, C)$: Utility function (task quality, completeness, accuracy)
- $O(a)$: Output produced by action sequence $a$

This transforms the agent from a general problem-solver into a **bounded optimizer** with clear objectives and constraints.

---

## 3. Contract Lifecycle and Dynamics

### 3.1 Contract States

An agent contract progresses through distinct states:

```
┌─────────────┐
│   DRAFTED   │ - Contract defined but not active
└──────┬──────┘
       │
       ↓
┌─────────────┐
│   ACTIVE    │ - Agent executing within constraints
└──────┬──────┘
       │
       ├────→ ┌─────────────┐
       │      │  FULFILLED  │ - Success criteria met
       │      └─────────────┘
       │
       ├────→ ┌─────────────┐
       │      │   VIOLATED  │ - Constraints breached
       │      └─────────────┘
       │
       ├────→ ┌─────────────┐
       │      │   EXPIRED   │ - Time limit reached
       │      └─────────────┘
       │
       └────→ ┌─────────────┐
              │  TERMINATED │ - External cancellation
              └─────────────┘
```

### 3.2 Runtime Monitoring

During execution, the contract enforcement system monitors:

**Resource Consumption Tracking**:
```python
class ResourceMonitor:
    def __init__(self, contract: Contract):
        self.budget = contract.R
        self.consumed = {r: 0 for r in contract.R}
        self.start_time = time.time()

    def consume(self, resource: str, amount: float):
        self.consumed[resource] += amount
        if self.consumed[resource] > self.budget[resource]:
            raise ContractViolation(f"{resource} budget exceeded")

    def get_remaining(self, resource: str) -> float:
        return self.budget[resource] - self.consumed[resource]

    def get_utilization(self) -> dict:
        return {r: self.consumed[r] / self.budget[r]
                for r in self.budget}
```

**Temporal Compliance Checking**:
```python
class TemporalMonitor:
    def __init__(self, contract: Contract):
        self.deadline = contract.T.deadline
        self.max_duration = contract.T.max_duration
        self.start_time = time.time()

    def check_deadline(self) -> bool:
        if time.time() > self.deadline:
            raise DeadlineViolation()
        return True

    def get_time_pressure(self) -> float:
        elapsed = time.time() - self.start_time
        remaining = self.deadline - time.time()
        return elapsed / (elapsed + remaining)
```

### 3.3 Dynamic Resource Allocation

Agents can make runtime decisions based on budget state:

**Budget-Aware Planning**:
```python
class BudgetAwareAgent:
    def plan_action(self, task, remaining_budget):
        # Calculate expected resource usage for strategies
        strategies = [
            {
                "name": "thorough_web_search",
                "quality": 0.95,
                "cost": {"tokens": 30000, "web": 8, "time": 45}
            },
            {
                "name": "cached_knowledge",
                "quality": 0.75,
                "cost": {"tokens": 5000, "web": 0, "time": 10}
            },
            {
                "name": "hybrid_approach",
                "quality": 0.85,
                "cost": {"tokens": 15000, "web": 3, "time": 25}
            }
        ]

        # Filter feasible strategies
        feasible = [s for s in strategies
                    if all(s["cost"][r] <= remaining_budget[r]
                           for r in s["cost"])]

        # Select highest quality feasible strategy
        if feasible:
            return max(feasible, key=lambda s: s["quality"])
        else:
            return self.fallback_strategy(remaining_budget)
```

**Adaptive Quality-Resource Tradeoff**:
```python
def adaptive_strategy(self, budget_utilization: float):
    """Adjust strategy based on budget consumption rate"""

    if budget_utilization < 0.3:
        # Plenty of budget remaining
        return "high_quality_mode"
    elif budget_utilization < 0.7:
        # Moderate budget usage
        return "balanced_mode"
    else:
        # Budget nearly exhausted
        return "conservation_mode"
```

### 3.4 Contract Renegotiation

Contracts can be modified during execution:

**Extension Request**:
```python
class ContractNegotiation:
    def request_extension(self, agent_id: str,
                         additional_resources: dict,
                         justification: str) -> bool:
        """
        Agent requests additional resources mid-execution
        """
        current_contract = self.get_contract(agent_id)
        current_progress = self.assess_progress(agent_id)

        # Decision logic
        if current_progress > 0.8 and \
           additional_resources["tokens"] < 20000:
            # Approve modest extension for near-complete task
            self.extend_contract(agent_id, additional_resources)
            return True
        else:
            # Reject and suggest task decomposition
            return False
```

**Automatic Degradation**:
```python
def graceful_degradation(self, remaining_budget: dict):
    """Automatically adjust quality targets as budget depletes"""

    if remaining_budget["tokens"] < 10000:
        # Switch to smaller, more efficient model
        self.model = "small-model"  # e.g., gpt-4o-mini, claude-haiku
        self.reduce_context_window(max_tokens=4000)

    if remaining_budget["web"] == 0:
        # Disable web search, use only cached knowledge
        self.disable_tool("web_search")

    # Return partial results with confidence scores
    return self.compile_results(mark_incomplete=True)
```

---

## 4. Multi-Agent Coordination

### 4.1 Hierarchical Contract Structure

Complex tasks can be decomposed into hierarchical contract structures:

```
ParentContract: 200K tokens, 100 API calls, 60 min
├─ SubContract_A (Research): 80K tokens, 40 API, 25 min
│  ├─ SubContract_A1 (Web Search): 30K tokens, 10 API, 10 min
│  └─ SubContract_A2 (Analysis): 50K tokens, 30 API, 15 min
├─ SubContract_B (Synthesis): 60K tokens, 20 API, 20 min
├─ SubContract_C (Verification): 40K tokens, 30 API, 10 min
└─ Reserve Buffer: 20K tokens, 10 API, 5 min
```

**Parent Agent Responsibilities**:
- Decompose task into subtasks
- Allocate resource budgets to child contracts
- Monitor child contract progress
- Reallocate resources from completed/failed contracts
- Synthesize results from child agents

**Allocation Strategy**:
```python
def allocate_resources(self, subtasks: list,
                      total_budget: dict) -> list:
    """
    Allocate parent budget across child contracts
    """
    # Reserve buffer (10% of budget)
    buffer = {r: total_budget[r] * 0.1 for r in total_budget}
    available = {r: total_budget[r] - buffer[r] for r in total_budget}

    # Allocate proportionally by estimated complexity
    total_complexity = sum(t.complexity for t in subtasks)

    allocations = []
    for task in subtasks:
        proportion = task.complexity / total_complexity
        allocation = {r: available[r] * proportion
                     for r in available}
        allocations.append(allocation)

    return allocations
```

### 4.2 Resource Markets and Trading

In systems with multiple concurrent agents, contracts enable resource trading:

**Resource Exchange Protocol**:
```python
class ResourceMarket:
    def __init__(self):
        self.offers = []
        self.demands = []

    def offer_resources(self, agent_id: str,
                       resource: str, amount: float):
        """Agent offers unused resources"""
        self.offers.append({
            "agent": agent_id,
            "resource": resource,
            "amount": amount,
            "timestamp": time.time()
        })

    def request_resources(self, agent_id: str,
                         resource: str, amount: float,
                         max_price: float):
        """Agent requests additional resources"""
        # Find matching offers
        matches = [o for o in self.offers
                  if o["resource"] == resource
                  and o["amount"] >= amount]

        if matches:
            # Execute trade
            offer = matches[0]
            self.transfer_resources(
                from_agent=offer["agent"],
                to_agent=agent_id,
                resource=resource,
                amount=amount
            )
            return True
        return False
```

**Priority-Based Allocation**:
```python
def allocate_scarce_resource(self, resource: str,
                            competing_agents: list):
    """
    Allocate scarce resources based on agent priority
    """
    # Calculate priority scores
    scores = []
    for agent in competing_agents:
        score = (
            agent.task_priority * 0.4 +
            agent.progress_rate * 0.3 +
            (1 - agent.budget_utilization) * 0.2 +
            agent.expected_completion * 0.1
        )
        scores.append((agent, score))

    # Sort by priority
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)

    # Allocate to highest priority agent
    return ranked[0][0]
```

### 4.3 Contract-Based Coordination Patterns

**Sequential Pipeline**:
```python
def sequential_pipeline(task, total_budget):
    """
    Chain of agents, each consuming output of previous
    """
    contracts = [
        Contract("DataCollector", budget=total_budget * 0.3),
        Contract("Analyzer", budget=total_budget * 0.4),
        Contract("Reporter", budget=total_budget * 0.3)
    ]

    result = task
    for contract in contracts:
        agent = spawn_agent(contract)
        result = agent.execute(result)
        if not contract.is_fulfilled():
            return partial_result(result, contract)

    return result
```

**Parallel Competition**:
```python
def parallel_competition(task, total_budget):
    """
    Multiple agents solve same task, best result wins
    """
    num_agents = 3
    budget_per_agent = total_budget / num_agents

    contracts = [
        Contract(f"Solver_{i}",
                budget=budget_per_agent,
                deadline=time.time() + 300)
        for i in range(num_agents)
    ]

    # Run in parallel
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(spawn_agent(c).execute, task)
                  for c in contracts]
        results = [f.result() for f in futures]

    # Select best result
    return max(results, key=lambda r: r.quality_score)
```

**Collaborative Ensemble**:
```python
def collaborative_ensemble(task, total_budget):
    """
    Specialized agents collaborate on different aspects
    """
    contracts = {
        "research": Contract("Researcher",
                           budget={"tokens": 50K, "web": 10}),
        "analysis": Contract("Analyst",
                           budget={"tokens": 40K, "api": 20}),
        "synthesis": Contract("Writer",
                            budget={"tokens": 30K})
    }

    # Execute in dependency order
    research_data = contracts["research"].execute(task)
    analysis_results = contracts["analysis"].execute(research_data)
    final_report = contracts["synthesis"].execute(analysis_results)

    return final_report
```

---

## 5. Implementation Architecture

### 5.1 System Components

```
┌─────────────────────────────────────────────────────┐
│                 Contract Manager                     │
│  - Contract Registry                                 │
│  - Lifecycle Management                              │
│  - Resource Pool                                     │
└──────────────┬──────────────────────────────────────┘
               │
    ┌──────────┴──────────┬──────────────┬──────────┐
    ↓                     ↓              ↓          ↓
┌─────────┐      ┌──────────────┐  ┌────────┐  ┌──────────┐
│ Agent   │      │   Resource   │  │ Temporal│  │Contract  │
│ Runtime │←────→│   Monitor    │  │Monitor │  │Enforcer  │
└─────────┘      └──────────────┘  └────────┘  └──────────┘
    │                     │              │          │
    ↓                     ↓              ↓          ↓
┌─────────────────────────────────────────────────────────┐
│                    Metrics & Logging                     │
│  - Resource consumption logs                             │
│  - Performance metrics                                   │
│  - Contract violation events                             │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Contract Specification Format

**YAML Contract Definition**:
```yaml
contract:
  id: "code-review-agent-2025-q4"
  version: "1.0"

  metadata:
    name: "Code Review Agent"
    description: "Automated code review for pull requests"
    created: "2025-10-01T00:00:00Z"
    expires: "2025-12-31T23:59:59Z"

  inputs:
    schema:
      type: "object"
      properties:
        repository: {type: "string"}
        pull_request_id: {type: "integer"}
        target_branch: {type: "string"}
    constraints:
      max_file_size: "1MB"
      max_files: 50

  outputs:
    schema:
      type: "object"
      properties:
        overall_score: {type: "number", min: 0, max: 100}
        issues: {type: "array", items: {type: "object"}}
        recommendations: {type: "array"}
    quality_criteria:
      min_confidence: 0.8
      max_false_positives: 0.05

  skills:
    - static_analysis
    - security_scanning
    - style_checking
    - complexity_analysis

  resources:
    tokens:
      budget: 50000
      rate_limit: 10000  # per minute
    api_calls:
      budget: 30
      allowed_apis:
        - github_api
        - static_analyzer
    web_searches:
      budget: 0  # no web access
    memory:
      budget: "500MB"
    cost:
      budget: 2.50  # USD

  temporal:
    max_duration: "5 minutes"
    deadline_type: "soft"
    soft_deadline: "3 minutes"
    quality_decay: 0.1  # per minute after soft deadline

  success_criteria:
    - name: "completion_rate"
      condition: "all_files_reviewed == true"
      weight: 0.4
    - name: "accuracy"
      condition: "false_positive_rate < 0.05"
      weight: 0.3
    - name: "timeliness"
      condition: "completion_time < 300"  # seconds
      weight: 0.3

  termination_conditions:
    - type: "time_limit"
      condition: "elapsed_time > max_duration"
    - type: "resource_exhaustion"
      condition: "any_resource_exceeded"
    - type: "task_completion"
      condition: "success_criteria_met"
    - type: "contract_expiration"
      condition: "current_time > expires"
```

### 5.3 Agent Runtime Integration

**Contract-Aware Agent Wrapper**:
```python
class ContractAgent:
    def __init__(self, contract: Contract, base_agent: Agent):
        self.contract = contract
        self.agent = base_agent
        self.resource_monitor = ResourceMonitor(contract)
        self.temporal_monitor = TemporalMonitor(contract)

    def execute(self, task):
        """Execute task within contract constraints"""
        try:
            # Activate contract
            self.contract.state = ContractState.ACTIVE

            # Inject contract awareness into agent
            self.agent.set_budget_callback(
                self.resource_monitor.get_remaining
            )
            self.agent.set_time_pressure_callback(
                self.temporal_monitor.get_time_pressure
            )

            # Execute with monitoring
            result = self._monitored_execution(task)

            # Verify success criteria
            if self._check_success_criteria(result):
                self.contract.state = ContractState.FULFILLED
                return result
            else:
                self.contract.state = ContractState.INCOMPLETE
                return PartialResult(result, self.contract)

        except ContractViolation as e:
            self.contract.state = ContractState.VIOLATED
            self._handle_violation(e)
            raise
        finally:
            self._log_metrics()

    def _monitored_execution(self, task):
        """Execute with resource/time monitoring"""
        result = None

        # Wrap agent tools with monitors
        for tool in self.agent.tools:
            tool.set_pre_hook(self._pre_tool_check)
            tool.set_post_hook(self._post_tool_update)

        # Execute with periodic checks
        check_interval = 5  # seconds
        last_check = time.time()

        for step in self.agent.execute_streaming(task):
            # Periodic monitoring
            if time.time() - last_check > check_interval:
                self.temporal_monitor.check_deadline()
                self._check_resource_health()
                last_check = time.time()

            result = step

        return result

    def _pre_tool_check(self, tool_name: str, args: dict):
        """Check before tool execution"""
        # Estimate resource cost
        estimated_cost = self._estimate_tool_cost(tool_name, args)

        # Verify sufficient budget
        for resource, cost in estimated_cost.items():
            remaining = self.resource_monitor.get_remaining(resource)
            if cost > remaining:
                raise InsufficientBudgetError(
                    f"Tool {tool_name} requires {cost} {resource}, "
                    f"only {remaining} remaining"
                )

    def _post_tool_update(self, tool_name: str,
                         result: dict, actual_cost: dict):
        """Update resource consumption after tool use"""
        for resource, cost in actual_cost.items():
            self.resource_monitor.consume(resource, cost)
```

### 5.4 LLM Integration: Contract-Aware Prompting

**System Prompt Enhancement**:
```python
def generate_contract_aware_prompt(contract: Contract, task: str):
    """Generate system prompt with contract constraints"""

    budget_info = "\n".join([
        f"- {resource}: {budget} {unit} "
        f"({contract.resource_monitor.get_remaining(resource)} remaining)"
        for resource, budget in contract.resources.items()
    ])

    time_info = f"""
    Deadline: {contract.temporal.deadline}
    Time remaining: {contract.temporal_monitor.get_remaining_time()}
    Time pressure: {contract.temporal_monitor.get_time_pressure():.0%}
    """

    prompt = f"""
You are operating under a formal Agent Contract with explicit resource and time constraints.

# Task
{task}

# Resource Budget
{budget_info}

# Temporal Constraints
{time_info}

# Strategic Guidance
Given your remaining budget and time:
1. Prioritize actions with highest information-to-cost ratio
2. If budget is running low (<30%), switch to conservative strategies
3. If time pressure is high (>70%), prioritize speed over thoroughness
4. If you cannot complete the full task, return partial results with confidence scores

# Available Tools
{format_tools(contract.skills)}

# Output Requirements
{format_output_spec(contract.outputs)}

Think step-by-step about how to allocate your remaining resources optimally.
"""
    return prompt
```

**Dynamic Prompting Based on Budget State**:
```python
def adaptive_instruction(budget_utilization: float):
    """Adjust instructions based on budget consumption"""

    if budget_utilization < 0.3:
        return """
        You have ample budget remaining. Prioritize thoroughness and quality.
        Use web search and comprehensive analysis tools liberally.
        """
    elif budget_utilization < 0.7:
        return """
        You're at moderate budget utilization. Balance quality and efficiency.
        Be selective about expensive tools like web search.
        """
    else:
        return """
        ⚠️ Budget is running low! Enter conservation mode:
        - Rely on parametric knowledge instead of web search
        - Use cached results where possible
        - Prepare to return partial results if needed
        - Focus only on highest-priority aspects
        """
```

---

## 6. Use Cases and Examples

### 6.1 Example 1: Research Report Generation

**Scenario**: Generate a comprehensive market research report

**Contract Specification**:
```python
research_contract = Contract(
    name="Market Research Report",
    inputs={
        "topic": "Electric Vehicle Market in Europe",
        "depth": "comprehensive"
    },
    outputs={
        "schema": "markdown_report",
        "min_length": 5000,
        "sections": ["executive_summary", "market_analysis",
                    "competitive_landscape", "recommendations"]
    },
    skills=["web_search", "data_analysis", "report_writing"],
    resources={
        "tokens": 150000,
        "web_searches": 20,
        "api_calls": 40
    },
    temporal={
        "deadline": "2 hours",
        "urgency": "normal"
    }
)
```

**Execution Strategy**:
```python
def execute_research_report(contract):
    """Execute research with budget-aware strategy"""

    # Phase 1: Information Gathering (40% of budget)
    web_budget = int(contract.resources["web_searches"] * 0.4)
    token_budget_p1 = int(contract.resources["tokens"] * 0.4)

    research_data = gather_information(
        topic=contract.inputs["topic"],
        max_searches=web_budget,
        max_tokens=token_budget_p1
    )

    # Phase 2: Analysis (30% of budget)
    token_budget_p2 = int(contract.resources["tokens"] * 0.3)

    analysis = analyze_data(
        data=research_data,
        max_tokens=token_budget_p2,
        max_api_calls=int(contract.resources["api_calls"] * 0.3)
    )

    # Phase 3: Report Writing (30% of budget)
    token_budget_p3 = int(contract.resources["tokens"] * 0.3)

    report = generate_report(
        analysis=analysis,
        max_tokens=token_budget_p3,
        required_sections=contract.outputs["sections"]
    )

    return report
```

**Time-Resource Tradeoff**:

| Mode | Time | Tokens | Web Searches | Quality |
|------|------|--------|--------------|---------|
| Urgent | 30 min | 200K | 30 | 85% |
| Normal | 2 hours | 150K | 20 | 95% |
| Economical | 4 hours | 80K | 10 | 90% |

### 6.2 Example 2: Code Review Agent

**Scenario**: Automated pull request review

**Contract Specification**:
```python
code_review_contract = Contract(
    name="PR Review Agent",
    inputs={
        "repository": "company/product",
        "pr_number": 1234,
        "files_changed": 15
    },
    outputs={
        "review_comments": List[ReviewComment],
        "overall_score": float,
        "approval_recommendation": bool
    },
    skills=["static_analysis", "security_scan", "style_check"],
    resources={
        "tokens": 30000,
        "api_calls": 25,  # GitHub API + analysis tools
        "cost": 1.50
    },
    temporal={
        "deadline": "5 minutes",  # Fast feedback
        "urgency": "high"
    }
)
```

**Budget Allocation Strategy**:
```python
def review_pull_request(contract):
    """Review PR with resource constraints"""

    files = fetch_pr_files(contract.inputs["pr_number"])
    num_files = len(files)

    # Allocate tokens per file
    tokens_per_file = contract.resources["tokens"] // num_files

    reviews = []
    for file in files:
        # Check remaining budget
        remaining = contract.resource_monitor.get_remaining("tokens")

        if remaining < tokens_per_file * 0.5:
            # Running low, use fast mode
            review = quick_review(file, max_tokens=remaining // 2)
        else:
            # Normal review
            review = thorough_review(file, max_tokens=tokens_per_file)

        reviews.append(review)

        # Check time pressure
        if contract.temporal_monitor.get_time_pressure() > 0.8:
            # Running out of time, summarize remaining files
            reviews.extend(batch_review(files[len(reviews):]))
            break

    return consolidate_reviews(reviews)
```

### 6.3 Example 3: Customer Support Multi-Agent System

**Scenario**: Handle customer support tickets with specialized agents

**Hierarchical Contract Structure**:
```python
# Coordinator Agent
coordinator_contract = Contract(
    name="Support Coordinator",
    resources={
        "tokens": 300000,
        "api_calls": 100,
        "duration": "1 hour"
    },
    temporal={
        "sla": "30 minutes",  # Response SLA
        "expiration": "24 hours"  # Agent lifecycle
    }
)

# Specialized Sub-Agents
specialist_contracts = {
    "triage": Contract(
        name="Triage Agent",
        resources={"tokens": 20000, "api": 10},
        temporal={"deadline": "2 minutes"}
    ),
    "technical": Contract(
        name="Technical Support",
        resources={"tokens": 100000, "api": 40, "web": 5},
        temporal={"deadline": "20 minutes"}
    ),
    "billing": Contract(
        name="Billing Support",
        resources={"tokens": 30000, "api": 20},
        temporal={"deadline": "10 minutes"}
    ),
    "escalation": Contract(
        name="Escalation Handler",
        resources={"tokens": 50000, "api": 20},
        temporal={"deadline": "15 minutes"}
    )
}
```

**Execution Flow**:
```python
def handle_support_ticket(ticket, coordinator_contract):
    """Route and handle support ticket"""

    # Triage (determine category and urgency)
    triage_agent = spawn_agent(specialist_contracts["triage"])
    category, urgency = triage_agent.classify(ticket)

    # Allocate resources based on urgency
    if urgency == "critical":
        # Relax resource constraints, tighten time
        specialist_contracts[category].resources["tokens"] *= 1.5
        specialist_contracts[category].temporal["deadline"] *= 0.5

    # Route to specialist
    specialist = spawn_agent(specialist_contracts[category])

    try:
        response = specialist.handle(ticket)

        if response.requires_escalation:
            # Escalate with remaining coordinator budget
            escalation = spawn_agent(specialist_contracts["escalation"])
            response = escalation.handle(ticket, context=response)

        return response

    except ContractViolation:
        # Graceful degradation
        return generate_holding_response(ticket)
```

### 6.4 Example 4: Data Pipeline with Quality-Speed Tradeoff

**Scenario**: Process large dataset with adjustable quality-speed tradeoff

**Adaptive Contract**:
```python
def create_adaptive_pipeline_contract(data_size, deadline, quality_target):
    """
    Create contract that adapts to data size and requirements
    """

    # Estimate base resources
    base_tokens = data_size * 100  # tokens per record
    base_api = data_size * 0.5     # API calls per record

    # Apply quality multiplier
    quality_multipliers = {
        "quick": 0.5,
        "standard": 1.0,
        "thorough": 2.0
    }
    multiplier = quality_multipliers[quality_target]

    return Contract(
        name=f"Data Pipeline - {quality_target}",
        resources={
            "tokens": int(base_tokens * multiplier),
            "api_calls": int(base_api * multiplier),
            "memory": f"{data_size * 10}MB"
        },
        temporal={
            "deadline": deadline,
            "time_budget": deadline * 0.9  # 10% buffer
        },
        outputs={
            "min_accuracy": 0.6 if quality_target == "quick" else 0.95
        }
    )
```

**Adaptive Processing**:
```python
def process_dataset(data, contract):
    """Process with adaptive quality based on constraints"""

    results = []

    for i, record in enumerate(data):
        # Calculate remaining budget per record
        records_remaining = len(data) - i
        tokens_per_record = (
            contract.resource_monitor.get_remaining("tokens")
            / records_remaining
        )

        # Adapt processing strategy
        if tokens_per_record > 500:
            # Thorough processing
            result = deep_analysis(record, max_tokens=500)
        elif tokens_per_record > 200:
            # Standard processing
            result = standard_analysis(record, max_tokens=200)
        else:
            # Quick processing
            result = quick_analysis(record, max_tokens=100)

        results.append(result)

        # Early exit if time running out
        if contract.temporal_monitor.get_time_pressure() > 0.95:
            # Process remaining records in batch
            results.extend(batch_process(data[i+1:]))
            break

    return results
```

---

## 7. Empirical Considerations

### 7.1 Performance Metrics

**Contract-Level Metrics**:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Budget Utilization | $U_r = \frac{c_r}{b_r}$ | Resource efficiency |
| Time Efficiency | $U_t = \frac{t_{actual}}{t_{budget}}$ | Temporal efficiency |
| Quality Score | $Q = f(accuracy, completeness)$ | Output quality |
| Cost Effectiveness | $E = \frac{Q}{cost}$ | Quality per dollar |
| SLA Compliance | $\frac{\text{on-time completions}}{\text{total tasks}}$ | Reliability |

**System-Level Metrics**:
- **Throughput**: Tasks completed per hour
- **Resource Utilization**: Average budget utilization across all agents
- **Violation Rate**: Percentage of contracts violated
- **Renegotiation Frequency**: How often contracts need adjustment

### 7.2 Optimization Opportunities

**Historical Learning**:
```python
class ContractOptimizer:
    def __init__(self):
        self.history = []  # Contract execution history

    def suggest_contract(self, task_type: str):
        """Suggest optimal contract based on historical data"""

        # Filter similar past tasks
        similar = [h for h in self.history
                  if h.task_type == task_type]

        if not similar:
            return default_contract(task_type)

        # Analyze successful contracts
        successful = [h for h in similar
                     if h.state == ContractState.FULFILLED]

        # Calculate statistics
        avg_tokens = np.mean([h.tokens_used for h in successful])
        avg_time = np.mean([h.time_elapsed for h in successful])

        # Add buffer (20%)
        recommended = Contract(
            resources={
                "tokens": int(avg_tokens * 1.2),
                "api_calls": int(np.mean([h.api_calls for h in successful]) * 1.2)
            },
            temporal={
                "deadline": avg_time * 1.2
            }
        )

        return recommended
```

**A/B Testing Contracts**:
```python
def ab_test_contracts(task_type: str, n_trials: int = 100):
    """Test different contract configurations"""

    variants = {
        "aggressive": Contract(
            resources={"tokens": 50000, "time": 300}
        ),
        "moderate": Contract(
            resources={"tokens": 100000, "time": 600}
        ),
        "conservative": Contract(
            resources={"tokens": 150000, "time": 900}
        )
    }

    results = {v: [] for v in variants}

    for _ in range(n_trials):
        for variant_name, contract in variants.items():
            task = generate_task(task_type)
            result = execute_with_contract(task, contract)
            results[variant_name].append(result)

    # Analyze results
    for variant_name, outcomes in results.items():
        success_rate = sum(1 for o in outcomes
                          if o.state == ContractState.FULFILLED) / n_trials
        avg_cost = np.mean([o.total_cost for o in outcomes])
        avg_quality = np.mean([o.quality_score for o in outcomes])

        print(f"{variant_name}: {success_rate:.1%} success, "
              f"${avg_cost:.2f} cost, {avg_quality:.2f} quality")
```

### 7.3 Common Pitfalls and Solutions

**Problem 1: Overly Restrictive Contracts**
- **Symptom**: High violation rate, low completion rate
- **Solution**: Implement automatic budget scaling based on task complexity

**Problem 2: Budget Waste**
- **Symptom**: Low utilization, excess budget remaining
- **Solution**: Reallocate unused budget to quality improvements or early termination

**Problem 3: Deadline Pressure**
- **Symptom**: Rushed outputs with low quality as deadline approaches
- **Solution**: Implement soft deadlines with graceful degradation strategies

**Problem 4: Resource Contention**
- **Symptom**: Multiple agents competing for scarce resources
- **Solution**: Implement priority queues and resource reservation systems

---

## 8. Future Directions

### 8.1 Machine Learning Integration

**Learned Resource Estimation**:
Train models to predict optimal resource allocation:
```python
class ResourcePredictor:
    def __init__(self):
        self.model = train_model(historical_contracts)

    def predict_requirements(self, task_description: str):
        """Predict optimal resources for task"""
        features = extract_features(task_description)
        predicted = self.model.predict(features)

        return {
            "tokens": predicted["tokens"],
            "api_calls": predicted["api_calls"],
            "time": predicted["time"],
            "confidence": predicted["confidence"]
        }
```

**Reinforcement Learning for Contract Negotiation**:
Agents learn to negotiate contracts optimally:
- State: Current budget, task progress, time remaining
- Actions: Request extension, reduce quality target, delegate subtask
- Reward: Task completion weighted by resource efficiency

### 8.2 Blockchain-Based Contract Enforcement

**Immutable Contract Records**:
- Store contracts on blockchain for auditability
- Smart contracts for automatic resource allocation
- Cryptographic verification of contract fulfillment

**Decentralized Agent Marketplaces**:
- Agents advertise capabilities via contracts
- Requesters post tasks with resource budgets
- Automated matching and execution

### 8.3 Cross-Organizational Contracts

**Federated Agent Systems**:
```yaml
cross_org_contract:
  parties:
    - organization: "CompanyA"
      agent: "ResearchAgent"
      contributes: {data: "market_data", resources: {tokens: 50K}}
    - organization: "CompanyB"
      agent: "AnalysisAgent"
      contributes: {skills: ["ml_analysis"], resources: {compute: "10 GPU-hours"}}

  shared_resources:
    tokens: 150000
    api_calls: 100

  resource_sharing:
    CompanyA_percentage: 60%
    CompanyB_percentage: 40%

  output_distribution:
    CompanyA: ["market_insights", "raw_data"]
    CompanyB: ["analysis_results", "ml_models"]
```

### 8.4 Human-in-the-Loop Contracts

**Approval Gates**:
```python
contract_with_human_approval = Contract(
    # ... standard fields ...
    approval_required=[
        {"stage": "before_expensive_operation",
         "threshold": {"cost": 10.00}},
        {"stage": "before_external_api",
         "apis": ["payment_api", "email_api"]},
        {"stage": "before_deadline_extension"}
    ]
)
```

**Dynamic Renegotiation with Humans**:
```python
def request_human_renegotiation(agent_id, reason):
    """
    Agent requests human approval for contract modification
    """
    notification = f"""
    Agent {agent_id} requests contract modification:

    Current State:
    - Progress: 75% complete
    - Resources: 90% of token budget consumed
    - Time: 80% of deadline elapsed

    Request:
    - Additional 30K tokens
    - 10-minute deadline extension

    Reason: {reason}

    [Approve] [Deny] [Modify]
    """

    return await_human_decision(notification)
```

---

## 9. Conclusion

### 9.1 Summary of Contributions

We have introduced **Agent Contracts** as a formal framework for governing autonomous AI agents through explicit resource and temporal constraints. The key contributions include:

1. **Formal Mathematical Framework**: A rigorous definition of agent contracts as multi-dimensional optimization problems with resource budgets, temporal constraints, and quality objectives.

2. **Time-Resource Tradeoff Surface**: Recognition that urgency and economy create a rich design space where agents can adapt strategies based on constraint priorities.

3. **Lifecycle Management**: Explicit contract states (drafted, active, fulfilled, violated, expired) that provide clear accountability and resource release mechanisms.

4. **Multi-Agent Coordination**: Hierarchical contract structures and resource markets that enable sophisticated multi-agent collaboration without centralized control.

5. **Practical Implementation**: Concrete architectures, code examples, and integration patterns for real-world deployment.

### 9.2 Addressing the Original Motivation

Returning to our initial question: **Is "Agent Contract" more than another buzzword?**

**Yes**, when it emphasizes:
- **Executable constraints** that shape agent behavior, not just descriptions
- **Resource lifecycle management** with explicit allocation and deallocation
- **Time-resource tradeoffs** that enable strategic adaptation
- **Formal verification** of agent behavior against contract terms
- **Economic and governance** implications for enterprise deployment

**Not valuable** if it merely:
- Renames existing concepts (system prompts, tool specifications)
- Adds complexity without improving outcomes
- Becomes pure documentation without runtime enforcement

### 9.3 When to Use Agent Contracts

**Strong Fit**:
- Production systems with cost/performance requirements
- Multi-agent systems requiring resource coordination
- Enterprise deployments needing governance and audit trails
- Applications with hard SLAs or deadlines
- Research on optimal agent behavior under constraints

**Weak Fit**:
- Exploratory development where flexibility matters most
- Simple single-purpose agents with unlimited resources
- Contexts where human oversight handles all constraints
- Systems where contracts would add more overhead than value

### 9.4 Impact on the Field

Agent Contracts represent a shift from **unbounded agency** to **bounded optimization**. This has implications for:

**Research**: New optimization problems around contract design, resource allocation, and multi-agent coordination under constraints.

**Industry**: Practical governance mechanisms for deploying autonomous systems at scale with predictable costs and behavior.

**Policy**: Clear accountability frameworks for AI systems operating with explicit boundaries and termination conditions.

### 9.5 Call to Action

We propose that the AI community:

1. **Standardize contract specifications** similar to OpenAPI for REST APIs
2. **Build contract enforcement layers** into major agentic frameworks
3. **Develop benchmarks** for contract-aware agent performance
4. **Create contract marketplaces** where agents advertise capabilities
5. **Research optimal contract design** through empirical studies and theoretical analysis

Agent Contracts are not the final answer to agentic AI governance, but they provide a concrete, implementable framework that addresses real production challenges. As the field matures from research prototypes to production systems, such formal mechanisms will become increasingly essential.

---

## References

1. Bolton, P., & Dewatripont, M. (2005). *Contract Theory*. MIT Press.

2. Wooldridge, M. (2009). *An Introduction to MultiAgent Systems*. Wiley.

3. Buttazzo, G. C. (2011). *Hard Real-Time Computing Systems*. Springer.

4. Bertsimas, D., & Tsitsiklis, J. N. (1997). *Introduction to Linear Optimization*. Athena Scientific.

5. Russell, S., & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.

6. Shoham, Y., & Leyton-Brown, K. (2008). *Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations*. Cambridge University Press.

7. Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *NeurIPS*.

8. Yao, S., et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR*.

9. Park, J. S., et al. (2023). "Generative Agents: Interactive Simulacra of Human Behavior." *UIST*.

10. Xi, Z., et al. (2023). "The Rise and Potential of Large Language Model Based Agents: A Survey." *arXiv:2309.07864*.

---

## Appendices

### Appendix A: Complete Contract Schema (JSON)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "AgentContract",
  "type": "object",
  "required": ["id", "inputs", "outputs", "skills", "resources", "temporal"],
  "properties": {
    "id": {"type": "string"},
    "version": {"type": "string"},
    "metadata": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "created": {"type": "string", "format": "date-time"},
        "expires": {"type": "string", "format": "date-time"}
      }
    },
    "inputs": {
      "type": "object",
      "properties": {
        "schema": {"type": "object"},
        "constraints": {"type": "object"}
      }
    },
    "outputs": {
      "type": "object",
      "properties": {
        "schema": {"type": "object"},
        "quality_criteria": {"type": "object"}
      }
    },
    "skills": {
      "type": "array",
      "items": {"type": "string"}
    },
    "resources": {
      "type": "object",
      "properties": {
        "tokens": {
          "type": "object",
          "properties": {
            "budget": {"type": "integer"},
            "rate_limit": {"type": "integer"}
          }
        },
        "api_calls": {
          "type": "object",
          "properties": {
            "budget": {"type": "integer"},
            "allowed_apis": {
              "type": "array",
              "items": {"type": "string"}
            }
          }
        },
        "web_searches": {
          "type": "object",
          "properties": {
            "budget": {"type": "integer"}
          }
        },
        "cost": {
          "type": "object",
          "properties": {
            "budget": {"type": "number"},
            "currency": {"type": "string"}
          }
        }
      }
    },
    "temporal": {
      "type": "object",
      "properties": {
        "max_duration": {"type": "string"},
        "deadline": {"type": "string", "format": "date-time"},
        "deadline_type": {
          "type": "string",
          "enum": ["hard", "soft"]
        },
        "soft_deadline": {"type": "string"},
        "quality_decay": {"type": "number"}
      }
    },
    "success_criteria": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "condition": {"type": "string"},
          "weight": {"type": "number"}
        }
      }
    },
    "termination_conditions": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "type": {"type": "string"},
          "condition": {"type": "string"}
        }
      }
    }
  }
}
```

### Appendix B: Reference Implementation (Python)

*Reference implementation planned for future release.*

Planned modules:
- `contract.py`: Core contract data structures
- `monitor.py`: Resource and temporal monitoring
- `enforcer.py`: Contract enforcement logic
- `optimizer.py`: Historical learning and optimization
- `coordinator.py`: Multi-agent coordination

### Appendix C: Comparison with Related Concepts

| Concept | Similarity | Difference |
|---------|-----------|------------|
| System Prompts | Both define agent behavior | Contracts are executable and enforceable |
| Tool Specifications | Both define capabilities | Contracts add resource budgets |
| SLAs | Both define performance expectations | Contracts include resource management |
| Resource Quotas | Both limit consumption | Contracts include optimization guidance |
| Workflow Definitions | Both structure tasks | Contracts add economic constraints |

---

**Document Version**: 1.0
**Last Updated**: October 29, 2025
**Authors**: Qing Ye (with assistance from Claude, Anthropic)
**License**: CC BY 4.0
