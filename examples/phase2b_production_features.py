"""Phase 2B: Production Governance Features - Examples

This file demonstrates the Phase 2B features:
1. ContractAgent base wrapper
2. LangChain integration
3. Contract templates
4. Execution logging and audit trails

Phase 2B makes Agent Contracts production-ready by providing easy integration
with popular frameworks and pre-configured templates for common use cases.

NOTE: This file has imports scattered throughout for tutorial purposes.
# ruff: noqa: E402
"""

# Example 1: Using Contract Templates
# ====================================

from agent_contracts import (
    CodeReviewContract,
    Contract,
    ContractMode,
    CustomerSupportContract,
    DataAnalysisContract,
    ResearchContract,
)

# Research contract (Whitepaper Section 6.1)
research_contract = ResearchContract.create(
    topic="Quantum Computing Applications in Cryptography",
    depth="comprehensive",
    mode=ContractMode.BALANCED,
)

print("Research Contract:")
print(f"  ID: {research_contract.id}")
print(f"  Tokens: {research_contract.resources.tokens}")
print(f"  Web Searches: {research_contract.resources.web_searches}")
print(f"  Max Duration: {research_contract.temporal.max_duration}")
print()

# Code review contract (Whitepaper Section 6.2)
code_review_contract = CodeReviewContract.create(
    repository="myorg/myapp",
    pr_number=1234,
    files_changed=12,  # Medium-sized PR
)

print("Code Review Contract:")
print(f"  ID: {code_review_contract.id}")
print(f"  Tokens: {code_review_contract.resources.tokens}")
print(f"  Cost Budget: ${code_review_contract.resources.cost_usd}")
print()

# Customer support contract
support_contract = CustomerSupportContract.create(
    ticket_id="TICKET-5678",
    priority="high",  # Auto-selects URGENT mode
)

print("Customer Support Contract:")
print(f"  ID: {support_contract.id}")
print(f"  Mode: {support_contract.mode.value}")
print(f"  Tokens: {support_contract.resources.tokens}")
print()

# Data analysis contract
analysis_contract = DataAnalysisContract.create(
    dataset_name="sales_data.csv",
    dataset_size_mb=5.2,  # Scales resources based on size
    analysis_type="statistical",
)

print("Data Analysis Contract:")
print(f"  ID: {analysis_contract.id}")
print(f"  Tokens: {analysis_contract.resources.tokens}")
print(f"  Memory: {analysis_contract.resources.memory_mb}MB")
print()

# Example 2: ContractAgent Base Wrapper
# ======================================

from agent_contracts.core import ContractAgent, ResourceConstraints  # noqa: E402


# Define a simple agent (any callable)
def research_agent(query: str) -> str:
    """Simple research agent that returns a report."""
    return f"Research report on: {query}\n\nKey findings: ..."


# Wrap with contract
contract = Contract(
    id="research-task",
    resources=ResourceConstraints(tokens=5000, cost_usd=0.50),
)

wrapped_agent = ContractAgent(contract=contract, agent=research_agent)

# Execute with contract enforcement
result = wrapped_agent.execute("AI Safety")

print("ContractAgent Execution:")
print(f"  Success: {result.success}")
print(f"  Output: {result.output[:50]}...")
print(f"  Tokens Used: {result.execution_log.resource_usage['tokens']}")
print(f"  Cost: ${result.execution_log.resource_usage['cost_usd']:.4f}")
print(f"  Violations: {result.violations}")
print()

# Example 3: LangChain Integration (Optional)
# ===========================================

# Note: This example requires langchain to be installed:
# pip install langchain langchain-openai

try:
    from langchain.chains import LLMChain
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate

    from agent_contracts.integrations.langchain import (
        ContractedChain,
        create_contracted_chain,
    )

    print("LangChain integration available!")

    # Method 1: Wrap existing chain
    prompt = PromptTemplate(input_variables=["topic"], template="Write a summary about {topic}")
    llm = OpenAI(temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt)

    # Wrap with contract
    contract = Contract(
        id="langchain-summary",
        resources=ResourceConstraints(tokens=1000, cost_usd=0.10),
    )
    contracted_chain = ContractedChain(contract=contract, chain=chain)

    # Execute (automatically tracks tokens and enforces budget)
    result = contracted_chain.execute({"topic": "Climate Change"})
    print(f"  LangChain Result: {result.output.get('text', '')[:50]}...")

    # Method 2: Use convenience function
    contracted_chain2 = create_contracted_chain(
        chain=chain,
        resources={"tokens": 1000, "cost_usd": 0.10},
        contract_id="quick-summary",
    )

except ImportError:
    print("LangChain not available (optional dependency)")
    print("Install with: pip install langchain langchain-openai")

print()

# Example 4: Execution Logging and Audit Trails
# =============================================

from datetime import timedelta  # noqa: E402

from agent_contracts.core import TemporalConstraints  # noqa: E402

# Create contract with temporal constraints
audit_contract = Contract(
    id="audited-task",
    resources=ResourceConstraints(tokens=3000, api_calls=10, cost_usd=0.30),
    temporal=TemporalConstraints(max_duration=timedelta(minutes=5)),
)


def audited_agent(task: str) -> str:
    """Agent with full execution logging."""
    return f"Completed: {task}"


# Execute with logging enabled (default)
wrapped = ContractAgent(contract=audit_contract, agent=audited_agent, enable_logging=True)

result = wrapped.execute("Process customer request")

# Access execution log
log = result.execution_log
print("Execution Audit Log:")
print(f"  Contract ID: {log.contract_id}")
print(f"  Start Time: {log.start_time}")
print(f"  End Time: {log.end_time}")
print(f"  Final State: {log.final_state.value}")
print("  Resource Usage:")
print(f"    - Tokens: {log.resource_usage['tokens']}")
print(f"    - API Calls: {log.resource_usage['api_calls']}")
print(f"    - Cost: ${log.resource_usage['cost_usd']:.4f}")
print("  Temporal Metrics:")
print(f"    - Elapsed: {log.temporal_metrics['elapsed_seconds']:.2f}s")
print(f"    - Deadline Met: {log.temporal_metrics['deadline_met']}")
print(f"  Events: {len(log.events)}")

# Export to JSON for external audit systems
log_json = wrapped.to_json()
print(f"\nExportable JSON log available with {len(str(log_json))} characters")

# Example 5: Budget Awareness
# ===========================

contract = Contract(
    id="budget-aware",
    resources=ResourceConstraints(tokens=10000, cost_usd=1.0),
)

agent = ContractAgent(contract=contract, agent=lambda x: f"Result: {x}")

# Agents can check remaining budget during execution
remaining = agent.get_remaining_budget()
print("\nBudget Awareness:")
print(f"  Remaining Tokens: {remaining['tokens']}")
print(f"  Remaining Cost: ${remaining['cost_usd']:.2f}")
print(f"  Remaining API Calls: {remaining['api_calls']}")

# Check time pressure (0.0 = no pressure, 1.0 = deadline reached)
time_pressure = agent.get_time_pressure()
print(f"  Time Pressure: {time_pressure:.2f}")

# Example 6: Custom Template Usage
# ================================

# Customize an existing template
custom_research = ResearchContract.create(
    topic="AI Ethics",
    mode=ContractMode.ECONOMICAL,  # Cost-optimized
    resources={
        "tokens": 50000,  # Override default
        "cost_usd": 1.0,  # Override default
        "web_searches": 15,  # Override default
    },
)

print("\nCustom Template:")
print(f"  Mode: {custom_research.mode.value}")
print(f"  Tokens: {custom_research.resources.tokens}")
print(f"  Cost: ${custom_research.resources.cost_usd}")

# Example 7: Complete Production Workflow
# =======================================

print("\n" + "=" * 60)
print("COMPLETE PRODUCTION WORKFLOW")
print("=" * 60)

# 1. Select template for your use case
contract = ResearchContract.create(
    topic="Machine Learning Interpretability",
    mode=ContractMode.BALANCED,
)

print(f"\n1. Created contract: {contract.id}")
print(f"   Mode: {contract.mode.value}")
print(f"   Budget: {contract.resources.tokens} tokens, ${contract.resources.cost_usd}")


# 2. Define your agent
def ml_research_agent(topic: str) -> str:
    # Your agent logic here
    # Could use LangChain, custom code, API calls, etc.
    return f"Research findings on {topic}..."


# 3. Wrap with contract
contracted_agent = ContractAgent(contract=contract, agent=ml_research_agent, strict_mode=True)

print("\n2. Wrapped agent with contract enforcement")

# 4. Execute
result = contracted_agent.execute("ML Interpretability")

print("\n3. Execution complete:")
print(f"   Success: {result.success}")
print(f"   State: {result.contract.state.value}")

# 5. Audit and monitor
print("\n4. Audit trail:")
print(f"   Tokens used: {result.execution_log.resource_usage['tokens']}")
print(f"   Cost: ${result.execution_log.resource_usage['cost_usd']:.4f}")
print(f"   Duration: {result.execution_log.temporal_metrics['elapsed_seconds']:.2f}s")
print(
    f"   Budget compliance: {result.execution_log.resource_usage['tokens'] <= contract.resources.tokens}"
)

# 6. Export logs for compliance
log_data = result.execution_log.to_dict()
print(f"\n5. Execution log exported ({len(str(log_data))} chars)")

print("\n" + "=" * 60)
print("Phase 2B Features Summary:")
print("=" * 60)
print("✅ ContractAgent base wrapper")
print("✅ LangChain integration (optional)")
print("✅ 4 Contract templates (Research, CodeReview, Support, DataAnalysis)")
print("✅ Execution logging and audit trails")
print("✅ Budget awareness APIs")
print("✅ Production-ready governance")
print("=" * 60)
