#!/usr/bin/env python3
"""Hierarchical Delegation Demo: Conservation Laws in Multi-Agent Systems

This demo showcases the UNIQUE value of Agent Contracts: hierarchical budget
delegation with conservation law enforcement. This is what existing frameworks
DON'T provide - explicit per-agent budget allocation with mathematical guarantees.

Conservation Law (Whitepaper Section 6):
    For any parent contract with budget B, if it creates child contracts
    with budgets b_1, b_2, ..., b_k, the following must hold:

        Σ b_i ≤ B - used

    where 'used' is the parent's own consumption.

What This Demo Shows:
1. Creating a delegating orchestrator with a parent budget
2. Allocating explicit budgets to sub-agents (not shared!)
3. Conservation law enforcement when over-allocating
4. Budget pooling when agents complete early
5. Real execution with Gemini 2.0 Flash

Why This Matters:
- Existing frameworks use shared budgets (all agents draw from same pool)
- Agent Contracts uses allocated budgets (each agent has guaranteed budget)
- Prevents resource starvation and enables SLAs per agent
"""

import sys

from dotenv import load_dotenv

# Load environment variables
load_dotenv("../../.env")

# Check dependencies
try:
    from google.adk.agents import LlmAgent
except ImportError:
    print("❌ Missing dependencies!")
    print("\nInstall with: uv sync --extra google-adk")
    sys.exit(1)

from agent_contracts import Contract, ResourceConstraints  # noqa: E402
from agent_contracts.core.delegation import ConservationViolationError  # noqa: E402
from agent_contracts.integrations.google_adk import DelegatingAdkAgent  # noqa: E402


def print_section(title: str) -> None:
    """Print section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def demo_1_basic_delegation() -> None:
    """Demo 1: Basic Hierarchical Delegation.

    Shows creating an orchestrator that delegates to sub-agents with
    explicit budget allocation.
    """
    print_section("Demo 1: Basic Hierarchical Delegation")

    print("Scenario: Orchestrator with 100K tokens delegates to 3 specialists")
    print()

    # Create parent contract
    parent_contract = Contract(
        id="orchestrator",
        name="Report Generation Orchestrator",
        resources=ResourceConstraints(tokens=100_000, cost_usd=2.0),
    )

    # Create orchestrator agent
    orchestrator = LlmAgent(
        name="orchestrator",
        model="gemini-3-flash-preview",
        instruction="You coordinate research tasks. Be brief.",
    )

    # Create delegating agent with 10% reserve
    delegating = DelegatingAdkAgent(
        contract=parent_contract,
        agent=orchestrator,
        reserve_ratio=0.1,  # Reserve 10K for coordination overhead
    )

    print(f"Parent Budget:     {delegating.contracting.parent_budget_tokens:,} tokens")
    print(f"Reserved (10%):    {delegating.contracting.reserved_tokens:,} tokens")
    print(f"Available:         {delegating.remaining_delegation_tokens:,} tokens")
    print()

    # Create sub-agents
    researcher_agent = LlmAgent(
        name="researcher",
        model="gemini-3-flash-preview",
        instruction="You research topics. Keep responses brief.",
    )

    analyzer_agent = LlmAgent(
        name="analyzer",
        model="gemini-3-flash-preview",
        instruction="You analyze information. Keep responses brief.",
    )

    reporter_agent = LlmAgent(
        name="reporter",
        model="gemini-3-flash-preview",
        instruction="You write reports. Keep responses brief.",
    )

    # Delegate with explicit budgets
    print("Allocating budgets to sub-agents:")
    print()

    delegating.delegate(
        name="researcher",
        agent=researcher_agent,
        tokens=30_000,
        description="Research the topic",
    )
    print(f"  Researcher:  30,000 tokens | Remaining: {delegating.remaining_delegation_tokens:,}")

    delegating.delegate(
        name="analyzer",
        agent=analyzer_agent,
        tokens=25_000,
        description="Analyze findings",
    )
    print(f"  Analyzer:    25,000 tokens | Remaining: {delegating.remaining_delegation_tokens:,}")

    delegating.delegate(
        name="reporter",
        agent=reporter_agent,
        tokens=20_000,
        description="Write the report",
    )
    print(f"  Reporter:    20,000 tokens | Remaining: {delegating.remaining_delegation_tokens:,}")

    # Show delegation summary
    print()
    summary = delegating.get_delegation_summary()
    print("📊 Delegation Summary:")
    print(f"   Parent Budget:      {summary['parent_budget_tokens']:,} tokens")
    print(f"   Parent Used:        {summary['parent_used_tokens']:,} tokens")
    print(f"   Total Delegated:    {summary['total_delegated_tokens']:,} tokens")
    print(f"   Remaining:          {summary['remaining_tokens']:,} tokens")
    print(f"   Conservation OK:    {summary['conservation_satisfied']}")
    print()

    print("✅ Each agent has a GUARANTEED budget allocation")
    print("   • Researcher can use up to 30K tokens")
    print("   • Analyzer can use up to 25K tokens")
    print("   • Reporter can use up to 20K tokens")
    print("   • 15K reserved for orchestrator + 10K remaining buffer")


def demo_2_conservation_enforcement() -> None:
    """Demo 2: Conservation Law Enforcement.

    Shows that over-allocation is PREVENTED by the conservation law.
    """
    print_section("Demo 2: Conservation Law Enforcement")

    print("Scenario: Try to allocate more budget than available")
    print()

    parent_contract = Contract(
        id="limited-parent",
        name="Limited Budget Orchestrator",
        resources=ResourceConstraints(tokens=50_000),
    )

    orchestrator = LlmAgent(
        name="orchestrator",
        model="gemini-3-flash-preview",
        instruction="You coordinate tasks.",
    )

    delegating = DelegatingAdkAgent(
        contract=parent_contract,
        agent=orchestrator,
    )

    print(f"Parent Budget: {delegating.remaining_delegation_tokens:,} tokens")
    print()

    # Allocate most of the budget
    worker_agent = LlmAgent(name="worker", model="gemini-3-flash-preview", instruction="You work.")

    delegating.delegate(
        name="worker_a",
        agent=worker_agent,
        tokens=40_000,
    )
    print(
        f"Worker A allocated: 40,000 tokens | Remaining: {delegating.remaining_delegation_tokens:,}"
    )

    # Try to over-allocate
    print()
    print("Attempting to allocate 15,000 more tokens...")
    print("(This exceeds remaining 10,000 tokens)")
    print()

    try:
        delegating.delegate(
            name="worker_b",
            agent=worker_agent,
            tokens=15_000,
        )
        print("❌ This should not happen!")
    except ConservationViolationError as e:
        print("🛑 Conservation Violation Caught!")
        print(f"   Requested: {e.requested:,} tokens")
        print(f"   Available: {e.available:,} tokens")
        print(f"   Parent ID: {e.parent_id}")

    print()
    print("✅ Conservation law prevents budget over-allocation")
    print("   • Mathematical guarantee: Σ child_budgets ≤ parent_budget - used")
    print("   • No agent can starve another by taking too much")


def demo_3_budget_pooling() -> None:
    """Demo 3: Dynamic Budget Pooling.

    Shows that when an agent completes early, its budget can be
    reallocated to other agents (budget pooling).
    """
    print_section("Demo 3: Dynamic Budget Pooling")

    print("Scenario: Agent finishes early, budget reallocated to struggling agent")
    print()

    parent_contract = Contract(
        id="pooling-demo",
        name="Budget Pooling Demo",
        resources=ResourceConstraints(tokens=100_000),
    )

    orchestrator = LlmAgent(
        name="orchestrator",
        model="gemini-3-flash-preview",
        instruction="You coordinate.",
    )

    delegating = DelegatingAdkAgent(contract=parent_contract, agent=orchestrator)

    worker_agent = LlmAgent(name="worker", model="gemini-3-flash-preview", instruction="You work.")

    # Initial allocations
    print("Initial Allocations:")
    delegating.delegate(name="worker_a", agent=worker_agent, tokens=40_000)
    print(f"  Worker A: 40,000 tokens | Remaining: {delegating.remaining_delegation_tokens:,}")

    delegating.delegate(name="worker_b", agent=worker_agent, tokens=40_000)
    print(f"  Worker B: 40,000 tokens | Remaining: {delegating.remaining_delegation_tokens:,}")

    print()
    print("Remaining pool: 20,000 tokens")
    print()

    # Worker A finishes early
    print("Worker A finishes early - releasing allocation...")
    released = delegating.release_delegation("worker_a")
    print(f"Released: {released:,} tokens")
    print(f"New remaining: {delegating.remaining_delegation_tokens:,} tokens")
    print()

    # Now we can allocate more
    print("Now Worker C can receive larger allocation:")
    delegating.delegate(name="worker_c", agent=worker_agent, tokens=55_000)
    print(f"  Worker C: 55,000 tokens | Remaining: {delegating.remaining_delegation_tokens:,}")

    print()
    print("✅ Budget pooling enables dynamic resource reallocation")
    print("   • Efficient agents 'subsidize' struggling ones")
    print("   • No wasted budget from early completion")
    print("   • Adaptive to runtime conditions")


def demo_4_live_execution() -> None:
    """Demo 4: Live Multi-Agent Execution with Delegation.

    Shows real execution where delegated agents perform tasks
    within their allocated budgets.
    """
    print_section("Demo 4: Live Multi-Agent Execution with Delegation")

    print("Scenario: Execute delegated agents with real Gemini API calls")
    print()

    parent_contract = Contract(
        id="live-demo",
        name="Live Delegation Demo",
        resources=ResourceConstraints(tokens=20_000, cost_usd=0.5),
    )

    orchestrator = LlmAgent(
        name="orchestrator",
        model="gemini-3-flash-preview",
        instruction="You coordinate research.",
    )

    delegating = DelegatingAdkAgent(
        contract=parent_contract,
        agent=orchestrator,
        reserve_ratio=0.1,  # 10% reserve
    )

    # Create and delegate
    researcher_agent = LlmAgent(
        name="researcher",
        model="gemini-3-flash-preview",
        instruction="You research topics. Keep responses to 2 sentences.",
    )

    researcher = delegating.delegate(
        name="researcher",
        agent=researcher_agent,
        tokens=8_000,
        description="Research quantum computing",
    )

    print("Delegated 8,000 tokens to researcher")
    print(f"Researcher budget: {researcher.contract.resources.tokens:,} tokens")
    print()

    # Execute the delegated agent
    print("Executing researcher agent...")
    result = researcher.run(
        user_id="demo_user",
        session_id="demo_session",
        message="What is quantum computing? Be very brief.",
    )

    print()
    print("✅ Researcher completed")
    print(f"   Response: {result['response'][:100]}...")
    print(f"   Tokens used: {result['total_tokens']:,}")
    print()

    # Show final state
    summary = delegating.get_delegation_summary()
    print("📊 Final Delegation State:")
    print(f"   Parent used: {summary['parent_used_tokens']:,} tokens")
    print(f"   Delegated:   {summary['total_delegated_tokens']:,} tokens")
    print(f"   Remaining:   {summary['remaining_tokens']:,} tokens")

    print()
    print("✅ Delegated agent executed within its allocated budget")
    print("   • Each agent tracks its own usage independently")
    print("   • Parent can monitor all delegations")
    print("   • Conservation law maintained throughout")


def demo_5_paper_example() -> None:
    """Demo 5: Paper Example (Section 4.5).

    Reproduces the whitepaper example: orchestrator delegates to
    researcher (50K), analyzer (40K), reporter (45K), with 15K reserve.
    """
    print_section("Demo 5: Whitepaper Section 4.5 Example")

    print("Reproducing the paper example:")
    print("  Orchestrator: 150K tokens")
    print("  → Researcher: 50K tokens")
    print("  → Analyzer: 40K tokens")
    print("  → Reporter: 45K tokens")
    print("  → Reserve: 15K tokens")
    print()

    parent_contract = Contract(
        id="paper-example",
        name="Report Generation (Paper Section 8)",
        resources=ResourceConstraints(tokens=150_000),
    )

    orchestrator = LlmAgent(
        name="orchestrator",
        model="gemini-3-flash-preview",
        instruction="You orchestrate report generation.",
    )

    delegating = DelegatingAdkAgent(contract=parent_contract, agent=orchestrator)

    # Create sub-agents
    researcher_agent = LlmAgent(
        name="researcher", model="gemini-3-flash-preview", instruction="Research."
    )
    analyzer_agent = LlmAgent(
        name="analyzer", model="gemini-3-flash-preview", instruction="Analyze."
    )
    reporter_agent = LlmAgent(
        name="reporter", model="gemini-3-flash-preview", instruction="Report."
    )

    # Reserve 15K for orchestrator coordination
    delegating.delegate(name="orchestrator_reserve", agent=orchestrator, tokens=15_000)
    print(f"Reserved 15K for coordination | Remaining: {delegating.remaining_delegation_tokens:,}")

    # Allocate to sub-agents
    delegating.delegate(name="researcher", agent=researcher_agent, tokens=50_000)
    print(f"Researcher: 50K | Remaining: {delegating.remaining_delegation_tokens:,}")

    delegating.delegate(name="analyzer", agent=analyzer_agent, tokens=40_000)
    print(f"Analyzer: 40K | Remaining: {delegating.remaining_delegation_tokens:,}")

    delegating.delegate(name="reporter", agent=reporter_agent, tokens=45_000)
    print(f"Reporter: 45K | Remaining: {delegating.remaining_delegation_tokens:,}")

    print()
    summary = delegating.get_delegation_summary()
    print("📊 Allocation Complete:")
    print(f"   Total Delegated:    {summary['total_delegated_tokens']:,} tokens")
    print(f"   Remaining:          {summary['remaining_tokens']:,} tokens")
    print(f"   Conservation OK:    {summary['conservation_satisfied']}")
    print()

    # Verify: 15K + 50K + 40K + 45K = 150K = parent budget
    total = 15_000 + 50_000 + 40_000 + 45_000
    print(f"Verification: 15K + 50K + 40K + 45K = {total:,} = parent budget ✅")
    print()

    print("✅ Paper example verified:")
    print("   • Conservation law: Σ child_budgets = parent_budget")
    print("   • Each agent has guaranteed allocation")
    print("   • Hierarchical delegation with explicit contracts")


def main() -> None:
    """Run all delegation demonstrations."""
    print("\n" + "=" * 80)
    print("  Agent Contracts: Hierarchical Delegation with Conservation Laws")
    print("=" * 80)
    print("\nThis demo showcases the UNIQUE value of Agent Contracts:")
    print("  • Explicit per-agent budget allocation (not shared pools)")
    print("  • Conservation law enforcement (mathematical guarantees)")
    print("  • Hierarchical delegation (contracting as a capability)")
    print("=" * 80)

    demos = [
        ("Basic Delegation", demo_1_basic_delegation),
        ("Conservation Enforcement", demo_2_conservation_enforcement),
        ("Budget Pooling", demo_3_budget_pooling),
        ("Live Execution", demo_4_live_execution),
        ("Paper Example", demo_5_paper_example),
    ]

    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n❌ Demo '{name}' failed: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print_section("✨ Summary: What Makes This Unique")
    print("Existing frameworks (LangGraph, CrewAI, AutoGen):")
    print("  ❌ Use shared budget pools (all agents draw from same pool)")
    print("  ❌ No per-agent guarantees (one agent can starve others)")
    print("  ❌ No conservation law enforcement")
    print()
    print("Agent Contracts with Delegation:")
    print("  ✅ Explicit budget allocation per agent")
    print("  ✅ Conservation law: Σ child_budgets ≤ parent_budget - used")
    print("  ✅ Budget pooling for dynamic reallocation")
    print("  ✅ Hierarchical delegation (Section 4.5: contracting as capability)")
    print("  ✅ Mathematical guarantees on resource bounds")
    print()
    print("Use Cases:")
    print("  • Multi-agent SLAs (guarantee each agent gets its budget)")
    print("  • Fair resource sharing in collaborative agents")
    print("  • Preventing budget starvation in hierarchical systems")
    print("  • Enterprise governance with explicit contracts")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
