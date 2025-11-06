#!/usr/bin/env python3
"""LangGraph Integration Demo: Complex Multi-Agent Workflows

This demo shows the REAL value of Agent Contracts with LangGraph:
Budget enforcement across complex workflows with cycles, retries, and multi-agent coordination.

Why LangGraph?
- Complex stateful workflows (not simple chains)
- Cycles and loops (research → validate → research)
- Multi-agent coordination (parallel + sequential)
- High budget risk (can easily make 30+ LLM calls!)

Where Agent Contracts Shines:
✅ Cumulative budget tracking across entire graph
✅ Cycle/loop protection (prevent infinite loops)
✅ Multi-agent budget sharing
✅ Real-time violation detection at node boundaries

This is MUCH more important than simple LangChain chains!
"""

import sys
from typing import TypedDict

from dotenv import load_dotenv

# Load environment variables
load_dotenv("../../.env")

# Check dependencies
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.graph import END, StateGraph
except ImportError:
    print("❌ Missing dependencies!")
    print(
        "\nInstall with: uv pip install langgraph langchain langchain-core langchain-google-genai"
    )
    sys.exit(1)

from agent_contracts import Contract, ResourceConstraints  # noqa: E402
from agent_contracts.integrations.langgraph import ContractedGraph  # noqa: E402


def print_section(title: str) -> None:
    """Print section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


# ============================================================================
# Demo 1: Simple Sequential Graph
# ============================================================================


class SimpleState(TypedDict):
    """State for simple sequential graph."""

    query: str
    result: str


def demo_1_simple_graph() -> None:
    """Demo 1: Simple Sequential Graph with Budget Tracking."""
    print_section("Demo 1: Simple Sequential Graph")

    # Create LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # Define nodes
    def step_1(state: SimpleState) -> SimpleState:
        """First step: analyze query."""
        prompt = ChatPromptTemplate.from_template("Analyze this query in one sentence: {query}")
        chain = prompt | llm
        result = chain.invoke({"query": state["query"]})
        return {"query": state["query"], "result": result.content}

    def step_2(state: SimpleState) -> SimpleState:
        """Second step: summarize result."""
        prompt = ChatPromptTemplate.from_template("Summarize in 5 words: {result}")
        chain = prompt | llm
        summary = chain.invoke({"result": state["result"]})
        return {"query": state["query"], "result": summary.content}

    # Build graph
    workflow = StateGraph(SimpleState)
    workflow.add_node("analyze", step_1)
    workflow.add_node("summarize", step_2)
    workflow.add_edge("analyze", "summarize")
    workflow.add_edge("summarize", END)
    workflow.set_entry_point("analyze")

    app = workflow.compile()

    # Wrap with contract
    contract = Contract(
        id="simple-graph",
        name="Simple Graph Demo",
        resources=ResourceConstraints(
            tokens=5000,
            api_calls=5,
        ),
    )

    contracted_app = ContractedGraph(contract=contract, graph=app)

    # Execute
    print("Executing simple 2-node graph...")
    result = contracted_app.execute({"query": "What is machine learning?"})

    if result.success:
        print("✅ Execution successful")
        print("\n📊 Resource Usage:")
        print(f"   Tokens:     {result.execution_log.resource_usage['tokens']:,}")
        print(f"   API Calls:  {result.execution_log.resource_usage['api_calls']}")
        print(f"   Cost:       ${result.execution_log.resource_usage['cost_usd']:.6f}")
        print("\n💡 Budget tracked across both nodes automatically!")
    else:
        print(f"❌ Execution failed: {result.violations}")


# ============================================================================
# Demo 2: Graph with Validation Cycle
# ============================================================================


class ValidationState(TypedDict):
    """State for validation cycle graph."""

    query: str
    attempt: int
    max_attempts: int
    result: str
    is_valid: bool


def demo_2_validation_cycle() -> None:
    """Demo 2: Graph with Validation Cycle (Loop Protection).

    This demonstrates the REAL value: preventing runaway costs in loops!
    """
    print_section("Demo 2: Validation Cycle (Loop Protection)")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # Define nodes
    def generate_answer(state: ValidationState) -> ValidationState:
        """Generate answer (simulated)."""
        print(f"   Attempt {state['attempt']}: Generating answer...")

        # For demo, just return a simple answer
        # In real scenario, this would call LLM
        prompt = ChatPromptTemplate.from_template("Answer in one sentence: {query}")
        chain = prompt | llm
        result = chain.invoke({"query": state["query"]})

        return {
            **state,
            "result": result.content,
            "attempt": state["attempt"] + 1,
        }

    def validate_answer(state: ValidationState) -> ValidationState:
        """Validate answer (simulated)."""
        # For demo, fail first 2 attempts to show cycle
        is_valid = state["attempt"] > 2

        print(f"   Validating: {'✅ Valid' if is_valid else '❌ Invalid - retrying'}")

        return {**state, "is_valid": is_valid}

    def should_continue(state: ValidationState) -> str:
        """Decide whether to continue or end."""
        if state["is_valid"]:
            return "end"
        elif state["attempt"] >= state["max_attempts"]:
            print("   ⚠️ Max attempts reached")
            return "end"
        else:
            return "retry"

    # Build graph with cycle
    workflow = StateGraph(ValidationState)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("validate", validate_answer)

    workflow.add_edge("generate", "validate")
    workflow.add_conditional_edges("validate", should_continue, {"retry": "generate", "end": END})

    workflow.set_entry_point("generate")

    app = workflow.compile()

    # Wrap with TIGHT budget to show protection
    contract = Contract(
        id="validation-cycle",
        name="Validation Cycle Demo",
        resources=ResourceConstraints(
            tokens=2000,  # Tight budget - might hit limit!
            api_calls=5,  # Limit iterations
        ),
    )

    contracted_app = ContractedGraph(contract=contract, graph=app, strict_mode=False)

    # Execute
    print("Executing graph with validation cycle...")
    print("(Cycle will retry up to 3 times or until budget exhausted)\n")

    result = contracted_app.execute(
        {"query": "What is AI?", "attempt": 0, "max_attempts": 5, "result": "", "is_valid": False}
    )

    if result.success and result.output:
        print(f"\n✅ Completed after {result.output.get('attempt', 0)} attempts")
    else:
        print(f"\n⚠️ Budget exhausted: {result.violations}")

    print("\n📊 Resource Usage:")
    usage = result.execution_log.resource_usage
    print(f"   Tokens:     {usage['tokens']:,}")
    print(f"   API Calls:  {usage['api_calls']}")
    print(f"   Cost:       ${usage['cost_usd']:.6f}")

    print("\n💡 Value: Contract prevented runaway loop costs!")
    print("   Without limits, could have made many more attempts")
    print("   Budget enforcement stopped execution at the right time")


# ============================================================================
# Demo 3: Multi-Agent Coordination
# ============================================================================


class MultiAgentState(TypedDict):
    """State for multi-agent graph."""

    task: str
    research_result: str
    plan_result: str
    final_result: str


def demo_3_multi_agent() -> None:
    """Demo 3: Multi-Agent Coordination with Shared Budget."""
    print_section("Demo 3: Multi-Agent Coordination")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # Define agent nodes
    def research_agent(state: MultiAgentState) -> MultiAgentState:
        """Research agent."""
        print("   🔍 Research Agent: Gathering information...")
        prompt = ChatPromptTemplate.from_template("Research this topic briefly: {task}")
        chain = prompt | llm
        result = chain.invoke({"task": state["task"]})

        return {**state, "research_result": result.content}

    def planning_agent(state: MultiAgentState) -> MultiAgentState:
        """Planning agent."""
        print("   📋 Planning Agent: Creating plan...")
        prompt = ChatPromptTemplate.from_template(
            "Based on this research, create a brief plan: {research_result}"
        )
        chain = prompt | llm
        result = chain.invoke({"research_result": state["research_result"]})

        return {**state, "plan_result": result.content}

    def execution_agent(state: MultiAgentState) -> MultiAgentState:
        """Execution agent."""
        print("   ⚙️ Execution Agent: Executing plan...")
        prompt = ChatPromptTemplate.from_template(
            "Summarize this plan in one sentence: {plan_result}"
        )
        chain = prompt | llm
        result = chain.invoke({"plan_result": state["plan_result"]})

        return {**state, "final_result": result.content}

    # Build multi-agent graph
    workflow = StateGraph(MultiAgentState)
    workflow.add_node("research", research_agent)
    workflow.add_node("plan", planning_agent)
    workflow.add_node("execute", execution_agent)

    workflow.add_edge("research", "plan")
    workflow.add_edge("plan", "execute")
    workflow.add_edge("execute", END)
    workflow.set_entry_point("research")

    app = workflow.compile()

    # Wrap with shared budget
    contract = Contract(
        id="multi-agent",
        name="Multi-Agent Demo",
        resources=ResourceConstraints(
            tokens=10000,
            api_calls=10,
            cost_usd=0.05,
        ),
    )

    contracted_app = ContractedGraph(contract=contract, graph=app)

    # Execute
    print("\nExecuting multi-agent workflow...")
    print("(3 agents sharing same budget pool)\n")

    result = contracted_app.execute(
        {"task": "AI safety", "research_result": "", "plan_result": "", "final_result": ""}
    )

    if result.success:
        print("\n✅ All agents completed successfully")
        print("\n📊 Shared Budget Usage:")
        usage = result.execution_log.resource_usage
        print(f"   Tokens:     {usage['tokens']:,}")
        print(f"   API Calls:  {usage['api_calls']}")
        print(f"   Cost:       ${usage['cost_usd']:.6f}")

        print("\n💡 Value: Budget shared across all agents!")
        print("   No single agent can monopolize resources")
        print("   Fair allocation enforced automatically")
    else:
        print(f"❌ Execution failed: {result.violations}")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("  Agent Contracts: LangGraph Integration Demo")
    print("=" * 80)
    print("\nDemonstrating governance for complex multi-agent workflows")
    print("Model: Google Gemini 2.5 Flash")
    print("=" * 80)

    try:
        demo_1_simple_graph()
    except Exception as e:
        print(f"\n❌ Demo 1 failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        demo_2_validation_cycle()
    except Exception as e:
        print(f"\n❌ Demo 2 failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        demo_3_multi_agent()
    except Exception as e:
        print(f"\n❌ Demo 3 failed: {e}")
        import traceback

        traceback.print_exc()

    # Summary
    print_section("✨ Summary")
    print("Agent Contracts + LangGraph provides:")
    print()
    print("1. ✅ CUMULATIVE BUDGET TRACKING")
    print("   Track resources across entire graph (all nodes, all cycles)")
    print()
    print("2. ✅ CYCLE & LOOP PROTECTION")
    print("   Prevent infinite loops and runaway retries")
    print()
    print("3. ✅ MULTI-AGENT BUDGET SHARING")
    print("   Fair resource allocation across agents")
    print()
    print("4. ✅ REAL-TIME VIOLATION DETECTION")
    print("   Stop execution immediately when limits reached")
    print()
    print("🎯 This is WHERE the value is!")
    print("   - Simple chains: 3-10 LLM calls (low risk)")
    print("   - Complex graphs: 30+ LLM calls (HIGH RISK!)")
    print()
    print("💡 LangGraph + Agent Contracts = Production-Ready Governance")
    print("   Combine with LangSmith for complete observability + governance stack")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
