#!/usr/bin/env python3
"""Google ADK Integration Demo: Honest Value Demonstration

This demo shows what Agent Contracts ACTUALLY provides when integrated with Google ADK,
with honest acknowledgment of limitations.

What Works:
✅ Token tracking with detailed breakdown (prompt, response, thinking tokens)
✅ Complete audit trails for compliance
✅ Multi-turn conversation budget protection
✅ Multi-agent system governance
✅ Tool execution monitoring
✅ Organizational policy enforcement

What Doesn't Work (Yet):
⚠️  Single-turn prevention (can't know tokens before API completes)

Focus: Governance, compliance, and multi-turn/multi-agent protection.
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
from agent_contracts.integrations.google_adk import (  # noqa: E402
    ContractedAdkAgent,
    create_contracted_adk_agent,
)


def print_section(title: str) -> None:
    """Print section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def demo_1_token_tracking() -> None:
    """Demo 1: Detailed Token Tracking & Cost Monitoring.

    Shows that Agent Contracts accurately tracks tokens with breakdown:
    - Prompt tokens (input)
    - Candidate tokens (output)
    - Thinking tokens (reasoning)
    - Cached tokens
    """
    print_section("Demo 1: Detailed Token Tracking & Cost Monitoring")

    # Create simple agent
    agent = LlmAgent(
        name="explainer",
        model="gemini-2.0-flash",
        instruction="You are a helpful assistant. Keep responses brief (2-3 sentences).",
    )

    # Create contract
    contract = Contract(
        id="token-tracking",
        name="Token Tracking Demo",
        resources=ResourceConstraints(tokens=10000, api_calls=10),
    )

    contracted_agent = ContractedAdkAgent(contract=contract, agent=agent)

    # Execute
    print("Asking: 'Explain quantum computing briefly'\n")
    result = contracted_agent.run(
        user_id="demo_user",
        session_id="demo_session",
        message="Explain quantum computing briefly",
    )

    print("✅ Execution successful")
    print(f"\n📝 Response: {result['response'][:100]}...")

    usage = result["usage_metadata"]
    print("\n📊 Detailed Token Tracking:")
    print(f"   Total Tokens:      {usage['total_tokens']:,}")
    print(f"   Prompt Tokens:     {usage['prompt_tokens']:,}")
    print(f"   Response Tokens:   {usage['candidates_tokens']:,}")
    print(f"   Thinking Tokens:   {usage['thoughts_tokens']:,}")
    print(f"   Cached Tokens:     {usage['cached_tokens']:,}")

    print("\n💡 Value: Automatic detailed tracking without manual instrumentation")
    print("   • Breakdown by token type for optimization")
    print("   • Cost estimation per API call")
    print("   • Zero developer effort")


def demo_2_multi_turn_protection() -> None:
    """Demo 2: Multi-Turn Conversation Protection.

    Shows budget enforcement across multiple turns in a conversation.
    This is critical for chatbots and interactive agents.
    """
    print_section("Demo 2: Multi-Turn Conversation Budget Protection")

    agent = LlmAgent(
        name="chat_agent",
        model="gemini-2.0-flash",
        instruction="You are a brief conversational assistant. Keep all responses to 1 sentence.",
    )

    # Tight budget for demonstration
    contract = Contract(
        id="multi-turn-protection",
        name="Multi-Turn Protection Demo",
        resources=ResourceConstraints(
            tokens=300,  # Very tight budget
            api_calls=3,
        ),
    )

    contracted_agent = ContractedAdkAgent(contract=contract, agent=agent, strict_mode=False)

    messages = [
        "Hi, how are you?",
        "Tell me about AI",
        "What about machine learning?",
        "Explain deep learning",
    ]

    print("Budget: 300 tokens, 3 API calls")
    print("Making multiple conversation turns...\n")

    session_id = "multi-turn-demo"
    for i, message in enumerate(messages, 1):
        try:
            result = contracted_agent.run(
                user_id="demo_user", session_id=session_id, message=message
            )

            usage = result["usage_metadata"]
            print(f"Turn {i}: '{message}'")
            print(f"  Total Tokens: {usage['total_tokens']:,} | API Calls: {i}")
            print(f"  Response: {result['response'][:60]}...")

            # Check if we're close to budget
            if usage["total_tokens"] > 250:
                print("  ⚠️  Approaching token limit!")
            else:
                print("  ✅ Within budget")

        except RuntimeError as e:
            print(f"\nTurn {i}: BUDGET EXCEEDED")
            print(f"  🛑 Execution stopped: {e}")
            break

        print()

    print("💡 Value: Multi-turn protection prevents runaway costs in conversations")
    print("   • Cumulative tracking across turns")
    print("   • Session-aware budgeting")
    print("   • Prevents infinite loops")


def demo_3_multi_agent_governance() -> None:
    """Demo 3: Multi-Agent System Governance.

    Shows budget enforcement across multiple coordinating agents.
    This is WHERE THE REAL VALUE IS for complex agentic workflows.
    """
    print_section("Demo 3: Multi-Agent System Budget Governance")

    # Create sub-agents
    researcher = LlmAgent(
        name="researcher",
        model="gemini-2.0-flash",
        instruction="You research topics. Keep responses to 1 sentence.",
    )

    summarizer = LlmAgent(
        name="summarizer",
        model="gemini-2.0-flash",
        instruction="You summarize information in 1 sentence.",
    )

    # Create coordinator
    coordinator = LlmAgent(
        name="coordinator",
        model="gemini-2.0-flash",
        instruction="You coordinate research and summarization. Be very brief.",
        sub_agents=[researcher, summarizer],
    )

    # Single budget for ENTIRE multi-agent system
    contract = Contract(
        id="multi-agent-governance",
        name="Multi-Agent Governance Demo",
        resources=ResourceConstraints(
            tokens=1000,  # For ALL agents combined
            api_calls=5,
            cost_usd=0.01,
        ),
    )

    contracted_system = ContractedAdkAgent(contract=contract, agent=coordinator, strict_mode=False)

    print("Multi-agent system: Coordinator -> [Researcher, Summarizer]")
    print("Shared budget: 1000 tokens, 5 API calls, $0.01")
    print("\nExecuting coordinated task...\n")

    try:
        result = contracted_system.run(
            user_id="demo_user",
            session_id="multi-agent-demo",
            message="Research and summarize what quantum computing is",
        )

        usage = result["usage_metadata"]
        print("✅ Multi-agent execution completed")
        print(f"\n📝 Final result: {result['response'][:100]}...")
        print("\n📊 Total Resource Usage (ALL agents):")
        print(f"   Total Tokens:  {usage['total_tokens']:,}")
        print(f"   Events:        {len(result['events'])}")

        print("\n💡 Value: Single budget governance across complex agent hierarchies")
        print("   • Prevents budget explosion from agent coordination")
        print("   • Tracks cumulative usage across all agents")
        print("   • Critical for production multi-agent systems")

    except RuntimeError as e:
        print(f"❌ Budget exceeded: {e}")
        print("🛑 Multi-agent system stopped before completion")


def demo_4_audit_trail() -> None:
    """Demo 4: Complete Audit Trails.

    Shows comprehensive execution logging for compliance and debugging.
    """
    print_section("Demo 4: Complete Audit Trails for Compliance")

    agent = LlmAgent(
        name="audited_agent",
        model="gemini-2.0-flash",
        instruction="You are a helpful assistant. Be brief.",
    )

    contract = Contract(
        id="audit-demo",
        name="Audit Trail Demo",
        resources=ResourceConstraints(tokens=10000, cost_usd=1.0),
    )

    contracted_agent = ContractedAdkAgent(contract=contract, agent=agent, enable_logging=True)

    # Execute with logging
    print("Executing with full audit logging...\n")
    result_exec = contracted_agent.execute(
        {
            "user_id": "audit_user",
            "session_id": "audit_session",
            "message": "What is blockchain?",
        }
    )

    if result_exec.success and result_exec.execution_log:
        log = result_exec.execution_log
        duration = (log.end_time - log.start_time).total_seconds() if log.end_time else 0

        print("✅ Execution logged")
        print("\n📋 Comprehensive Audit Log:")
        print(f"   Contract ID:   {log.contract_id}")
        print(f"   Start Time:    {log.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Duration:      {duration:.3f}s")
        print(f"   Final State:   {log.final_state.value}")
        print(f"   Tokens Used:   {log.resource_usage['tokens']:,}")
        print(f"   API Calls:     {log.resource_usage['api_calls']}")
        print(f"   Cost:          ${log.resource_usage['cost_usd']:.6f}")

        print("\n💡 Value: Complete audit trail for:")
        print("   • Compliance documentation (SOC2, GDPR, etc.)")
        print("   • Cost attribution by contract/user/session")
        print("   • Debugging and performance optimization")
        print("   • Historical analysis and reporting")


def demo_5_convenience_api() -> None:
    """Demo 5: Convenience API for Quick Setup.

    Shows simplified API for creating contracted agents.
    """
    print_section("Demo 5: Simplified Convenience API")

    agent = LlmAgent(
        name="simple_agent",
        model="gemini-2.0-flash",
        instruction="You are helpful and brief.",
    )

    print("Creating contracted agent with simplified API:\n")
    print("contracted = create_contracted_adk_agent(")
    print("    agent=agent,")
    print("    resources={'tokens': 50000, 'cost_usd': 2.0},")
    print("    temporal={'max_duration': 600}")
    print(")\n")

    # Use convenience function
    contracted = create_contracted_adk_agent(
        agent=agent,
        resources={"tokens": 50000, "cost_usd": 2.0, "api_calls": 25},
        temporal={"max_duration": 600},  # 10 minutes
        contract_id="simple-demo",
    )

    result = contracted.run_debug("What is AI?")

    print("✅ Agent created and executed")
    print(f"   Response: {result['response'][:60]}...")
    print(f"   Tokens: {result['total_tokens']}")

    print("\n💡 Value: Quick setup without boilerplate")
    print("   • 3 lines of code vs. manual contract creation")
    print("   • Type-safe resource constraints")
    print("   • Ideal for notebooks and experiments")


def main() -> None:
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("  Agent Contracts: Google ADK Integration Demo")
    print("=" * 80)
    print("\nDemonstrating governance and compliance capabilities")
    print("Model: Google Gemini 2.0 Flash")
    print("Framework: Google Agent Development Kit (ADK)")
    print("=" * 80)

    demos = [
        ("Token Tracking", demo_1_token_tracking),
        ("Multi-Turn Protection", demo_2_multi_turn_protection),
        ("Multi-Agent Governance", demo_3_multi_agent_governance),
        ("Audit Trails", demo_4_audit_trail),
        ("Convenience API", demo_5_convenience_api),
    ]

    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n❌ Demo '{name}' failed: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print_section("✨ Summary")
    print("Agent Contracts + Google ADK provides:")
    print()
    print("1. ✅ DETAILED TOKEN TRACKING")
    print("   Automatic extraction with breakdown (prompt/response/thinking)")
    print()
    print("2. ✅ MULTI-TURN PROTECTION")
    print("   Budget enforcement across conversation turns")
    print()
    print("3. ✅ MULTI-AGENT GOVERNANCE")
    print("   Single budget for complex agent hierarchies")
    print()
    print("4. ✅ AUDIT TRAILS")
    print("   Complete execution logs for compliance")
    print()
    print("5. ✅ ZERO-EFFORT INTEGRATION")
    print("   Wrap existing agents with 2-3 lines of code")
    print()
    print("⚠️  Limitation: Single-turn prevention not possible")
    print("   (tokens unknown until after API completes)")
    print()
    print("💡 Best for:")
    print("   • Enterprise governance and compliance")
    print("   • Multi-agent systems and complex workflows")
    print("   • Cost control for production deployments")
    print("   • Organizations using Google ADK")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
