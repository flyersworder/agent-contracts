"""Claude Agent SDK integration for Agent Contracts.

This module provides contract-aware wrappers for Claude Agent SDK agents,
enabling resource governance, per-tool enforcement, temporal constraints,
and audit trails via the SDK's hook system.

All SDK features (tools, MCP servers, subagents, skills, permissions)
remain fully available — the contract wraps on top, not replacing anything.

Example:
    >>> from agent_contracts import Contract, ResourceConstraints
    >>> from agent_contracts.integrations.claude_agent_sdk import ContractedClaudeAgent
    >>>
    >>> contract = Contract(
    ...     id="my-agent",
    ...     resources=ResourceConstraints(tokens=50000, cost_usd=2.0, iterations=10)
    ... )
    >>> contracted = ContractedClaudeAgent(
    ...     contract=contract,
    ...     prompt="Review auth.py",
    ... )
    >>> result = await contracted.aexecute()
"""

from datetime import datetime
from typing import Any

from agent_contracts.core.contract import Contract, ContractState
from agent_contracts.core.enforcement import ContractEnforcer, EnforcementEvent
from agent_contracts.core.monitor import ResourceMonitor, TemporalMonitor
from agent_contracts.core.wrapper import ExecutionLog, ExecutionResult

try:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        HookMatcher,
        ResultMessage,
        query,
    )

    CLAUDE_AGENT_SDK_AVAILABLE = True
except ImportError:
    CLAUDE_AGENT_SDK_AVAILABLE = False
    ClaudeAgentOptions = Any  # type: ignore
    HookMatcher = Any  # type: ignore
    AssistantMessage = Any  # type: ignore
    ResultMessage = Any  # type: ignore
    query = Any  # type: ignore


class ContractedClaudeAgent:
    """Contract-governed wrapper for Claude Agent SDK agents.

    Wraps the Claude Agent SDK's query function with contract enforcement,
    mapping contract constraints to SDK options and injecting enforcement
    hooks for per-tool and budget governance.

    Attributes:
        contract: The contract governing this agent's execution
        prompt: The prompt/task to execute
        strict_mode: If True, violations immediately terminate execution
    """

    def __init__(
        self,
        contract: Contract,
        prompt: str,
        options: "ClaudeAgentOptions | None" = None,
        strict_mode: bool = True,
    ) -> None:
        """Initialize the ContractedClaudeAgent.

        Args:
            contract: Contract defining resource and temporal constraints
            prompt: The prompt or task to pass to the Claude agent
            options: Optional ClaudeAgentOptions to merge with contract constraints
            strict_mode: If True, constraint violations halt execution immediately
        """
        self.contract = contract
        self.prompt = prompt
        self.strict_mode = strict_mode
        self._user_options = options

        # Set up monitoring and enforcement
        self._resource_monitor = ResourceMonitor(contract.resources)
        self._temporal_monitor = TemporalMonitor(contract)
        self._enforcer = ContractEnforcer(
            contract,
            strict_mode=strict_mode,
            monitor=self._resource_monitor,
        )

    def _build_options(self) -> "ClaudeAgentOptions":
        """Merge contract constraints into ClaudeAgentOptions.

        Contract constraints take the following precedence rules:
        - For numeric limits (max_turns, max_budget_usd): use min of contract and user
        - For lists (allowed_tools): merge both sets
        - For strings (system_prompt): contract instructions prepended to user's
        - All other user options passed through unchanged

        Returns:
            A new ClaudeAgentOptions with contract constraints applied
        """
        # Build kwargs dict to avoid passing None for unset fields
        kwargs: dict[str, Any] = {}

        # Start with user options if provided
        if self._user_options is not None:
            user = self._user_options
            if user.permission_mode is not None:
                kwargs["permission_mode"] = user.permission_mode
            if user.model is not None:
                kwargs["model"] = user.model
            if user.fallback_model is not None:
                kwargs["fallback_model"] = user.fallback_model
            if user.max_turns is not None:
                kwargs["max_turns"] = user.max_turns
            if user.max_budget_usd is not None:
                kwargs["max_budget_usd"] = user.max_budget_usd
            if user.system_prompt is not None:
                kwargs["system_prompt"] = user.system_prompt
            if user.cwd is not None:
                kwargs["cwd"] = user.cwd
            if user.allowed_tools:
                kwargs["allowed_tools"] = list(user.allowed_tools)
            if user.disallowed_tools:
                kwargs["disallowed_tools"] = list(user.disallowed_tools)
            if user.mcp_servers:
                kwargs["mcp_servers"] = user.mcp_servers
            if user.agents is not None:
                kwargs["agents"] = user.agents
            if user.hooks:
                kwargs["hooks"] = user.hooks

        # Map contract.resources.iterations → max_turns (more restrictive wins)
        if self.contract.resources.iterations is not None:
            contract_max_turns = self.contract.resources.iterations
            if "max_turns" in kwargs:
                kwargs["max_turns"] = min(kwargs["max_turns"], contract_max_turns)
            else:
                kwargs["max_turns"] = contract_max_turns

        # Map contract.resources.cost_usd → max_budget_usd (more restrictive wins)
        if self.contract.resources.cost_usd is not None:
            contract_budget = self.contract.resources.cost_usd
            if "max_budget_usd" in kwargs:
                kwargs["max_budget_usd"] = min(kwargs["max_budget_usd"], contract_budget)
            else:
                kwargs["max_budget_usd"] = contract_budget

        # Merge capabilities.tools → allowed_tools
        if self.contract.capabilities is not None and self.contract.capabilities.tools:
            existing_tools = list(kwargs.get("allowed_tools", []))
            contract_tools = list(self.contract.capabilities.tools)
            merged_tools = list(dict.fromkeys(existing_tools + contract_tools))
            kwargs["allowed_tools"] = merged_tools

        # Merge capabilities.instructions → prepend to system_prompt
        if (
            self.contract.capabilities is not None
            and self.contract.capabilities.instructions is not None
        ):
            contract_instructions = self.contract.capabilities.instructions
            existing_prompt = kwargs.get("system_prompt")
            if existing_prompt:
                kwargs["system_prompt"] = f"{contract_instructions}\n\n{existing_prompt}"
            else:
                kwargs["system_prompt"] = contract_instructions

        # Build enforcement hooks
        existing_hooks = kwargs.pop("hooks", None)
        kwargs["hooks"] = self._build_hooks(existing_hooks)

        return ClaudeAgentOptions(**kwargs)

    def _build_hooks(
        self,
        existing_hooks: "dict[str, list[HookMatcher]] | None",
    ) -> "dict[str, list[HookMatcher]]":
        """Merge contract enforcement hooks with user-provided hooks.

        Inserts PreToolUse and PostToolUse HookMatcher placeholders that
        route through the contract enforcement layer.

        Args:
            existing_hooks: Any hooks already set by the user

        Returns:
            Merged hooks dict with enforcement hooks added
        """
        hooks: dict[str, list[Any]] = {}

        # Copy existing user hooks
        if existing_hooks:
            for event_type, matchers in existing_hooks.items():
                hooks[event_type] = list(matchers)

        # Add PreToolUse enforcement hook
        pre_hook_matcher = HookMatcher(
            matcher=None,
            hooks=[self._pre_tool_use_hook],
        )
        hooks.setdefault("PreToolUse", [])
        hooks["PreToolUse"].append(pre_hook_matcher)

        # Add PostToolUse enforcement hook
        post_hook_matcher = HookMatcher(
            matcher=None,
            hooks=[self._post_tool_use_hook],
        )
        hooks.setdefault("PostToolUse", [])
        hooks["PostToolUse"].append(post_hook_matcher)

        return hooks

    async def _pre_tool_use_hook(
        self, hook_input: Any, session_id: Any, context: Any
    ) -> dict[str, Any]:
        """Pre-tool-use enforcement hook.

        Called before each tool invocation. Enforces per-tool limits,
        aggregate tool invocation budget, web search limits, and temporal
        constraints. Returns a block decision if any constraint is exceeded.

        Args:
            hook_input: PreToolUseHookInput from the SDK
            session_id: Current session identifier
            context: HookContext from the SDK

        Returns:
            Empty dict to allow, or {"decision": "block", "reason": "..."} to block
        """
        tool_name = hook_input.get("tool_name", "unknown")

        # 1. Check per-tool limits and aggregate tool invocations
        if not self._resource_monitor.can_use_tool(tool_name):
            return {"decision": "block", "reason": f"Tool limit exceeded for '{tool_name}'"}

        # 2. Check web search limit
        constraints = self.contract.resources
        if (
            tool_name == "WebSearch"
            and constraints.web_searches is not None
            and self._resource_monitor.usage.web_searches >= constraints.web_searches
        ):
            return {"decision": "block", "reason": "WebSearch limit exceeded"}

        # 4. Check temporal constraints
        if self._temporal_monitor.is_over_duration() or self._temporal_monitor.is_past_deadline():
            return {"decision": "block", "reason": "Contract temporal limit exceeded"}

        return {}

    async def _post_tool_use_hook(
        self, hook_input: Any, session_id: Any, context: Any
    ) -> dict[str, Any]:
        """Post-tool-use audit hook.

        Called after each tool invocation completes. Tracks tool usage counts
        (aggregate and per-tool), web search counts, and emits an enforcement
        event for the audit trail.

        Args:
            hook_input: PostToolUseHookInput from the SDK
            session_id: Current session identifier
            context: HookContext from the SDK

        Returns:
            Empty dict (no blocking action)
        """
        tool_name = hook_input.get("tool_name", "unknown")

        # Track tool usage
        self._resource_monitor.usage.tool_invocations += 1
        self._resource_monitor.usage.tool_usage_by_name[tool_name] = (
            self._resource_monitor.usage.tool_usage_by_name.get(tool_name, 0) + 1
        )

        # Track web searches
        if tool_name == "WebSearch":
            self._resource_monitor.usage.web_searches += 1

        # Emit enforcement event for audit trail
        self._enforcer._emit_event(
            EnforcementEvent(
                event_type="tool_use",
                contract=self.contract,
                message=f"Tool '{tool_name}' executed",
                data={
                    "tool_name": tool_name,
                    "tool_use_id": hook_input.get("tool_use_id", ""),
                    "agent_id": hook_input.get("agent_id", ""),
                    "agent_type": hook_input.get("agent_type", ""),
                },
            )
        )

        return {}

    async def aexecute(self) -> "ExecutionResult[str]":
        """Execute the agent asynchronously within contract constraints.

        Streams messages from the Claude Agent SDK's query function,
        tracking token usage per AssistantMessage and capturing the final
        result from ResultMessage.

        Returns:
            ExecutionResult with output, success status, violations, and audit log
        """
        start_time = datetime.now()
        output: str | None = None
        violations: list[str] = []
        events: list[dict[str, Any]] = []

        # Start monitoring
        self._temporal_monitor.start()
        if not self._enforcer._enforcement_active:
            self._enforcer.start()  # Also activates contract (DRAFTED → ACTIVE)

        merged_options = self._build_options()

        try:
            async for message in query(prompt=self.prompt, options=merged_options):
                if isinstance(message, AssistantMessage):
                    usage = message.usage
                    if usage:
                        token_count = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                        self._resource_monitor.usage.add_tokens(token_count)
                        self._resource_monitor.usage.api_calls += 1

                        # Check constraints after each AssistantMessage
                        constraint_violations = self._resource_monitor.check_constraints()
                        if constraint_violations:
                            for v in constraint_violations:
                                msg = f"{v.resource}: {v.actual} > {v.limit}"
                                violations.append(msg)
                                events.append(
                                    {
                                        "type": "constraint_violated",
                                        "message": msg,
                                        "timestamp": datetime.now().isoformat(),
                                    }
                                )
                            if self.strict_mode:
                                break

                elif isinstance(message, ResultMessage):
                    output = message.result

        except Exception as e:
            violations.append(str(e))
            events.append(
                {
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            )

        end_time = datetime.now()
        success = len(violations) == 0 and output is not None

        # Update contract state
        if violations and self.contract.state == ContractState.ACTIVE:
            self.contract.violate()
        # else: keep ACTIVE for cumulative tracking

        execution_log = ExecutionLog(
            contract_id=self.contract.id,
            start_time=start_time,
            end_time=end_time,
            final_state=self.contract.state,
            resource_usage=self._resource_monitor.usage.to_dict(),
            temporal_metrics={
                "elapsed_seconds": (end_time - start_time).total_seconds(),
                "deadline_met": not self._temporal_monitor.is_past_deadline(),
            },
            events=events,
            metadata={},
        )

        return ExecutionResult(
            output=output,
            contract=self.contract,
            success=success,
            violations=violations,
            execution_log=execution_log,
            metadata={"elapsed_seconds": (end_time - start_time).total_seconds()},
        )

    def execute(self) -> "ExecutionResult[str]":
        """Execute the agent synchronously by wrapping aexecute().

        Handles both running and non-running event loops gracefully.

        Returns:
            ExecutionResult with output, success status, violations, and audit log
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.aexecute())
                return future.result()
        else:
            return asyncio.run(self.aexecute())
