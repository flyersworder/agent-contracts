"""Code execution and testing utilities.

This module provides safe code execution for validating generated solutions
against test cases.

Example:
    >>> from evaluation.code_review_pipeline.execution import execute_code
    >>> result = execute_code(
    ...     code="print(int(input()) * 2)",
    ...     test_input="5",
    ...     expected_output="10",
    ... )
    >>> print(f"Passed: {result.passed}, Output: {result.actual_output}")
"""

import ast
import os
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum

from .tasks import CodeTask


class FailureReason(Enum):
    """Categorizes why code execution failed."""

    NONE = "none"  # Execution succeeded
    SYNTAX_ERROR = "syntax_error"  # Code has syntax errors
    RUNTIME_ERROR = "runtime_error"  # Code crashed during execution
    TIMEOUT = "timeout"  # Code execution timed out
    WRONG_ANSWER = "wrong_answer"  # Code ran but produced wrong output
    TRUNCATION = "truncation"  # Code appears truncated (hit token limit)


@dataclass
class TestResult:
    """Result from executing code against a test case.

    Attributes:
        passed: Whether the test passed
        actual_output: The actual output from the code
        expected_output: The expected output
        failure_reason: Why the test failed (if applicable)
        error_message: Detailed error message
        execution_time_ms: How long execution took
        stderr: Standard error output
    """

    passed: bool
    actual_output: str
    expected_output: str
    failure_reason: FailureReason = FailureReason.NONE
    error_message: str = ""
    execution_time_ms: float = 0.0
    stderr: str = ""


@dataclass
class ExecutionResult:
    """Result from executing code against all test cases.

    Attributes:
        all_passed: Whether all tests passed
        tests_passed: Number of tests that passed
        tests_total: Total number of tests
        test_results: Individual test results
        failure_reason: Primary failure reason
        is_truncated: Whether code appears truncated
    """

    all_passed: bool
    tests_passed: int
    tests_total: int
    test_results: list[TestResult]
    failure_reason: FailureReason = FailureReason.NONE
    is_truncated: bool = False


def check_syntax(code: str) -> tuple[bool, str]:
    """Check if Python code has valid syntax.

    Args:
        code: Python code to check

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


def detect_truncation(
    code: str,
    tokens_used: int | None = None,
    token_limit: int | None = None,
) -> bool:
    """Detect if code was likely truncated due to token limits.

    Heuristics:
    - Syntax error at end of file
    - Unclosed brackets/parentheses/quotes
    - Token usage near limit

    Args:
        code: The generated code
        tokens_used: Optional token count used
        token_limit: Optional token budget limit

    Returns:
        True if code appears truncated
    """
    # Check if near token limit
    if tokens_used and token_limit and tokens_used >= token_limit - 10:
        # Check for syntax validity
        is_valid, _ = check_syntax(code)
        if not is_valid:
            return True

    # Check for unclosed structures
    code = code.strip()

    # Unclosed brackets
    open_brackets = code.count("(") - code.count(")")
    open_squares = code.count("[") - code.count("]")
    open_braces = code.count("{") - code.count("}")

    if open_brackets > 0 or open_squares > 0 or open_braces > 0:
        return True

    # Ends mid-string
    if code.count('"""') % 2 != 0 or code.count("'''") % 2 != 0:
        return True

    # Check for incomplete statements
    incomplete_patterns = [
        "def ",  # Incomplete function
        "class ",  # Incomplete class
        "if ",  # Incomplete if
        "for ",  # Incomplete for
        "while ",  # Incomplete while
    ]
    lines = code.split("\n")
    if lines:
        last_line = lines[-1].strip()
        for pattern in incomplete_patterns:
            if last_line.startswith(pattern) and not last_line.endswith(":"):
                return True

    return False


def clean_code(code: str) -> str:
    """Clean generated code by removing markdown formatting.

    Args:
        code: Raw code that may contain markdown

    Returns:
        Cleaned Python code
    """
    code = code.strip()

    # Remove markdown code blocks
    if code.startswith("```python"):
        code = code[len("```python") :].lstrip()
    elif code.startswith("```"):
        code = code[3:].lstrip()

    if code.endswith("```"):
        code = code[:-3].rstrip()

    return code


def execute_code(
    code: str,
    test_input: str,
    expected_output: str,
    timeout_seconds: float = 5.0,
) -> TestResult:
    """Execute Python code with given input and compare to expected output.

    Args:
        code: Python code to execute
        test_input: Input to provide via stdin
        expected_output: Expected stdout output
        timeout_seconds: Maximum execution time

    Returns:
        TestResult with execution details
    """
    import time

    # Clean the code
    code = clean_code(code)

    # Check syntax first
    is_valid, syntax_error = check_syntax(code)
    if not is_valid:
        return TestResult(
            passed=False,
            actual_output="",
            expected_output=expected_output,
            failure_reason=FailureReason.SYNTAX_ERROR,
            error_message=syntax_error,
        )

    # Write code to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_file = f.name

    start_time = time.time()

    try:
        result = subprocess.run(
            ["python", temp_file],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

        execution_time_ms = (time.time() - start_time) * 1000
        actual_output = result.stdout.strip()
        expected_clean = expected_output.strip()

        if result.returncode != 0:
            return TestResult(
                passed=False,
                actual_output=actual_output,
                expected_output=expected_clean,
                failure_reason=FailureReason.RUNTIME_ERROR,
                error_message=f"Exit code {result.returncode}: {result.stderr[:500]}",
                execution_time_ms=execution_time_ms,
                stderr=result.stderr,
            )

        if actual_output == expected_clean:
            return TestResult(
                passed=True,
                actual_output=actual_output,
                expected_output=expected_clean,
                failure_reason=FailureReason.NONE,
                execution_time_ms=execution_time_ms,
            )
        else:
            return TestResult(
                passed=False,
                actual_output=actual_output,
                expected_output=expected_clean,
                failure_reason=FailureReason.WRONG_ANSWER,
                error_message=f"Expected: {expected_clean[:100]}, Got: {actual_output[:100]}",
                execution_time_ms=execution_time_ms,
            )

    except subprocess.TimeoutExpired:
        return TestResult(
            passed=False,
            actual_output="",
            expected_output=expected_output.strip(),
            failure_reason=FailureReason.TIMEOUT,
            error_message=f"Execution timed out after {timeout_seconds}s",
            execution_time_ms=timeout_seconds * 1000,
        )

    except Exception as e:
        return TestResult(
            passed=False,
            actual_output="",
            expected_output=expected_output.strip(),
            failure_reason=FailureReason.RUNTIME_ERROR,
            error_message=str(e),
        )

    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def execute_task(
    code: str,
    task: CodeTask,
    tokens_used: int | None = None,
    token_limit: int | None = None,
) -> ExecutionResult:
    """Execute code against all test cases for a task.

    Args:
        code: Python code to execute
        task: CodeTask with test cases
        tokens_used: Optional token count for truncation detection
        token_limit: Optional token limit for truncation detection

    Returns:
        ExecutionResult with all test results
    """
    # Check for truncation
    is_truncated = detect_truncation(code, tokens_used, token_limit)

    # Run all test cases
    test_results: list[TestResult] = []
    for test_case in task.test_cases:
        result = execute_code(
            code=code,
            test_input=test_case.input,
            expected_output=test_case.expected_output,
        )

        # Override failure reason if truncated
        if is_truncated and result.failure_reason == FailureReason.SYNTAX_ERROR:
            result = TestResult(
                passed=result.passed,
                actual_output=result.actual_output,
                expected_output=result.expected_output,
                failure_reason=FailureReason.TRUNCATION,
                error_message=result.error_message,
                execution_time_ms=result.execution_time_ms,
                stderr=result.stderr,
            )

        test_results.append(result)

    # Summarize results
    tests_passed = sum(1 for r in test_results if r.passed)
    all_passed = tests_passed == len(test_results)

    # Determine primary failure reason
    if all_passed:
        failure_reason = FailureReason.NONE
    elif is_truncated:
        failure_reason = FailureReason.TRUNCATION
    else:
        # Use the first failure's reason
        for r in test_results:
            if not r.passed:
                failure_reason = r.failure_reason
                break
        else:
            failure_reason = FailureReason.WRONG_ANSWER

    return ExecutionResult(
        all_passed=all_passed,
        tests_passed=tests_passed,
        tests_total=len(test_results),
        test_results=test_results,
        failure_reason=failure_reason,
        is_truncated=is_truncated,
    )


def format_test_result(task: CodeTask, result: ExecutionResult) -> str:
    """Format execution result as feedback for the coder.

    Args:
        task: The coding task
        result: Execution result

    Returns:
        Formatted feedback string
    """
    if result.all_passed:
        return f"PASS: All {result.tests_total} tests passed!"

    feedback = f"FAIL: {result.tests_passed}/{result.tests_total} tests passed.\n\n"

    # Find first failure
    for i, (tc, tr) in enumerate(zip(task.test_cases, result.test_results, strict=True)):
        if not tr.passed:
            feedback += f"**Failed Test Case {i + 1}:**\n"
            feedback += f"- Input: {tc.input[:100]}{'...' if len(tc.input) > 100 else ''}\n"
            feedback += f"- Expected: {tr.expected_output[:100]}{'...' if len(tr.expected_output) > 100 else ''}\n"
            feedback += f"- Actual: {tr.actual_output[:100]}{'...' if len(tr.actual_output) > 100 else ''}\n"
            feedback += f"- Reason: {tr.failure_reason.value}\n"

            if tr.error_message:
                feedback += f"- Error: {tr.error_message[:200]}\n"

            break  # Only show first failure

    return feedback
