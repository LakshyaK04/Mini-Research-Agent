"""
tools.py — Tools for the Mini Research Agent

Two tools:
1. search_web  — Uses Tavily to search the internet for current information
2. calculator  — Safe arithmetic evaluator using Python's ast module
"""

import ast
import operator as op

from langchain_core.tools import tool


# ──────────────────────────────────────────────
#  Tool 1: Web Search (Tavily)
# ──────────────────────────────────────────────

# Lazy singleton — Tavily validates the API key at construction time,
# so we delay creation until the tool is actually called.
_tavily_client = None


def _get_tavily():
    """Return (and cache) the TavilySearch client."""
    global _tavily_client
    if _tavily_client is None:
        from langchain_tavily import TavilySearch
        _tavily_client = TavilySearch(max_results=3, search_depth="basic")
    return _tavily_client


@tool
def search_web(query: str) -> str:
    """Search the web for current information.

    Use this when you need up-to-date facts, statistics, news, or any
    information you don't confidently know.

    Args:
        query: The search query string.

    Returns:
        Search results with titles, URLs, and content snippets.
    """
    try:
        results = _get_tavily().invoke({"query": query})
        # Format and truncate snippet length to stay within Groq free-tier TPM limits (6000 tokens)
        res_str = str(results)
        if len(res_str) > 2000:
            res_str = res_str[:2000] + "... [truncated for brevity]"
        return res_str
    except Exception as e:
        return f"Search failed: {e}. Try answering with the information you already have."


# ──────────────────────────────────────────────
#  Tool 2: Calculator (safe arithmetic via ast)
# ──────────────────────────────────────────────

# Only these operations are allowed — no exec / eval of arbitrary code
_ALLOWED_OPS = {
    ast.Add:  op.add,
    ast.Sub:  op.sub,
    ast.Mult: op.mul,
    ast.Div:  op.truediv,
    ast.Mod:  op.mod,
    ast.Pow:  op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """Walk an AST tree and evaluate only allowed arithmetic nodes."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        op_func = _ALLOWED_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_func(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_func = _ALLOWED_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_func(_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Perform arithmetic calculations safely.

    Use this for any math instead of calculating in your head.
    Supports: +, -, *, /, % (modulo), ** (power), and parentheses.

    Args:
        expression: A math expression, e.g. '25 * 48' or '(85000 - 70000) / 70000 * 100'

    Returns:
        The numeric result as a string.
    """
    try:
        # Strip commas (e.g. 38,140,000 -> 38140000) so commas don't get parsed as Python tuples
        clean_expr = expression.replace(",", "").strip()
        tree = ast.parse(clean_expr, mode="eval")
        result = _safe_eval(tree)
        # Format nicely: drop .0 for whole numbers
        if isinstance(result, float) and result == int(result) and abs(result) < 1e15:
            return str(int(result))
        return str(result)
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError) as e:
        return f"Calculator error: {e}"
