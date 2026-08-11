import ast
import operator as op
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

# Safe arithmetic operators
_ALLOWED_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Mod: op.mod, ast.Pow: op.pow,
    ast.USub: op.neg, ast.UAdd: op.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    elif isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported math expression")


@tool
def search_web(query: str) -> str:
    """Search the web for current information."""
    try:
        tavily = TavilySearch(max_results=3, search_depth="basic")
        res = str(tavily.invoke({"query": query}))
        return res[:2000] if len(res) > 2000 else res
    except Exception as e:
        return f"Search failed: {e}"


@tool
def calculator(expression: str) -> str:
    """Perform simple arithmetic calculations."""
    try:
        clean = expression.replace(",", "").strip()
        tree = ast.parse(clean, mode="eval")
        res = _safe_eval(tree)
        return str(int(res)) if isinstance(res, float) and res == int(res) else str(res)
    except Exception as e:
        return f"Calculator error: {e}"
