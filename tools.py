"""Client-side tools the agent can call, plus the shared tool schema list.

Two of the three tools the agent has (calculator, search_knowledge_base) are
"client-side": Claude asks for them, and our agent loop (agent.py) is the one
that actually runs the Python function and reports the result back. The
third (web_search) is "server-side": Claude and Anthropic's infrastructure
handle the whole round trip, so it never appears in CLIENT_TOOLS below --
see agent.py's loop for how the two kinds are told apart.
"""
import ast
import operator

from rag_tool import search_knowledge_base as _search_knowledge_base

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate a restricted arithmetic AST -- only numbers, the
    operators above, and parentheses are legal. No names, no function calls,
    no attribute access -- so the model can never smuggle arbitrary code
    through the expression string the way a bare eval() would allow.

    For readers new to this: an AST (Abstract Syntax Tree) is what you get
    when Python parses an expression like "2 + 3 * 4" into a tree instead
    of just a string -- e.g. a "+" node with "2" as one branch and a "*"
    node (itself holding "3" and "4") as the other. Walking that tree node
    by node, as this function does, lets us compute the result ourselves
    while only ever allowing the specific node types listed above -- so
    there's no way to hide a function call or file access inside the
    "expression" the model sends us, unlike a plain eval(expression) would."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression component: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Evaluate a plain arithmetic expression and return the result as a string."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree.body))
    except Exception as exc:
        return f"Error: could not evaluate '{expression}' ({exc})"


def search_knowledge_base(query: str) -> str:
    """Search the Nimbus Cloud Storage internal knowledge base and return the top matching snippets."""
    return _search_knowledge_base(query)


# Tool schemas sent to the API. web_search is a server-side tool (Anthropic
# runs it; no Python function of ours is ever called for it), which is why
# it has no entry in CLIENT_TOOLS below.
TOOLS = [
    {
        "name": "calculator",
        "description": (
            "Evaluate a plain arithmetic expression. Use this for any math instead of "
            "computing it yourself -- addition, subtraction, multiplication, division, "
            "powers, modulo, parentheses. Example: '(49.99 * 2) - 15'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Arithmetic expression, e.g. '12 * (3 + 4)'"},
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_knowledge_base",
        "description": (
            "Search Nimbus Cloud Storage's internal policy documents (employee handbook, "
            "security policy, refund policy, product FAQ, incident runbook). Call this "
            "when the question depends on Nimbus-specific policy, pricing, or procedure "
            "-- do not answer those from general knowledge."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The question or topic to search for"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {"type": "web_search_20260209", "name": "web_search"},
]

CLIENT_TOOLS = {
    "calculator": calculator,
    "search_knowledge_base": search_knowledge_base,
}
