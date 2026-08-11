"""
graph.py — LangGraph workflow for the Mini Research Agent

Defines:
- AgentState   — the data that flows through the graph
- Nodes        — agent_node, tool_node, evaluate_research, final_answer_node
- Routing      — conditional edges that decide the next step
- create_graph — builds and compiles the full StateGraph
"""

from typing import Annotated, TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent import get_llm, get_llm_with_tools, SYSTEM_PROMPT
from tools import search_web, calculator


# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────

MAX_RESEARCH_STEPS = 3


# ──────────────────────────────────────────────
#  State
# ──────────────────────────────────────────────

class AgentState(TypedDict):
    # Full message history — the add_messages reducer APPENDS
    # new messages instead of overwriting.
    messages: Annotated[list, add_messages]

    # How many research iterations have been performed so far.
    research_count: int

    # Result of the last research evaluation: "sufficient" or "insufficient".
    research_decision: str


# ──────────────────────────────────────────────
#  Structured output for research evaluation
# ──────────────────────────────────────────────

class ResearchEvaluation(BaseModel):
    """Structured decision about whether enough information has been gathered."""
    sufficient: str = Field(
        description="Set to 'YES' if enough information has been gathered "
                    "to fully answer the user's question, or 'NO' if key information is missing."
    )
    reason: str = Field(
        description="One-sentence explanation of the decision."
    )


# ──────────────────────────────────────────────
#  Tool map — name → callable
# ──────────────────────────────────────────────

TOOL_MAP = {
    "search_web": search_web,
    "calculator": calculator,
}


# ──────────────────────────────────────────────
#  Node 1: Agent
# ──────────────────────────────────────────────

def agent_node(state: AgentState) -> dict:
    """Call the LLM with the current messages and tools.

    The LLM will either:
    - request one or more tool calls, OR
    - respond directly (no tools needed).
    """
    llm = get_llm_with_tools()
    response = llm.invoke(state["messages"])

    # ── Deduplicate tool calls if LLM generated identical duplicates ──
    if response.tool_calls:
        seen = set()
        unique_tool_calls = []
        for tc in response.tool_calls:
            key = (tc["name"], str(tc["args"]))
            if key not in seen:
                seen.add(key)
                unique_tool_calls.append(tc)
        response.tool_calls = unique_tool_calls

    # ── Observable output ──
    if response.tool_calls:
        for tc in response.tool_calls:
            name = tc["name"]
            args = tc["args"]
            if name == "search_web":
                print(f'\n🔎 Agent → Web Search')
                print(f'   Query: "{args.get("query", "")}"')
            elif name == "calculator":
                print(f'\n🧮 Agent → Calculator')
                print(f'   Expression: {args.get("expression", "")}')
            else:
                print(f"\n🤖 Agent → {name}({args})")
    else:
        print("\n🤖 Agent → No tools needed")

    return {"messages": [response]}


# ──────────────────────────────────────────────
#  Node 2: Tool Execution
# ──────────────────────────────────────────────

def tool_node(state: AgentState) -> dict:
    """Execute every tool call requested by the agent and return results."""
    last_message = state["messages"][-1]
    results = []

    for tc in last_message.tool_calls:
        name = tc["name"]
        args = tc["args"]

        try:
            func = TOOL_MAP.get(name)
            if func is None:
                content = f"Error: unknown tool '{name}'"
                print(f"   ⚠ Unknown tool: {name}")
            else:
                content = func.invoke(args)
                # Observable output
                if name == "calculator":
                    print(f"   ✓ Result: {content}")
                else:
                    print(f"   ✓ Search completed")
        except Exception as e:
            content = f"Tool error: {e}"
            print(f"   ⚠ {name} failed: {e}")

        results.append(
            ToolMessage(content=str(content), tool_call_id=tc["id"])
        )

    return {"messages": results}


# ──────────────────────────────────────────────
#  Node 3: Evaluate Research
# ──────────────────────────────────────────────

def evaluate_research(state: AgentState) -> dict:
    """Decide whether the gathered information is sufficient.

    Uses structured LLM output (Pydantic) instead of fragile string parsing.
    """
    count = state.get("research_count", 0) + 1

    # ── Hard limit check ──
    if count >= MAX_RESEARCH_STEPS:
        print(f"\n⚠  Max research steps reached ({MAX_RESEARCH_STEPS}).")
        print(f"   Generating answer with available information.")
        return {"research_count": count, "research_decision": "sufficient"}

    # ── Ask LLM to evaluate ──
    eval_llm = get_llm().with_structured_output(ResearchEvaluation, method="function_calling")
    eval_prompt = [
        SystemMessage(
            content=(
                "You are evaluating whether enough information has been "
                "gathered to answer the user's question. Review the full "
                "conversation including all tool results. "
                "Decide if the information is SUFFICIENT (YES) or if critical information is still MISSING (NO). "
                "Note: If the user asked for a fact (e.g. population) and a calculation (e.g. 5%), and BOTH a web search result and a successful calculator result are present in the messages, mark it as SUFFICIENT (YES)."
            )
        ),
        *state["messages"],
    ]

    try:
        decision = eval_llm.invoke(eval_prompt)
        is_sufficient = str(decision.sufficient).strip().upper() in ("YES", "TRUE")
        status = "YES" if is_sufficient else "NO"
        print(f"\n🧠 Research Evaluation (step {count}/{MAX_RESEARCH_STEPS})")
        print(f"   Enough information: {status}")
        print(f"   Reason: {decision.reason}")
        if not is_sufficient:
            feedback = HumanMessage(
                content=(
                    f"RESEARCH EVALUATION: The gathered information is INSUFFICIENT. "
                    f"Reason: {decision.reason}. "
                    f"Please use search_web or calculator to gather the missing details."
                )
            )
            return {
                "messages": [feedback],
                "research_count": count,
                "research_decision": "insufficient",
            }

        return {
            "research_count": count,
            "research_decision": "sufficient",
        }
    except Exception as e:
        # If evaluation itself fails, default to sufficient to avoid loops
        print(f"\n🧠 Research Evaluation: defaulting to sufficient (error: {e})")
        return {"research_count": count, "research_decision": "sufficient"}


# ──────────────────────────────────────────────
#  Node 4: Final Answer
# ──────────────────────────────────────────────

def final_answer_node(state: AgentState) -> dict:
    """Generate the polished final answer using all gathered information."""
    llm = get_llm()  # No tools — just answer generation

    # Clean message history: filter out any empty AIMessages (no text, no tool calls)
    clean_messages = [
        m for m in state["messages"]
        if not (isinstance(m, AIMessage) and not m.content and not getattr(m, "tool_calls", None))
    ]

    # Append a clear instruction to generate the final answer
    synthesis_instruction = HumanMessage(
        content=(
            "Based on all the tool results and research above, write a detailed, "
            "comprehensive final answer to the user's original question. "
            "If calculations were performed, show the key numbers clearly. "
            "If web search was used, include a 'Sources:' section at the end "
            "with numbered source titles and URLs. Do not invent any facts or URLs."
        )
    )

    response = llm.invoke(clean_messages + [synthesis_instruction])

    # ── Clean up LLM output formatting ──
    content = response.content.strip()
    import re
    # Remove repetitive header prefixes if LLM echoed them
    while True:
        cleaned = re.sub(r'^(?:📋\s*FINAL\s*ANSWER|FINAL\s*ANSWER|\*\*Final\s*Answer:\*\*|───+|\-\-\-+)\s*', '', content, flags=re.IGNORECASE).strip()
        if cleaned == content:
            break
        content = cleaned

    # ── Display ──
    print(f"\n{'─' * 40}")
    print(f"\n📋 FINAL ANSWER\n")
    print(content)

    return {"messages": [response]}


# ──────────────────────────────────────────────
#  Routing functions
# ──────────────────────────────────────────────

def route_after_agent(state: AgentState) -> str:
    """After the agent node, check if it requested any tools."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tool_node"
    return "final_answer"


def route_after_evaluation(state: AgentState) -> str:
    """After research evaluation, decide: more research or final answer."""
    if state.get("research_decision") == "insufficient":
        return "agent_node"
    return "final_answer"


# ──────────────────────────────────────────────
#  Build the graph
# ──────────────────────────────────────────────

def create_graph():
    """Construct and compile the LangGraph StateGraph.

    Graph structure:

        START
          │
          ▼
      agent_node ──── tool calls? ──── NO ───► final_answer ──► END
          │                                         ▲
         YES                                        │
          │                                         │
          ▼                                         │
      tool_node                                     │
          │                                         │
          ▼                                         │
    evaluate_research ── sufficient? ── YES ────────┘
          │
          NO
          │
          ▼
      agent_node  (loop back)
    """
    graph = StateGraph(AgentState)

    # ── Add nodes ──
    graph.add_node("agent_node", agent_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("evaluate_research", evaluate_research)
    graph.add_node("final_answer", final_answer_node)

    # ── Edges ──
    graph.add_edge(START, "agent_node")

    graph.add_conditional_edges(
        "agent_node",
        route_after_agent,
        {"tool_node": "tool_node", "final_answer": "final_answer"},
    )

    graph.add_edge("tool_node", "evaluate_research")

    graph.add_conditional_edges(
        "evaluate_research",
        route_after_evaluation,
        {"agent_node": "agent_node", "final_answer": "final_answer"},
    )

    graph.add_edge("final_answer", END)

    return graph.compile()
