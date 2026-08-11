import re
from typing import Annotated, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent import get_llm, get_llm_with_tools
from tools import search_web, calculator

MAX_RESEARCH_STEPS = 3
TOOL_MAP = {"search_web": search_web, "calculator": calculator}


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    research_count: int
    research_decision: str


class ResearchEvaluation(BaseModel):
    sufficient: bool = Field(description="Is the gathered information sufficient?")
    reason: str = Field(description="Reason for decision")


# ── Nodes ──────────────────────────────────────

def agent_node(state: AgentState) -> dict:
    llm = get_llm_with_tools()
    response = llm.invoke(state["messages"])

    # Deduplicate tool calls if LLM repeated any
    if response.tool_calls:
        seen = set()
        unique = []
        for tc in response.tool_calls:
            key = (tc["name"], str(tc["args"]))
            if key not in seen:
                seen.add(key)
                unique.append(tc)
        response.tool_calls = unique

    # Console feedback
    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"-> Agent calling tool: {tc['name']} with args {tc['args']}")
    else:
        print("-> Agent: No tools needed")

    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    last_msg = state["messages"][-1]
    results = []

    for tc in last_msg.tool_calls:
        name = tc["name"]
        args = tc["args"]
        func = TOOL_MAP.get(name)
        res = func.invoke(args) if func else f"Unknown tool: {name}"
        print(f"   [Tool Output] {name}: {str(res)[:100]}...")
        results.append(ToolMessage(content=str(res), tool_call_id=tc["id"]))

    return {"messages": results}


def evaluate_research(state: AgentState) -> dict:
    count = state.get("research_count", 0) + 1

    if count >= MAX_RESEARCH_STEPS:
        print(f"-> Research limit reached ({MAX_RESEARCH_STEPS})")
        return {"research_count": count, "research_decision": "sufficient"}

    eval_llm = get_llm().with_structured_output(ResearchEvaluation, method="function_calling")
    prompt = [
        SystemMessage(content="Evaluate if enough info is gathered. Return sufficient=True/False and reason."),
        *state["messages"],
    ]

    try:
        decision = eval_llm.invoke(prompt)
        print(f"-> Research Evaluation (Step {count}): Sufficient={decision.sufficient} ({decision.reason})")
        if not decision.sufficient:
            feedback = HumanMessage(
                content=f"EVALUATION: Information INSUFFICIENT. Reason: {decision.reason}. Please use tools to get missing info."
            )
            return {"messages": [feedback], "research_count": count, "research_decision": "insufficient"}
        return {"research_count": count, "research_decision": "sufficient"}
    except Exception as e:
        return {"research_count": count, "research_decision": "sufficient"}


def final_answer_node(state: AgentState) -> dict:
    llm = get_llm()
    clean_messages = [
        m for m in state["messages"]
        if not (isinstance(m, AIMessage) and not m.content and not getattr(m, "tool_calls", None))
    ]
    instruction = HumanMessage(content="Based on research above, write a complete final answer. Include a 'Sources:' section at the end if web search was used.")
    response = llm.invoke(clean_messages + [instruction])

    content = response.content.strip()
    while True:
        cleaned = re.sub(r'^(?:📋\s*FINAL\s*ANSWER|FINAL\s*ANSWER|\*\*Final\s*Answer:\*\*|───+|\-\-\-+)\s*', '', content, flags=re.IGNORECASE).strip()
        if cleaned == content:
            break
        content = cleaned

    print("\n" + "=" * 40 + "\nFINAL ANSWER:\n" + "=" * 40)
    print(content)
    return {"messages": [response]}


# ── Routing ────────────────────────────────────

def route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tool_node"
    return "final_answer"


def route_after_evaluation(state: AgentState) -> str:
    if state.get("research_decision") == "insufficient":
        return "agent_node"
    return "final_answer"


# ── Build Graph ────────────────────────────────

def create_graph():
    builder = StateGraph(AgentState)
    builder.add_node("agent_node", agent_node)
    builder.add_node("tool_node", tool_node)
    builder.add_node("evaluate_research", evaluate_research)
    builder.add_node("final_answer", final_answer_node)

    builder.add_edge(START, "agent_node")
    builder.add_conditional_edges("agent_node", route_after_agent, {"tool_node": "tool_node", "final_answer": "final_answer"})
    builder.add_edge("tool_node", "evaluate_research")
    builder.add_conditional_edges("evaluate_research", route_after_evaluation, {"agent_node": "agent_node", "final_answer": "final_answer"})
    builder.add_edge("final_answer", END)

    return builder.compile()
