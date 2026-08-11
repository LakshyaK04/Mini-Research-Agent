from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tools import search_web, calculator

load_dotenv()

TOOLS = [search_web, calculator]

SYSTEM_PROMPT = (
    "You are a helpful research assistant with access to two tools:\n\n"
    "1. search_web — Use ONLY when the user needs current, changing, specific, "
    "or externally verified information. Do NOT use search_web for general knowledge "
    "or conceptual questions (e.g., 'explain what a transformer is') that you can answer "
    "confidently from existing knowledge.\n\n"
    "2. calculator — Use for all mathematical calculations. Calculator expressions "
    "MUST use Python-style arithmetic. Use ** for powers/exponents, NEVER ^.\n\n"
    "Guidelines:\n"
    "- For questions requiring both current information and math, search FIRST to find "
    "the exact figures, then use the calculator in the next step.\n"
    "- If a tool returns an ERROR, do NOT estimate or calculate in your head. "
    "Correct the tool input and invoke the tool again.\n"
    "- When you have enough verified information, answer directly without calling tools."
)


def get_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0)


def get_llm_with_tools():
    return get_llm().bind_tools(TOOLS)
