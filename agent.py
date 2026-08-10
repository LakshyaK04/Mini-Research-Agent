"""
agent.py — LLM configuration and tool binding

Centralizes:
- Environment variable loading
- Model selection
- System prompt
- Tool list and LLM-with-tools factory
"""

import os
import sys
from dotenv import load_dotenv

# Ensure UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Load environment variables FIRST ──────────
# Must happen before importing anything that reads API keys.
load_dotenv()

# ── Map provider key to OpenAI key ───────────
# ChatOpenAI reads OPENAI_API_KEY. We map whichever provider
# key is in .env so everything works seamlessly.
for key_name in ("GROQ_API_KEY", "XAI_API_KEY"):
    if os.environ.get(key_name) and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ[key_name]

from langchain_openai import ChatOpenAI
from tools import search_web, calculator

# ── Tools available to the agent ─────────────
TOOLS = [search_web, calculator]

# ── System prompt ─────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful research assistant. "
    "Your job is to answer the user's question accurately.\n\n"
    "You have access to two tools:\n"
    "1. search_web — Search the internet for current or external information.\n"
    "2. calculator — Perform arithmetic calculations.\n\n"
    "Guidelines:\n"
    "- Use search_web when you need current facts, statistics, news, "
    "or any information you do not confidently know.\n"
    "- Use calculator for any arithmetic — do NOT calculate in your head.\n"
    "- Do NOT use tools when you can confidently answer from general knowledge.\n"
    "- Always output valid JSON parameters when invoking a tool.\n"
    "- Always use information returned by tools — never invent facts or URLs.\n"
    "- When you have enough information, answer the question directly "
    "without calling another tool.\n"
)


# ── Factory functions ─────────────────────────

def get_llm():
    """Return an LLM instance supporting OpenAI, Groq, or xAI based on env variables."""
    if os.environ.get("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    if os.environ.get("XAI_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        return ChatOpenAI(model="grok-3-mini", base_url="https://api.x.ai/v1", temperature=0)

    # Default: OpenAI
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def get_llm_with_tools():
    """Return an LLM instance with both tools bound for tool-calling."""
    return get_llm().bind_tools(TOOLS)
