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
from langchain_groq import ChatGroq

from tools import search_web, calculator

# Ensure UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Load environment variables (.env)
load_dotenv()

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


# ── LLM Factory functions ─────────────────────

def get_llm():
    """Return the free Groq LLM instance."""
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0)


def get_llm_with_tools():
    """Return the Groq LLM instance with tools bound."""
    return get_llm().bind_tools(TOOLS)
