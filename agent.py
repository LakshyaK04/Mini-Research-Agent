from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tools import search_web, calculator

load_dotenv()

TOOLS = [search_web, calculator]

SYSTEM_PROMPT = (
    "You are a helpful research assistant with access to two tools:\n"
    "1. search_web — Search the internet for facts, statistics, and current info.\n"
    "2. calculator — Perform math calculations.\n\n"
    "Guidelines:\n"
    "- Use search_web for facts and calculator for math.\n"
    "- For questions requiring search + math, search FIRST, then calculate.\n"
    "- When you have enough info, answer directly without calling tools."
)


def get_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0)


def get_llm_with_tools():
    return get_llm().bind_tools(TOOLS)
