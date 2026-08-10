"""
main.py — Terminal interface for the Mini Research Agent

Provides an interactive loop where the user can ask questions.
Each question creates a fresh research state.
"""

import sys
from langchain_core.messages import HumanMessage, SystemMessage

from agent import SYSTEM_PROMPT
from graph import create_graph

# Ensure UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def print_banner():
    """Print the startup banner."""
    print()
    print("=" * 44)
    print("        🔬  MINI RESEARCH AGENT  🔬")
    print("=" * 44)
    print()
    print("  Ask me anything. I can search the web")
    print("  and perform calculations to help you.")
    print()
    print("  Type 'exit' or 'quit' to leave.")
    print("=" * 44)


def run_query(app, question: str):
    """Run a single research query through the graph."""

    print(f"\n{'─' * 40}")
    print(f"\n💬 You: {question}")
    print(f"\n{'─' * 40}")

    # Build fresh state for each question
    initial_state = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question),
        ],
        "research_count": 0,
        "research_decision": "",
    }

    try:
        # invoke() runs the full graph and returns the final state
        final_state = app.invoke(
            initial_state,
            # Recursion limit prevents runaway loops in the graph itself
            config={"recursion_limit": 25},
        )
    except Exception as e:
        print(f"\n⚠  Error: {e}")
        print("   The agent encountered a problem. Please try again.")

    print(f"\n{'=' * 44}\n")


def main():
    """Entry point — interactive terminal loop."""
    print_banner()

    # Compile the graph once, reuse for every question
    app = create_graph()

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("\nGoodbye! 👋")
            break

        run_query(app, question)


if __name__ == "__main__":
    main()
