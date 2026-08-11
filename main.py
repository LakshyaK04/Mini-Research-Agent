from langchain_core.messages import HumanMessage, SystemMessage

from agent import SYSTEM_PROMPT
from graph import create_graph


def main():
    app = create_graph()

    print("=== Mini Research Agent ===")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        state = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=question),
            ],
            "research_count": 0,
            "research_decision": "",
        }

        try:
            app.invoke(state)
        except Exception as e:
            print(f"Error: {e}")

        print()


if __name__ == "__main__":
    main()
