import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from agent import SYSTEM_PROMPT
from graph import create_graph

# ── Page Configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="Mini Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Premium Design Aesthetics ───────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .header-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a855f7 0%, #6366f1 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .header-sub {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    
    /* Custom card styles */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    
    .tool-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    
    .badge-web {
        background-color: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    
    .badge-calc {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    
    .badge-eval {
        background-color: rgba(168, 85, 247, 0.2);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.4);
    }
    
    /* Code block tweak */
    div.stCodeBlock {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Initialize State & Graph ────────────────────────────────────
if "graph_app" not in st.session_state:
    st.session_state["graph_app"] = create_graph()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 Research Agent Config")
    st.markdown("---")
    
    st.markdown("#### ⚡ Infrastructure")
    st.markdown("**LLM Provider:** Groq (`llama-3.1-8b-instant`)")
    st.markdown("**Orchestration:** LangGraph StateMachine")
    
    st.markdown("#### 🛠 Available Tools")
    st.markdown('<span class="tool-badge badge-web">🔎 search_web</span> Tavily API', unsafe_allow_html=True)
    st.markdown('<span class="tool-badge badge-calc">🧮 calculator</span> Python AST', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### 🚀 Quick Demo Prompts")
    
    sample_prompts = [
        "What is 17.5% of 85,000?",
        "Explain what a transformer is in simple terms.",
        "What is India's current population and what is 2% of it?",
        "Compare the latest NVIDIA and AMD AI GPUs.",
    ]
    
    selected_prompt = None
    for p in sample_prompts:
        if st.button(p, use_container_width=True):
            selected_prompt = p
            
    st.markdown("---")
    if st.button("🗑 Clear Chat History", use_container_width=True):
        st.session_state["chat_history"] = []
        st.rerun()

# ── Header ──────────────────────────────────────────────────────
st.markdown('<div class="header-title">🔬 Mini Research Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="header-sub">An autonomous multi-step research assistant powered by Groq & LangGraph.</div>',
    unsafe_allow_html=True,
)

# ── Render Chat History ─────────────────────────────────────────
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "steps" in msg and msg["steps"]:
            with st.expander("🔍 View Research Execution Steps", expanded=False):
                for step in msg["steps"]:
                    st.markdown(step)

# ── Chat Input & Processing ────────────────────────────────────
user_input = st.chat_input("Ask a research question or request a calculation...")

if selected_prompt:
    user_input = selected_prompt

if user_input:
    # Display user message
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Agent execution block
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.markdown("⏳ *Agent is reasoning and gathering research...*")
        
        execution_logs = []
        
        state = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_input),
            ],
            "research_count": 0,
            "research_decision": "",
        }

        try:
            # Execute LangGraph workflow
            final_state = st.session_state["graph_app"].invoke(state)
            
            # Parse intermediate execution steps for display
            messages = final_state.get("messages", [])
            
            for m in messages:
                if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                    for tc in m.tool_calls:
                        tool_name = tc["name"]
                        args = tc["args"]
                        if tool_name == "search_web":
                            execution_logs.append(f"🔎 **Agent requested Web Search:** `{args.get('query', '')}`")
                        elif tool_name == "calculator":
                            execution_logs.append(f"🧮 **Agent requested Calculator:** `{args.get('expression', '')}`")
                elif isinstance(m, ToolMessage):
                    execution_logs.append(f"✓ **Tool Result:** `{m.content[:150]}...`" if len(m.content) > 150 else f"✓ **Tool Result:** `{m.content}`")
                elif isinstance(m, HumanMessage) and "RESEARCH EVALUATION" in m.content:
                    execution_logs.append(f"🧠 **Evaluation Feedback:** {m.content}")

            # Extract final response content
            last_message = messages[-1]
            final_answer = last_message.content if isinstance(last_message, AIMessage) else "Unable to generate response."
            
            status_placeholder.empty()
            
            # Display final answer
            st.markdown(final_answer)
            
            # Display execution steps in expander
            if execution_logs:
                with st.expander("🔍 View Research Execution Steps", expanded=True):
                    for log in execution_logs:
                        st.markdown(log)

            # Store in session state
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": final_answer,
                "steps": execution_logs,
            })

        except Exception as e:
            status_placeholder.empty()
            st.error(f"Error during agent execution: {e}")
