import streamlit as st
import asyncio
import sys
import os
sys.path.append('..')

from src.core.config import Config
from src.core.system import AskFollowupsSystem

st.set_page_config(
    page_title="Ask Follow-ups Demo",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Ask Follow-ups System Demo")
st.write("Intelligent Q&A system with automatic clarification")

# Initialize system (cache it)
@st.cache_resource
def get_system():
    config = Config(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        confidence_threshold=0.7
    )
    return AskFollowupsSystem(config)

# Load system
try:
    system = get_system()
    st.success("✅ System initialized successfully")
except Exception as e:
    st.error(f"❌ Failed to initialize system: {e}")
    st.stop()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    # Process query
    with st.chat_message("assistant"):
        with st.spinner("🔄 Processing your question..."):
            result = asyncio.run(system.process_query(
                prompt, 
                st.session_state.messages[:-1]
            ))
        
        # Show confidence metrics in sidebar
        with st.sidebar:
            st.subheader("📊 Confidence Metrics")
            for metric, score in result["confidence_scores"].items():
                st.metric(
                    metric.replace("_", " ").title(), 
                    f"{score:.3f}",
                    delta=None
                )
        
        if result["action"] == "ask_clarification":
            st.write("❓ **I need some clarification:**")
            for i, question in enumerate(result["clarification_questions"], 1):
                st.write(f"{i}. {question}")
        else:
            st.write(result["answer"])
    
    # Add assistant message to history
    if result["action"] == "ask_clarification":
        assistant_msg = f"Clarifying questions: {'; '.join(result['clarification_questions'])}"
    else:
        assistant_msg = result["answer"]
    
    st.session_state.messages.append({"role": "assistant", "content": assistant_msg})
