import gradio as gr
import asyncio
import sys
import os
sys.path.append('..')

from src.core.config import Config
from src.core.system import AskFollowupsSystem

# Initialize system
config = Config(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    confidence_threshold=0.7
)
system = AskFollowupsSystem(config)

async def process_message(message, history):
    """Process user message and return response"""
    
    # Convert Gradio history format to our format
    conversation_history = []
    for human, assistant in history:
        if human:
            conversation_history.append({"role": "user", "content": human})
        if assistant:
            conversation_history.append({"role": "assistant", "content": assistant})
    
    # Process query
    result = await system.process_query(message, conversation_history)
    
    # Format response
    if result["action"] == "ask_clarification":
        response = "❓ **I need some clarification:**\n\n"
        for i, q in enumerate(result["clarification_questions"], 1):
            response += f"{i}. {q}\n"
        response += f"\n*Confidence: {result['confidence_scores'].get('overall_confidence', 0):.2f}*"
    else:
        response = result["answer"]
        response += f"\n\n*Confidence: {result['confidence_scores'].get('overall_confidence', 0):.2f}*"
    
    return response

def gradio_wrapper(message, history):
    """Synchronous wrapper for async function"""
    return asyncio.run(process_message(message, history))

# Create interface
demo = gr.ChatInterface(
    fn=gradio_wrapper,
    title="🤖 Ask Follow-ups System",
    description="Intelligent Q&A with automatic clarification detection",
    examples=[
        "How do I learn programming?",
        "What's the best web framework?", 
        "Explain machine learning concepts",
        "How do I build a REST API?",
        "What is Python?"
    ],
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )
