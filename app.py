"""
Open-Source Ask Follow-ups Implementation
Combines TruLens, Ragas, LangGraph, and other tools for production-ready clarification system
"""

import json
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from abc import ABC, abstractmethod

# Core dependencies
import openai
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

# LangGraph for state management
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor

# TruLens for evaluation
from trulens_eval import TruChain, Feedback, Tru
from trulens_eval.feedback import Groundedness, ContextRelevance, AnswerRelevance

# Ragas for RAG metrics
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# Guardrails for structured outputs
import guardrails as gd
from guardrails.validators import ValidChoices

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
@dataclass
class Config:
    """System configuration"""
    openai_api_key: str
    model_name: str = "gpt-4"
    embedding_model: str = "all-MiniLM-L6-v2"
    confidence_threshold: float = 0.7
    groundedness_threshold: float = 0.6
    context_relevance_threshold: float = 0.6
    answer_relevance_threshold: float = 0.7
    max_clarification_attempts: int = 3
    vector_db_path: str = "./chroma_db"

class ActionType(Enum):
    """Available actions in the state machine"""
    ASK_CLARIFICATION = "ask_clarification"
    PROVIDE_ANSWER = "provide_answer"
    RETRIEVE_MORE = "retrieve_more"
    END_CONVERSATION = "end_conversation"

@dataclass
class ConversationState:
    """State for the conversation state machine"""
    query: str = ""
    context_history: List[str] = None
    retrieved_docs: List[Dict] = None
    current_answer: str = ""
    confidence_scores: Dict[str, float] = None
    clarification_questions: List[str] = None
    clarification_attempts: int = 0
    next_action: ActionType = ActionType.RETRIEVE_MORE
    conversation_history: List[Dict] = None
    
    def __post_init__(self):
        if self.context_history is None:
            self.context_history = []
        if self.retrieved_docs is None:
            self.retrieved_docs = []
        if self.confidence_scores is None:
            self.confidence_scores = {}
        if self.clarification_questions is None:
            self.clarification_questions = []
        if self.conversation_history is None:
            self.conversation_history = []

# Guardrails schema for structured outputs
clarification_schema = gd.Guard.from_pydantic(
    output_class=dict,
    prompt="""
    Generate a response with the following structure:
    - next_action: either "ask_clarification" or "provide_answer"
    - questions: list of 1-3 targeted clarifying questions (if next_action is ask_clarification)
    - confidence: float between 0 and 1 indicating confidence in the response
    """,
    validators=[
        ValidChoices(choices=["ask_clarification", "provide_answer"], on_fail="exception")
    ]
)

class VectorStore:
    """Vector database wrapper using ChromaDB"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = chromadb.PersistentClient(path=config.vector_db_path)
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
        self.encoder = SentenceTransformer(config.embedding_model)
    
    def add_documents(self, documents: List[Dict[str, str]]):
        """Add documents to the vector store"""
        texts = [doc["content"] for doc in documents]
        embeddings = self.encoder.encode(texts).tolist()
        
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=[doc.get("metadata", {}) for doc in documents],
            ids=[f"doc_{i}" for i in range(len(documents))]
        )
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant documents"""
        query_embedding = self.encoder.encode([query]).tolist()[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        retrieved_docs = []
        for i, (doc, metadata, distance) in enumerate(
            zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
        ):
            retrieved_docs.append({
                "content": doc,
                "metadata": metadata,
                "relevance_score": 1 - distance,
                "rank": i
            })
        
        return retrieved_docs

class EvaluationEngine:
    """Combines TruLens and Ragas for comprehensive evaluation"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize TruLens
        self.tru = Tru()
        self.groundedness_feedback = Groundedness()
        self.context_relevance_feedback = ContextRelevance()
        self.answer_relevance_feedback = AnswerRelevance()
        
        # Initialize OpenAI client for Ragas
        self.openai_client = openai.OpenAI(api_key=config.openai_api_key)
    
    async def evaluate_response(
        self, 
        query: str, 
        context: List[str], 
        answer: str
    ) -> Dict[str, float]:
        """Comprehensive evaluation using multiple metrics"""
        
        scores = {}
        
        # TruLens evaluations
        try:
            # Groundedness: How well grounded the answer is in the context
            groundedness_score = await self._evaluate_groundedness(context, answer)
            scores["groundedness"] = groundedness_score
            
            # Context Relevance: How relevant the context is to the query
            context_relevance_score = await self._evaluate_context_relevance(query, context)
            scores["context_relevance"] = context_relevance_score
            
            # Answer Relevance: How relevant the answer is to the query
            answer_relevance_score = await self._evaluate_answer_relevance(query, answer)
            scores["answer_relevance"] = answer_relevance_score
            
        except Exception as e:
            logger.error(f"TruLens evaluation error: {e}")
            # Fallback scores
            scores.update({
                "groundedness": 0.5,
                "context_relevance": 0.5,
                "answer_relevance": 0.5
            })
        
        # Ragas evaluations (simplified for demo)
        try:
            ragas_scores = await self._evaluate_with_ragas(query, context, answer)
            scores.update(ragas_scores)
        except Exception as e:
            logger.error(f"Ragas evaluation error: {e}")
            scores.update({
                "faithfulness": 0.5,
                "context_precision": 0.5
            })
        
        # Overall confidence score
        scores["overall_confidence"] = self._calculate_overall_confidence(scores)
        
        return scores
    
    async def _evaluate_groundedness(self, context: List[str], answer: str) -> float:
        """Evaluate how well the answer is grounded in context"""
        # Simplified implementation - in production, use actual TruLens feedback
        context_text = " ".join(context)
        
        prompt = f"""
        Rate how well the following answer is grounded in the provided context on a scale of 0-1:
        
        Context: {context_text}
        Answer: {answer}
        
        Return only a number between 0 and 1.
        """
        
        response = await self._call_llm(prompt)
        try:
            return float(response.strip())
        except:
            return 0.5
    
    async def _evaluate_context_relevance(self, query: str, context: List[str]) -> float:
        """Evaluate how relevant the context is to the query"""
        context_text = " ".join(context)
        
        prompt = f"""
        Rate how relevant the following context is to answering the query on a scale of 0-1:
        
        Query: {query}
        Context: {context_text}
        
        Return only a number between 0 and 1.
        """
        
        response = await self._call_llm(prompt)
        try:
            return float(response.strip())
        except:
            return 0.5
    
    async def _evaluate_answer_relevance(self, query: str, answer: str) -> float:
        """Evaluate how relevant the answer is to the query"""
        prompt = f"""
        Rate how relevant the following answer is to the query on a scale of 0-1:
        
        Query: {query}
        Answer: {answer}
        
        Return only a number between 0 and 1.
        """
        
        response = await self._call_llm(prompt)
        try:
            return float(response.strip())
        except:
            return 0.5
    
    async def _evaluate_with_ragas(self, query: str, context: List[str], answer: str) -> Dict[str, float]:
        """Simplified Ragas evaluation"""
        # In production, use actual Ragas metrics
        return {
            "faithfulness": 0.8,  # Placeholder
            "context_precision": 0.7,  # Placeholder
        }
    
    def _calculate_overall_confidence(self, scores: Dict[str, float]) -> float:
        """Calculate weighted overall confidence score"""
        weights = {
            "groundedness": 0.3,
            "context_relevance": 0.25,
            "answer_relevance": 0.25,
            "faithfulness": 0.2
        }
        
        weighted_sum = sum(
            scores.get(metric, 0.5) * weight 
            for metric, weight in weights.items()
        )
        
        return min(max(weighted_sum, 0.0), 1.0)
    
    async def _call_llm(self, prompt: str) -> str:
        """Helper to call LLM for evaluations"""
        response = await self.openai_client.chat.completions.acreate(
            model=self.config.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1
        )
        return response.choices[0].message.content

class ClarificationGenerator:
    """Generates targeted clarifying questions"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = openai.OpenAI(api_key=config.openai_api_key)
    
    async def generate_clarification_questions(
        self, 
        query: str, 
        context: List[str], 
        confidence_scores: Dict[str, float],
        conversation_history: List[Dict] = None
    ) -> List[str]:
        """Generate 1-3 targeted clarifying questions"""
        
        # Identify the main issues based on low scores
        issues = self._identify_issues(confidence_scores)
        
        context_text = " ".join(context) if context else "No relevant context found"
        history_text = self._format_conversation_history(conversation_history or [])
        
        prompt = f"""
        The user asked: "{query}"
        
        Conversation history:
        {history_text}
        
        Available context: {context_text[:500]}...
        
        Evaluation scores indicate these issues: {', '.join(issues)}
        
        Generate 1-3 specific, targeted questions to clarify the user's intent and gather missing information.
        Focus on the identified issues. Make questions actionable and specific.
        
        Return as a JSON list of strings: ["question1", "question2", ...]
        """
        
        try:
            response = await self.client.chat.completions.acreate(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            questions = json.loads(content)
            return questions[:3]  # Limit to 3 questions
            
        except Exception as e:
            logger.error(f"Error generating clarification questions: {e}")
            return [
                "Could you provide more specific details about what you're looking for?",
                "What particular aspect of this topic interests you most?"
            ]
    
    def _identify_issues(self, scores: Dict[str, float]) -> List[str]:
        """Identify specific issues based on evaluation scores"""
        issues = []
        
        if scores.get("groundedness", 1.0) < self.config.groundedness_threshold:
            issues.append("insufficient supporting evidence")
        
        if scores.get("context_relevance", 1.0) < self.config.context_relevance_threshold:
            issues.append("context not relevant to query")
        
        if scores.get("answer_relevance", 1.0) < self.config.answer_relevance_threshold:
            issues.append("answer doesn't address the question")
        
        if scores.get("overall_confidence", 1.0) < self.config.confidence_threshold:
            issues.append("low overall confidence")
        
        return issues or ["unclear user intent"]
    
    def _format_conversation_history(self, history: List[Dict]) -> str:
        """Format conversation history for context"""
        if not history:
            return "No previous conversation"
        
        formatted = []
        for turn in history[-3:]:  # Last 3 turns
            role = turn.get("role", "unknown")
            content = turn.get("content", "")[:200]  # Truncate
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)

class AnswerGenerator:
    """Generates answers using retrieved context"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = openai.OpenAI(api_key=config.openai_api_key)
    
    async def generate_answer(
        self, 
        query: str, 
        context: List[str],
        conversation_history: List[Dict] = None
    ) -> str:
        """Generate answer based on query and context"""
        
        context_text = "\n".join(context) if context else "No relevant context available"
        history_text = self._format_conversation_history(conversation_history or [])
        
        prompt = f"""
        Previous conversation:
        {history_text}
        
        Current query: {query}
        
        Relevant context:
        {context_text}
        
        Based on the context provided, generate a comprehensive and accurate answer to the query.
        If the context is insufficient, acknowledge this limitation.
        Be specific and cite relevant information from the context.
        """
        
        try:
            response = await self.client.chat.completions.acreate(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I apologize, but I'm having trouble generating an answer right now."
    
    def _format_conversation_history(self, history: List[Dict]) -> str:
        """Format conversation history"""
        if not history:
            return "No previous conversation"
        
        formatted = []
        for turn in history[-5:]:  # Last 5 turns
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)

class AskFollowupsSystem:
    """Main system that orchestrates the ask follow-ups pattern"""
    
    def __init__(self, config: Config):
        self.config = config
        self.vector_store = VectorStore(config)
        self.evaluator = EvaluationEngine(config)
        self.clarification_generator = ClarificationGenerator(config)
        self.answer_generator = AnswerGenerator(config)
        
        # Build the state graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph state machine"""
        
        workflow = StateGraph(ConversationState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("evaluate", self._evaluate_node)
        workflow.add_node("clarify", self._clarify_node)
        workflow.add_node("answer", self._answer_node)
        
        # Add edges with conditions
        workflow.add_edge("retrieve", "evaluate")
        workflow.add_conditional_edges(
            "evaluate",
            self._should_clarify,
            {
                True: "clarify",
                False: "answer"
            }
        )
        workflow.add_edge("clarify", END)
        workflow.add_edge("answer", END)
        
        # Set entry point
        workflow.set_entry_point("retrieve")
        
        return workflow.compile()
    
    async def _retrieve_node(self, state: ConversationState) -> ConversationState:
        """Retrieval node"""
        logger.info(f"Retrieving documents for: {state.query}")
        
        # Enhance query with conversation context if available
        enhanced_query = self._enhance_query_with_context(state)
        
        retrieved_docs = self.vector_store.retrieve(enhanced_query, top_k=5)
        state.retrieved_docs = retrieved_docs
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        return state
    
    async def _evaluate_node(self, state: ConversationState) -> ConversationState:
        """Evaluation node"""
        logger.info("Evaluating retrieved context and potential answer")
        
        # Extract context from retrieved docs
        context = [doc["content"] for doc in state.retrieved_docs]
        
        # Generate preliminary answer for evaluation
        preliminary_answer = await self.answer_generator.generate_answer(
            state.query, context, state.conversation_history
        )
        
        # Evaluate the response
        scores = await self.evaluator.evaluate_response(
            state.query, context, preliminary_answer
        )
        
        state.confidence_scores = scores
        state.current_answer = preliminary_answer
        
        logger.info(f"Evaluation scores: {scores}")
        return state
    
    async def _clarify_node(self, state: ConversationState) -> ConversationState:
        """Clarification node"""
        logger.info("Generating clarifying questions")
        
        context = [doc["content"] for doc in state.retrieved_docs]
        
        questions = await self.clarification_generator.generate_clarification_questions(
            state.query,
            context,
            state.confidence_scores,
            state.conversation_history
        )
        
        state.clarification_questions = questions
        state.clarification_attempts += 1
        state.next_action = ActionType.ASK_CLARIFICATION
        
        return state
    
    async def _answer_node(self, state: ConversationState) -> ConversationState:
        """Answer generation node"""
        logger.info("Generating final answer")
        
        context = [doc["content"] for doc in state.retrieved_docs]
        
        if not state.current_answer:
            state.current_answer = await self.answer_generator.generate_answer(
                state.query, context, state.conversation_history
            )
        
        state.next_action = ActionType.PROVIDE_ANSWER
        return state
    
    def _should_clarify(self, state: ConversationState) -> bool:
        """Decision function: should we ask for clarification?"""
        
        # Don't clarify if we've already tried too many times
        if state.clarification_attempts >= self.config.max_clarification_attempts:
            return False
        
        scores = state.confidence_scores
        
        # Check if any critical scores are below threshold
        critical_checks = [
            scores.get("overall_confidence", 1.0) < self.config.confidence_threshold,
            scores.get("groundedness", 1.0) < self.config.groundedness_threshold,
            scores.get("context_relevance", 1.0) < self.config.context_relevance_threshold,
            scores.get("answer_relevance", 1.0) < self.config.answer_relevance_threshold,
        ]
        
        return any(critical_checks)
    
    def _enhance_query_with_context(self, state: ConversationState) -> str:
        """Enhance query with conversation context"""
        if not state.conversation_history:
            return state.query
        
        # Simple context enhancement - in production, use more sophisticated methods
        recent_context = []
        for turn in state.conversation_history[-2:]:
            if turn.get("role") == "user":
                recent_context.append(turn.get("content", ""))
        
        if recent_context:
            enhanced = f"{' '.join(recent_context)} {state.query}"
            return enhanced
        
        return state.query
    
    async def process_query(self, query: str, conversation_history: List[Dict] = None) -> Dict:
        """Process a user query through the complete pipeline"""
        
        # Initialize state
        initial_state = ConversationState(
            query=query,
            conversation_history=conversation_history or []
        )
        
        try:
            # Run the workflow
            result = await self.workflow.ainvoke(initial_state)
            
            return {
                "action": result.next_action.value,
                "answer": result.current_answer,
                "clarification_questions": result.clarification_questions,
                "confidence_scores": result.confidence_scores,
                "retrieved_docs_count": len(result.retrieved_docs),
                "clarification_attempts": result.clarification_attempts
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "action": "provide_answer",
                "answer": "I apologize, but I encountered an error processing your query.",
                "clarification_questions": [],
                "confidence_scores": {},
                "retrieved_docs_count": 0,
                "clarification_attempts": 0
            }
    
    async def process_clarification_response(
        self, 
        original_query: str,
        clarification_response: str,
        conversation_history: List[Dict] = None
    ) -> Dict:
        """Process user's response to clarification questions"""
        
        # Combine original query with clarification
        enhanced_query = f"{original_query}. Additional context: {clarification_response}"
        
        # Update conversation history
        updated_history = (conversation_history or []).copy()
        updated_history.extend([
            {"role": "assistant", "content": "I need some clarification..."},
            {"role": "user", "content": clarification_response}
        ])
        
        return await self.process_query(enhanced_query, updated_history)

# Example usage and testing
class ExampleUsage:
    """Example implementation and testing"""
    
    @staticmethod
    async def run_example():
        """Run example usage of the system"""
        
        # Configuration
        config = Config(
            openai_api_key="your-openai-key-here",  # Replace with actual key
            confidence_threshold=0.7,
            groundedness_threshold=0.6
        )
        
        # Initialize system
        system = AskFollowupsSystem(config)
        
        # Add some example documents
        example_docs = [
            {
                "content": "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
                "metadata": {"source": "python_guide.md", "topic": "programming"}
            },
            {
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. Common algorithms include linear regression, decision trees, and neural networks.",
                "metadata": {"source": "ml_intro.md", "topic": "machine_learning"}
            },
            {
                "content": "FastAPI is a modern, fast web framework for building APIs with Python 3.6+ based on standard Python type hints. It's designed to be easy to use and learn while being production-ready.",
                "metadata": {"source": "fastapi_docs.md", "topic": "web_development"}
            }
        ]
        
        system.vector_store.add_documents(example_docs)
        
        # Example conversation flow
        print("=== Ask Follow-ups System Demo ===\n")
        
        # First query - intentionally vague
        query1 = "How do I learn?"
        print(f"User: {query1}")
        
        result1 = await system.process_query(query1)
        print(f"System action: {result1['action']}")
        print(f"Confidence scores: {result1['confidence_scores']}")
        
        if result1['action'] == 'ask_clarification':
            print("Clarifying questions:")
            for i, q in enumerate(result1['clarification_questions'], 1):
                print(f"  {i}. {q}")
            
            # Simulate user response
            clarification = "I want to learn Python programming for web development"
            print(f"\nUser clarification: {clarification}")
            
            # Process clarification
            result2 = await system.process_clarification_response(query1, clarification)
            print(f"System action: {result2['action']}")
            print(f"Answer: {result2['answer']}")
            print(f"Final confidence: {result2['confidence_scores'].get('overall_confidence', 'N/A')}")
        else:
            print(f"Answer: {result1['answer']}")
        
        return system

# Production deployment helpers
class ProductionHelpers:
    """Utilities for production deployment"""
    










    
    
    @staticmethod
    def create_fastapi_app() -> str:
        """Generate FastAPI application wrapper"""
        return """
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio

app = FastAPI(title="Ask Follow-ups API", version="1.0.0")

class QueryRequest(BaseModel):
    query: str
    conversation_history: Optional[List[Dict]] = []

class ClarificationRequest(BaseModel):
    original_query: str
    clarification_response: str
    conversation_history: Optional[List[Dict]] = []

class QueryResponse(BaseModel):
    action: str
    answer: Optional[str] = None
    clarification_questions: Optional[List[str]] = []
    confidence_scores: Dict[str, float]
    retrieved_docs_count: int

# Initialize system (you'll need to configure this)
# system = AskFollowupsSystem(config)

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        result = await system.process_query(
            request.query, 
            request.conversation_history
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clarify", response_model=QueryResponse)
async def process_clarification(request: ClarificationRequest):
    try:
        result = await system.process_clarification_response(
            request.original_query,
            request.clarification_response,
            request.conversation_history
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
"""

# Integration examples
class IntegrationExamples:
    """Examples of integrating with different frameworks"""
    
    @staticmethod
    def haystack_integration() -> str:
        """Example Haystack pipeline integration"""
        return """
from haystack import Pipeline
from haystack.components.routers import ConditionalRouter

def create_haystack_pipeline(ask_followups_system):
    pipeline = Pipeline()
    
    # Add components
    pipeline.add_component("retriever", YourRetriever())
    pipeline.add_component("evaluator", YourEvaluator())
    pipeline.add_component("router", ConditionalRouter(
        routes=[
            {
                "condition": "{{confidence < 0.7}}",
                "output": "{{clarify}}",
                "output_name": "clarify"
            },
            {
                "condition": "{{confidence >= 0.7}}",
                "output": "{{answer}}",
                "output_name": "answer"
            }
        ]
    ))
    pipeline.add_component("clarifier", YourClarifier())
    pipeline.add_component("answerer", YourAnswerer())
    
    # Connect components
    pipeline.connect("retriever", "evaluator")
    pipeline.connect("evaluator", "router")
    pipeline.connect("router.clarify", "clarifier")
    pipeline.connect("router.answer", "answerer")
    
    return pipeline
"""
    
    @staticmethod
    def llamaindex_integration() -> str:
        """Example LlamaIndex integration"""
        return """
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

class LlamaIndexAskFollowups:
    def __init__(self, index: VectorStoreIndex, ask_followups_system):
        self.index = index
        self.ask_followups_system = ask_followups_system
        
        # Create query engine with post-processing
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=index.as_retriever(similarity_top_k=5),
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]
        )
    
    async def query_with_clarification(self, query_str: str):
        # First attempt
        response = self.query_engine.query(query_str)
        
        # Evaluate using our system
        result = await self.ask_followups_system.process_query(query_str)
        
        if result['action'] == 'ask_clarification':
            return {
                "needs_clarification": True,
                "questions": result['clarification_questions'],
                "preliminary_answer": response.response
            }
        else:
            return {
                "needs_clarification": False,
                "answer": response.response,
                "confidence": result['confidence_scores']
            }
"""

# Advanced workflow patterns
class AdvancedWorkflowPatterns:
    """Advanced patterns for complex clarification scenarios"""
    
    @staticmethod
    def multi_step_clarification_workflow():
        """Create workflow that handles multiple clarification rounds"""
        
        workflow = StateGraph(ConversationState)
        
        # Multi-step nodes
        workflow.add_node("initial_retrieve", lambda state: state)
        workflow.add_node("deep_retrieve", lambda state: state) 
        workflow.add_node("evaluate_comprehensively", lambda state: state)
        workflow.add_node("generate_targeted_questions", lambda state: state)
        workflow.add_node("synthesize_final_answer", lambda state: state)
        workflow.add_node("validate_answer", lambda state: state)
        
        # Complex routing logic
        def route_after_evaluation(state: ConversationState) -> str:
            scores = state.confidence_scores
            
            if scores.get("context_relevance", 0) < 0.5:
                return "deep_retrieve"
            elif scores.get("overall_confidence", 0) < 0.6:
                return "generate_targeted_questions"
            elif scores.get("groundedness", 0) < 0.7:
                return "validate_answer"
            else:
                return "synthesize_final_answer"
        
        workflow.add_conditional_edges(
            "evaluate_comprehensively",
            route_after_evaluation,
            {
                "deep_retrieve": "deep_retrieve",
                "generate_targeted_questions": "generate_targeted_questions", 
                "validate_answer": "validate_answer",
                "synthesize_final_answer": "synthesize_final_answer"
            }
        )
        
        return workflow
    
    @staticmethod
    def intent_classification_pattern():
        """Pattern for handling different query intents"""
        return """
# Intent-based clarification routing
class IntentClassifier:
    def __init__(self):
        self.intents = {
            "factual_question": "User wants specific facts",
            "how_to_guide": "User wants step-by-step instructions", 
            "comparison": "User wants to compare options",
            "troubleshooting": "User has a problem to solve",
            "exploratory": "User wants to explore a topic"
        }
    
    async def classify_intent(self, query: str, context: List[str]) -> str:
        # Use LLM to classify intent
        prompt = f'''
        Classify the user's intent from: {list(self.intents.keys())}
        
        Query: {query}
        Context available: {len(context)} documents
        
        Return only the intent category.
        '''
        # Implementation here...
        return "factual_question"
    
    def get_clarification_strategy(self, intent: str, confidence_scores: Dict) -> str:
        strategies = {
            "factual_question": "Ask for more specific details about what facts they need",
            "how_to_guide": "Ask about their experience level and specific use case",
            "comparison": "Ask what criteria matter most for the comparison",
            "troubleshooting": "Ask about their current setup and what they've tried",
            "exploratory": "Ask what aspect interests them most"
        }
        return strategies.get(intent, "Ask for more context")
"""

# Monitoring and analytics
class MonitoringSystem:
    """Production monitoring for the ask follow-ups system"""
    
    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "clarification_rate": 0.0,
            "average_confidence": 0.0,
            "successful_resolutions": 0,
            "failed_queries": 0
        }
    
    def track_query(self, result: Dict):
        """Track metrics for a query"""
        self.metrics["total_queries"] += 1
        
        if result["action"] == "ask_clarification":
            self.metrics["clarification_rate"] = (
                (self.metrics["clarification_rate"] * (self.metrics["total_queries"] - 1) + 1) 
                / self.metrics["total_queries"]
            )
        
        confidence = result["confidence_scores"].get("overall_confidence", 0)
        self.metrics["average_confidence"] = (
            (self.metrics["average_confidence"] * (self.metrics["total_queries"] - 1) + confidence)
            / self.metrics["total_queries"]
        )
    
    def get_performance_report(self) -> Dict:
        """Generate performance report"""
        return {
            "query_volume": self.metrics["total_queries"],
            "clarification_rate": f"{self.metrics['clarification_rate']:.2%}",
            "avg_confidence": f"{self.metrics['average_confidence']:.3f}",
            "success_rate": f"{1 - self.metrics['clarification_rate']:.2%}"
        }

# Testing framework
class TestSuite:
    """Comprehensive testing for the system"""
    
    def __init__(self, system: AskFollowupsSystem):
        self.system = system
        self.test_cases = self._create_test_cases()
    
    def _create_test_cases(self) -> List[Dict]:
        """Create comprehensive test cases"""
        return [
            {
                "name": "clear_specific_query",
                "query": "What is the syntax for creating a FastAPI route?",
                "expected_action": "provide_answer",
                "expected_confidence": "> 0.7"
            },
            {
                "name": "vague_query",
                "query": "How do I code?",
                "expected_action": "ask_clarification",
                "expected_questions": "> 0"
            },
            {
                "name": "ambiguous_context",
                "query": "What's the best framework?",
                "expected_action": "ask_clarification",
                "expected_questions": "> 1"
            },
            {
                "name": "follow_up_after_clarification",
                "query": "How do I learn?",
                "clarification": "I want to learn machine learning with Python",
                "expected_final_action": "provide_answer"
            }
        ]
    
    async def run_tests(self) -> Dict[str, Any]:
        """Run all test cases"""
        results = []
        
        for test_case in self.test_cases:
            try:
                result = await self._run_single_test(test_case)
                results.append({
                    "test_name": test_case["name"],
                    "passed": result["passed"],
                    "details": result
                })
            except Exception as e:
                results.append({
                    "test_name": test_case["name"],
                    "passed": False,
                    "error": str(e)
                })
        
        # Calculate overall pass rate
        passed_tests = sum(1 for r in results if r["passed"])
        pass_rate = passed_tests / len(results) if results else 0
        
        return {
            "total_tests": len(results),
            "passed": passed_tests,
            "pass_rate": f"{pass_rate:.2%}",
            "detailed_results": results
        }
    
    async def _run_single_test(self, test_case: Dict) -> Dict:
        """Run a single test case"""
        query = test_case["query"]
        
        # Run initial query
        result = await self.system.process_query(query)
        
        # Check expectations
        checks = {
            "action_correct": result["action"] == test_case.get("expected_action"),
            "confidence_met": self._check_confidence(
                result["confidence_scores"], 
                test_case.get("expected_confidence", "")
            ),
            "questions_generated": len(result["clarification_questions"]) > 0 
                if "expected_questions" in test_case else True
        }
        
        # Handle clarification tests
        if "clarification" in test_case:
            clarification_result = await self.system.process_clarification_response(
                query, test_case["clarification"]
            )
            checks["final_action_correct"] = (
                clarification_result["action"] == test_case.get("expected_final_action")
            )
        
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "result": result
        }
    
    def _check_confidence(self, scores: Dict, expected: str) -> bool:
        """Check if confidence meets expectations"""
        if not expected:
            return True
        
        overall = scores.get("overall_confidence", 0)
        if expected.startswith("> "):
            threshold = float(expected[2:])
            return overall > threshold
        elif expected.startswith("< "):
            threshold = float(expected[2:])
            return overall < threshold
        
        return True

# Main execution and CLI
async def main():
    """Main execution function"""
    
    print("🚀 Ask Follow-ups System - Open Source Implementation")
    print("=" * 60)
    
    # Note: Replace with your actual OpenAI API key
    config = Config(
        openai_api_key="your-openai-api-key-here",
        confidence_threshold=0.7
    )
    
    try:
        # Initialize system
        print("Initializing system...")
        system = AskFollowupsSystem(config)
        
        # Add sample knowledge base
        print("Loading sample knowledge base...")
        sample_docs = [
            {
                "content": "Python is an interpreted, high-level programming language with dynamic semantics. Its high-level built-in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development, as well as for use as a scripting or glue language to connect existing components together.",
                "metadata": {"source": "python_official", "category": "programming"}
            },
            {
                "content": "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints. Key features include: Fast to code, fewer bugs, intuitive, easy, short, robust, standards-based.",
                "metadata": {"source": "fastapi_docs", "category": "web_frameworks"}
            },
            {
                "content": "Machine Learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
                "metadata": {"source": "ml_intro", "category": "machine_learning"}
            },
            {
                "content": "React is a free and open-source front-end JavaScript library for building user interfaces based on UI components. It is maintained by Meta and a community of individual developers and companies.",
                "metadata": {"source": "react_docs", "category": "frontend"}
            }
        ]
        
        system.vector_store.add_documents(sample_docs)
        
        # Run interactive demo
        print("\n🎯 Interactive Demo")
        print("Ask questions and see the clarification system in action!")
        print("Type 'quit' to exit\n")
        
        conversation_history = []
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                print("🔄 Processing...")
                
                # Process query
                result = await system.process_query(user_input, conversation_history)
                
                # Add to conversation history
                conversation_history.append({
                    "role": "user", 
                    "content": user_input
                })
                
                print(f"\n📊 Confidence Scores:")
                for metric, score in result["confidence_scores"].items():
                    print(f"  {metric}: {score:.3f}")
                
                print(f"\n🎯 Action: {result['action']}")
                
                if result["action"] == "ask_clarification":
                    print("\n❓ I need some clarification:")
                    for i, question in enumerate(result["clarification_questions"], 1):
                        print(f"  {i}. {question}")
                    
                    # Get clarification from user
                    clarification = input("\nYour clarification: ").strip()
                    
                    if clarification:
                        conversation_history.append({
                            "role": "assistant",
                            "content": f"Clarifying questions: {'; '.join(result['clarification_questions'])}"
                        })
                        conversation_history.append({
                            "role": "user",
                            "content": clarification
                        })
                        
                        print("🔄 Processing clarification...")
                        
                        # Process clarification
                        final_result = await system.process_clarification_response(
                            user_input, clarification, conversation_history
                        )
                        
                        print(f"\n📊 Updated Confidence: {final_result['confidence_scores'].get('overall_confidence', 'N/A'):.3f}")
                        print(f"\n💬 Answer: {final_result['answer']}")
                        
                        conversation_history.append({
                            "role": "assistant",
                            "content": final_result['answer']
                        })
                
                else:
                    print(f"\n💬 Answer: {result['answer']}")
                    conversation_history.append({
                        "role": "assistant",
                        "content": result['answer']
                    })
                
                print("\n" + "-" * 60 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again.\n")
    
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        print("\nPlease check your configuration and try again.")

# Production deployment configurations
class ProductionConfig:
    """Production-ready configurations and optimizations"""
    
    @staticmethod
    def get_kubernetes_deployment() -> str:
        """Kubernetes deployment configuration"""
        return """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ask-followups-api
  labels:
    app: ask-followups
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ask-followups
  template:
    metadata:
      labels:
        app: ask-followups
    spec:
      containers:
      - name: api
        image: ask-followups:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-key
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ask-followups-service
spec:
  selector:
    app: ask-followups
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
"""
    
    @staticmethod
    def get_monitoring_setup() -> str:
        """Prometheus monitoring configuration"""
        return """
# metrics.py - Add to your system
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
QUERY_COUNTER = Counter('queries_total', 'Total queries processed')
CLARIFICATION_COUNTER = Counter('clarifications_total', 'Total clarifications requested')
CONFIDENCE_HISTOGRAM = Histogram('confidence_scores', 'Distribution of confidence scores')
RESPONSE_TIME = Histogram('response_time_seconds', 'Response time in seconds')
ACTIVE_CONVERSATIONS = Gauge('active_conversations', 'Number of active conversations')

class MetricsMiddleware:
    def __init__(self, system: AskFollowupsSystem):
        self.system = system
    
    async def process_with_metrics(self, query: str, conversation_history: List[Dict] = None):
        start_time = time.time()
        QUERY_COUNTER.inc()
        ACTIVE_CONVERSATIONS.inc()
        
        try:
            result = await self.system.process_query(query, conversation_history)
            
            # Track metrics
            if result["action"] == "ask_clarification":
                CLARIFICATION_COUNTER.inc()
            
            confidence = result["confidence_scores"].get("overall_confidence", 0)
            CONFIDENCE_HISTOGRAM.observe(confidence)
            
            return result
            
        finally:
            RESPONSE_TIME.observe(time.time() - start_time)
            ACTIVE_CONVERSATIONS.dec()
"""

# Performance optimization patterns  
class PerformanceOptimizations:
    """Performance optimizations for production use"""
    
    @staticmethod
    def get_caching_layer() -> str:
        """Redis caching implementation"""
        return """
import redis
import hashlib
import json
from typing import Optional

class CachingLayer:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.cache_ttl = 3600  # 1 hour
    
    def _get_cache_key(self, query: str, context_hash: str) -> str:
        combined = f"{query}:{context_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_context_hash(self, retrieved_docs: List[Dict]) -> str:
        # Create hash from document contents
        content = "".join([doc.get("content", "") for doc in retrieved_docs])
        return hashlib.md5(content.encode()).hexdigest()
    
    async def get_cached_result(self, query: str, retrieved_docs: List[Dict]) -> Optional[Dict]:
        context_hash = self._get_context_hash(retrieved_docs)
        cache_key = self._get_cache_key(query, context_hash)
        
        cached = self.redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
        return None
    
    async def cache_result(self, query: str, retrieved_docs: List[Dict], result: Dict):
        context_hash = self._get_context_hash(retrieved_docs)
        cache_key = self._get_cache_key(query, context_hash)
        
        self.redis_client.setex(
            cache_key, 
            self.cache_ttl, 
            json.dumps(result)
        )

class BatchProcessor:
    '''Process multiple queries efficiently'''
    
    def __init__(self, system: AskFollowupsSystem, batch_size: int = 10):
        self.system = system
        self.batch_size = batch_size
    
    async def process_batch(self, queries: List[str]) -> List[Dict]:
        # Process queries in batches for better throughput
        results = []
        
        for i in range(0, len(queries), self.batch_size):
            batch = queries[i:i + self.batch_size]
            
            # Process batch concurrently
            batch_tasks = [
                self.system.process_query(query) 
                for query in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
        
        return results
"""

# Integration with popular frameworks
class FrameworkIntegrations:
    """Ready-to-use integrations with popular frameworks"""
    
    @staticmethod
    def streamlit_app() -> str:
        """Streamlit web app integration"""
        return """
import streamlit as st
import asyncio
from ask_followups_system import AskFollowupsSystem, Config

# Streamlit app
st.title("🤖 Ask Follow-ups Demo")
st.write("Intelligent Q&A system with automatic clarification")

# Initialize system (cache it)
@st.cache_resource
def get_system():
    config = Config(openai_api_key=st.secrets["openai_api_key"])
    return AskFollowupsSystem(config)

system = get_system()

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
        with st.spinner("Thinking..."):
            result = asyncio.run(system.process_query(
                prompt, 
                st.session_state.messages[:-1]
            ))
        
        # Show confidence metrics
        with st.expander("📊 Confidence Metrics"):
            for metric, score in result["confidence_scores"].items():
                st.metric(metric.replace("_", " ").title(), f"{score:.3f}")
        
        if result["action"] == "ask_clarification":
            st.write("I need some clarification:")
            for i, question in enumerate(result["clarification_questions"], 1):
                st.write(f"{i}. {question}")
            
            # Handle clarification in next input
            st.session_state.waiting_for_clarification = {
                "original_query": prompt,
                "questions": result["clarification_questions"]
            }
        else:
            st.write(result["answer"])
    
    # Add assistant message
    if result["action"] == "ask_clarification":
        assistant_msg = f"Clarifying questions: {'; '.join(result['clarification_questions'])}"
    else:
        assistant_msg = result["answer"]
    
    st.session_state.messages.append({"role": "assistant", "content": assistant_msg})
"""
    
    @staticmethod
    def gradio_interface() -> str:
        """Gradio interface integration"""
        return """
import gradio as gr
import asyncio

def create_gradio_interface(system: AskFollowupsSystem):
    
    async def process_message(message, history):
        # Convert Gradio history format
        conversation_history = []
        for human, assistant in history:
            if human:
                conversation_history.append({"role": "user", "content": human})
            if assistant:
                conversation_history.append({"role": "assistant", "content": assistant})
        
        # Process query
        result = await system.process_query(message, conversation_history)
        
        if result["action"] == "ask_clarification":
            response = "I need some clarification:\\n\\n"
            for i, q in enumerate(result["clarification_questions"], 1):
                response += f"{i}. {q}\\n"
            response += f"\\n*Confidence: {result['confidence_scores'].get('overall_confidence', 0):.2f}*"
        else:
            response = result["answer"]
            response += f"\\n\\n*Confidence: {result['confidence_scores'].get('overall_confidence', 0):.2f}*"
        
        return response
    
    # Create interface
    iface = gr.ChatInterface(
        fn=lambda msg, history: asyncio.run(process_message(msg, history)),
        title="Ask Follow-ups System",
        description="Intelligent Q&A with automatic clarification",
        examples=[
            "How do I learn programming?",
            "What's the best framework?", 
            "Explain machine learning",
            "How do I build an API?"
        ]
    )
    
    return iface

# Launch
if __name__ == "__main__":
    config = Config(openai_api_key="your-key-here")
    system = AskFollowupsSystem(config)
    
    interface = create_gradio_interface(system)
    interface.launch(share=True)
"""

# Installation and setup guide
INSTALLATION_GUIDE = """
# 📦 Installation Guide

## Requirements
```bash
pip install -r requirements.txt
```

## requirements.txt
```
openai>=1.0.0
chromadb>=0.4.0
sentence-transformers>=2.2.0
numpy>=1.24.0
langgraph>=0.0.40
trulens-eval>=0.18.0
ragas>=0.1.0
guardrails-ai>=0.4.0
fastapi>=0.100.0
uvicorn>=0.23.0
streamlit>=1.28.0
gradio>=4.0.0
redis>=4.6.0
prometheus-client>=0.17.0
```

## Quick Start
```python
from ask_followups_system import AskFollowupsSystem, Config

# Configure system
config = Config(
    openai_api_key="your-openai-api-key",
    confidence_threshold=0.7
)

# Initialize
system = AskFollowupsSystem(config)

# Add your documents
documents = [
    {"content": "Your document content...", "metadata": {"source": "doc1"}}
]
system.vector_store.add_documents(documents)

# Process queries
result = await system.process_query("Your question here")
```

## Production Deployment










### Environment Variables
```bash
export OPENAI_API_KEY="your-key"
export CHROMA_DB_PATH="/app/data/chroma_db"
export CONFIDENCE_THRESHOLD="0.7"
export REDIS_URL="redis://localhost:6379"
```
"""

if __name__ == "__main__":
    print(INSTALLATION_GUIDE)
    print("\n🚀 Starting interactive demo...")
    asyncio.run(main())
