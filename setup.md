# 🚀 Ask Follow-ups System - Complete Setup Guide

## 📋 Prerequisites

Before starting, ensure you have:
- **Python 3.9+** installed ([Download Python](https://python.org/downloads/))
- **Git** installed ([Download Git](https://git-scm.com/downloads))
- **OpenAI API key** ([Get API key](https://platform.openai.com/api-keys))
- **GitHub account** (for repository hosting)
- **Docker** installed (optional, for containerized deployment)

## 🏗️ Step 1: Repository Setup

### 1.1 Create GitHub Repository
```bash
# Option A: Create new repository on GitHub, then clone
git clone https://github.com/yourusername/ask-followups-system.git
cd ask-followups-system

# Option B: Create local directory and push to GitHub
mkdir ask-followups-system
cd ask-followups-system
git init
git checkout -b main
```

### 1.2 Create Project Structure
```bash
# Create all necessary directories
mkdir -p src/{core,evaluation,retrieval,clarification,api}
mkdir -p tests
mkdir -p deployment/{docker,kubernetes,scripts,monitoring}
mkdir -p docs
mkdir -p examples
mkdir -p data
mkdir -p logs

# Create __init__.py files
touch src/__init__.py
touch src/core/__init__.py
touch src/evaluation/__init__.py
touch src/retrieval/__init__.py
touch src/clarification/__init__.py
touch src/api/__init__.py
touch tests/__init__.py
```

### 1.3 Add Essential Files
Copy the provided files to your project:
- `.env.example` → Root directory
- `.gitignore` → Root directory
- `requirements.txt` → Root directory
- `requirements-dev.txt` → Root directory
- `docker-compose.yml` → Root directory
- `Dockerfile` → Root directory
- `.github/workflows/ci.yml` → `.github/workflows/` directory
- `app.py` → Root directory (main implementation)
- `test_setup.py` → Root directory
- `scripts/setup.sh` → `scripts/` directory

## 🔧 Step 2: Local Environment Setup

### 2.1 Python Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Verify activation
which python  # Should point to venv/bin/python
```

### 2.2 Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install main dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Verify installation
pip list | grep -E "(openai|chromadb|langgraph|trulens|ragas)"
```

### 2.3 Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your actual values
nano .env  # or use your preferred editor
```

**Required environment variables:**
```env
# CRITICAL: Replace with your actual OpenAI API key
OPENAI_API_KEY=sk-your-actual-openai-api-key-here

# System thresholds (adjust as needed)
CONFIDENCE_THRESHOLD=0.7
GROUNDEDNESS_THRESHOLD=0.6
CONTEXT_RELEVANCE_THRESHOLD=0.6
ANSWER_RELEVANCE_THRESHOLD=0.7
MAX_CLARIFICATION_ATTEMPTS=3

# Database and caching
CHROMA_DB_PATH=./data/chroma_db
REDIS_URL=redis://localhost:6379

# API settings
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

## 📊 Step 3: Initialize Database and Sample Data

### 3.1 Create Sample Knowledge Base
```bash
# Create sample documents file
cat > data/sample_docs.json << 'EOF'
[
  {
    "content": "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python has extensive libraries for web development, data science, machine learning, and automation.",
    "metadata": {"source": "python_guide", "category": "programming", "difficulty": "beginner"}
  },
  {
    "content": "FastAPI is a modern, fast web framework for building APIs with Python 3.6+ based on standard Python type hints. Key features include automatic API documentation, data validation, serialization, and high performance. It's designed to be easy to use while being production-ready.",
    "metadata": {"source": "fastapi_docs", "category": "web_development", "difficulty": "intermediate"}
  },
  {
    "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. Common approaches include supervised learning (with labeled data), unsupervised learning (finding patterns), and reinforcement learning (learning through rewards).",
    "metadata": {"source": "ml_handbook", "category": "ai_ml", "difficulty": "intermediate"}
  },
  {
    "content": "React is a free and open-source JavaScript library for building user interfaces, particularly web applications. It uses a component-based architecture and virtual DOM for efficient rendering. React applications are built using reusable components that manage their own state.",
    "metadata": {"source": "react_docs", "category": "frontend", "difficulty": "intermediate"}
  }
]
EOF
```

### 3.2 Create Data Loading Script
```bash
# Create data loading script
cat > scripts/load_sample_data.py << 'EOF'
import json
import os
import sys
sys.path.append('.')

from src.core.config import Config
from src.core.system import AskFollowupsSystem

def load_sample_data():
    """Load sample documents into the vector store"""
    
    # Load configuration
    config = Config(
        openai_api_key=os.getenv("OPENAI_API_KEY", "placeholder"),
        vector_db_path="./data/chroma_db"
    )
    
    # Initialize system
    system = AskFollowupsSystem(config)
    
    # Load sample documents
    with open("data/sample_docs.json", "r") as f:
        documents = json.load(f)
    
    # Add to vector store
    system.vector_store.add_documents(documents)
    
    print(f"✅ Loaded {len(documents)} sample documents")
    return len(documents)

if __name__ == "__main__":
    load_sample_data()
EOF

chmod +x scripts/load_sample_data.py
```

## 🧪 Step 4: Test the Setup

### 4.1 Basic System Test
```bash
# Run the setup test
python test_setup.py
```

**Expected output:**
```
🧪 Testing Ask Follow-ups System Setup
==================================================
1. Initializing system...
✅ System initialized successfully
2. Loading sample documents...
✅ Loaded 4 sample documents
3. Testing query processing...

   Test 1: 'How do I learn programming?'
   Action: ask_clarification
   Confidence: 0.451
   Questions: 2

   Test 2: 'What is Python used for?'
   Action: provide_answer
   Confidence: 0.823
   Answer length: 245 chars

✅ All tests passed! System is working correctly.
```

### 4.2 Interactive Demo Test
```bash
# Run the interactive demo from app.py
python app.py
```

This will start an interactive session where you can test queries like:
- "How do I learn?" (should trigger clarification)
- "What is FastAPI?" (should provide direct answer)
- "Which framework is best?" (should ask for clarification)

## 🌐 Step 5: API Server Setup

### 5.1 Create FastAPI Application
Create `src/api/main.py`:
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from ..core.config import Config
from ..core.system import AskFollowupsSystem

# Initialize FastAPI app
app = FastAPI(
    title="Ask Follow-ups API",
    description="Intelligent Q&A system with automatic clarification",
    version="1.0.0"
)

# Request/Response models
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

# Initialize system
config = Config(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
)
system = AskFollowupsSystem(config)

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query"""
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
    """Process clarification response"""
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
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Ask Follow-ups API", "docs": "/docs"}
```

### 5.2 Start the API Server
```bash
# Start development server with auto-reload
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Test the API
curl http://localhost:8000/health
curl http://localhost:8000/  # Should return API info

# View interactive API docs
open http://localhost:8000/docs
```

## 🐳 Step 6: Docker Setup (Optional)

### 6.1 Build and Run with Docker
```bash
# Create .env file with your actual API key
cp .env.example .env
# Edit .env with your OpenAI API key

# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f app
```

### 6.2 Test Docker Deployment
```bash
# Test API endpoint
curl http://localhost:8000/health

# Test query endpoint
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I learn programming?"}'
```

## 🎨 Step 7: User Interface Options

### 7.1 Streamlit Web App
Create `examples/streamlit_app.py`:
```python
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
```

**Run Streamlit app:**
```bash
streamlit run examples/streamlit_app.py
```

### 7.2 Gradio Interface
Create `examples/gradio_demo.py`:
```python
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
```

**Run Gradio interface:**
```bash
python examples/gradio_demo.py
```

## 🧪 Step 8: Testing and Validation

### 8.1 Run Unit Tests
```bash
# Create basic test structure
mkdir -p tests

# Create conftest.py for test configuration
cat > tests/conftest.py << 'EOF'
import pytest
import os
from src.core.config import Config

@pytest.fixture
def test_config():
    """Test configuration"""
    return Config(
        openai_api_key=os.getenv("OPENAI_API_KEY", "test-key"),
        confidence_threshold=0.7,
        vector_db_path="./test_data/chroma_db"
    )

@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        {
            "content": "Python is a programming language",
            "metadata": {"source": "test", "category": "programming"}
        }
    ]
EOF

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

### 8.2 Integration Testing
```bash
# Test the complete workflow
python -c "
import asyncio
from app import AskFollowupsSystem, Config
import os

async def test():
    config = Config(openai_api_key=os.getenv('OPENAI_API_KEY'))
    system = AskFollowupsSystem(config)
    
    # Add sample data
    docs = [{'content': 'Test document', 'metadata': {}}]
    system.vector_store.add_documents(docs)
    
    # Test query
    result = await system.process_query('What is programming?')
    print(f'Action: {result[\"action\"]}')
    print(f'Confidence: {result[\"confidence_scores\"]}')

asyncio.run(test())
"
```

## 🔧 Step 9: Development Tools Setup

### 9.1 Code Quality Tools
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
EOF

# Test pre-commit
pre-commit run --all-files
```

### 9.2 Development Scripts
```bash
# Make setup script executable
chmod +x scripts/setup.sh

# Run setup script
./scripts/setup.sh
```

## 🚀 Step 10: Deployment Options

### 10.1 Local Development Server
```bash
# Start API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Start Streamlit (in another terminal)
streamlit run examples/streamlit_app.py

# Start Gradio (alternative)
python examples/gradio_demo.py
```

### 10.2 Docker Production Deployment
```bash
# Set your OpenAI API key in .env
echo "OPENAI_API_KEY=sk-your-actual-key-here" > .env

# Build and start services
docker-compose up --build -d

# Check all services are running
docker-compose ps

# Test the deployment
curl http://localhost:8000/health
curl http://localhost:8000/docs
```

### 10.3 Cloud Deployment

**Heroku:**
```bash
# Install Heroku CLI, then:
heroku create your-app-name
heroku config:set OPENAI_API_KEY=your_actual_key_here
heroku config:set REDIS_URL=redis://your-redis-url

# For container deployment:
heroku container:push web
heroku container:release web

# Check status
heroku logs --tail
```

**Google Cloud Run:**
```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/your-project-id/ask-followups

# Deploy to Cloud Run
gcloud run deploy ask-followups \
  --image gcr.io/your-project-id/ask-followups \
  --platform managed \
  --region us-central1 \
  --set-env-vars OPENAI_API_KEY=your_key_here \
  --allow-unauthenticated
```

## 📊 Step 11: Monitoring Setup

### 11.1 Access Monitoring Dashboards
If using Docker Compose with monitoring:

```bash
# Prometheus metrics
open http://localhost:9090

# Grafana dashboards  
open http://localhost:3000
# Login: admin/admin
```

### 11.2 Health Monitoring
```bash
# Check API health
curl http://localhost:8000/health

# Check metrics (if implemented)
curl http://localhost:8000/metrics

# Monitor logs
docker-compose logs -f app
```

## 🎯 Step 12: Usage Examples

### 12.1 API Usage
```bash
# Test basic query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I learn programming?",
    "conversation_history": []
  }'

# Test clarification flow
curl -X POST "http://localhost:8000/clarify" \
  -H "Content-Type: application/json" \
  -d '{
    "original_query": "How do I learn?",
    "clarification_response": "I want to learn Python for web development",
    "conversation_history": []
  }'
```

### 12.2 Python Integration
```python
import asyncio
from src.core.config import Config
from src.core.system import AskFollowupsSystem

async def example_usage():
    # Initialize system
    config = Config(openai_api_key="your-key-here")
    system = AskFollowupsSystem(config)
    
    # Add your documents
    documents = [
        {
            "content": "Your knowledge base content here...",
            "metadata": {"source": "your_source", "category": "your_category"}
        }
    ]
    system.vector_store.add_documents(documents)
    
    # Process queries
    result = await system.process_query("Your question here")
    
    if result["action"] == "ask_clarification":
        print("Clarification needed:")
        for q in result["clarification_questions"]:
            print(f"- {q}")
    else:
        print(f"Answer: {result['answer']}")

# Run example
asyncio.run(example_usage())
```

## 🆘 Troubleshooting Common Issues

### Issue 1: OpenAI API Key Problems
```bash
# Test your API key
python -c "
import openai
import os
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
try:
    response = client.models.list()
    print('✅ API key is valid')
except Exception as e:
    print(f'❌ API key error: {e}')
"
```

### Issue 2: Import Errors
```bash
# Install in development mode
pip install -e .

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

### Issue 3: ChromaDB Permission Issues
```bash
# Fix data directory permissions
sudo chown -R $USER:$USER data/
chmod -R 755 data/
```

### Issue 4: Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process (replace PID)
kill -9 <PID>

# Or use different port
uvicorn src.api.main:app --port 8001
```

### Issue 5: Redis Connection Issues
```bash
# Start Redis locally
redis-server

# Or use Docker
docker run -d -p 6379:6379 redis:7-alpine

# Test Redis connection
redis-cli ping
```

## ✅ Verification Checklist

- [ ] Python environment activated
- [ ] All dependencies installed
- [ ] `.env` file configured with valid OpenAI API key
- [ ] Sample data loaded successfully
- [ ] `test_setup.py` passes all tests
- [ ] API server starts without errors
- [ ] Health check endpoint responds
- [ ] Interactive demo works
- [ ] Docker containers start (if using Docker)
- [ ] Web interface accessible

## 🎉 Next Steps

1. **Customize for your use case:**
   - Add your own documents to the knowledge base
   - Adjust confidence thresholds
   - Modify clarification strategies

2. **Production deployment:**
   - Set up proper secret management
   - Configure monitoring and alerting
   - Set up backup strategies

3. **Integration:**
   - Connect to your existing systems
   - Add authentication if needed
   - Scale based on usage patterns

## 📚 Additional Resources

- **API Documentation**: http://localhost:8000/docs (when running)
- **GitHub Repository**: Your repository URL
- **Issues/Support**: GitHub Issues page
- **OpenAI Documentation**: https://platform.openai.com/docs

---

**🎯 You're ready to go!** The Ask Follow-ups System is now set up and ready for development and deployment. The system will automatically ask clarifying questions when confidence is low, leading to better user experiences and more accurate responses.
