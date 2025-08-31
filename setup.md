# 🚀 Complete Setup Guide - Ask Follow-ups System

## 📋 Prerequisites

Before starting, ensure you have:
- Python 3.9+ installed
- Git installed
- GitHub account
- OpenAI API key
- Docker installed (optional, for containerized deployment)

## 🏗️ Step 1: GitHub Repository Setup

### 1.1 Create New Repository
```bash
# Create and navigate to your project directory
mkdir ask-followups-system
cd ask-followups-system

# Initialize git repository
git init

# Create main branch
git checkout -b main
```

### 1.2 Project Structure
Create the following directory structure:
```
ask-followups-system/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── models.py
│   │   └── system.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluators.py
│   │   └── metrics.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── vector_store.py
│   ├── clarification/
│   │   ├── __init__.py
│   │   └── generator.py
│   └── api/
│       ├── __init__.py
│       ├── main.py
│       └── routes.py
├── tests/
│   ├── __init__.py
│   ├── test_system.py
│   ├── test_evaluation.py
│   └── conftest.py
├── deployment/
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   ├── kubernetes/
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   └── scripts/
│       ├── setup.sh
│       └── deploy.sh
├── docs/
│   ├── README.md
│   ├── API.md
│   └── DEPLOYMENT.md
├── examples/
│   ├── streamlit_app.py
│   ├── gradio_demo.py
│   └── jupyter_notebook.ipynb
├── data/
│   └── sample_docs.json
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── setup.py
├── pyproject.toml
└── README.md
```

### 1.3 Essential Files Setup

**Create `.gitignore`:**
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Database
*.db
*.sqlite3
chroma_db/
data/chroma_db/

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# API Keys
*.key
secrets/
```

**Create `.env.example`:**
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# System Configuration  
CONFIDENCE_THRESHOLD=0.7
GROUNDEDNESS_THRESHOLD=0.6
CONTEXT_RELEVANCE_THRESHOLD=0.6
ANSWER_RELEVANCE_THRESHOLD=0.7
MAX_CLARIFICATION_ATTEMPTS=3

# Database
CHROMA_DB_PATH=./data/chroma_db
REDIS_URL=redis://localhost:6379

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Monitoring (optional)
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

**Create `requirements.txt`:**
```txt
# Core dependencies
openai>=1.12.0
chromadb>=0.4.22
sentence-transformers>=2.2.2
numpy>=1.24.3
pandas>=2.0.3

# LangGraph and LangChain
langgraph>=0.0.55
langchain>=0.1.17
langchain-openai>=0.1.7

# Evaluation frameworks
trulens-eval>=0.24.1
ragas>=0.1.7
datasets>=2.14.0

# Validation and structure
guardrails-ai>=0.4.5
pydantic>=2.5.0

# Web frameworks
fastapi>=0.104.1
uvicorn[standard]>=0.24.0
streamlit>=1.31.0
gradio>=4.15.0

# Caching and storage
redis>=5.0.1
aioredis>=2.0.1

# Monitoring and metrics
prometheus-client>=0.19.0
structlog>=23.2.0

# Development and testing
pytest>=7.4.0
pytest-asyncio>=0.21.1
black>=23.11.0
flake8>=6.1.0
mypy>=1.7.0
```

**Create `requirements-dev.txt`:**
```txt
-r requirements.txt

# Development tools
jupyter>=1.0.0
notebook>=7.0.0
ipykernel>=6.25.0

# Testing
pytest-cov>=4.1.0
pytest-mock>=3.12.0
httpx>=0.25.0

# Documentation
mkdocs>=1.5.0
mkdocs-material>=9.4.0

# Code quality
pre-commit>=3.6.0
isort>=5.12.0
bandit>=1.7.5
```

## 🔧 Step 2: Local Development Setup

### 2.1 Clone and Setup Environment
```bash
# Clone your repository (after pushing initial structure)
git clone https://github.com/yourusername/ask-followups-system.git
cd ask-followups-system

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2.2 Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your actual values
nano .env  # or use your preferred editor
```

### 2.3 Initialize Database and Sample Data
```bash
# Create data directory
mkdir -p data

# Create sample documents file
cat > data/sample_docs.json << 'EOF'
[
  {
    "content": "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms and has extensive libraries.",
    "metadata": {"source": "python_guide", "category": "programming", "difficulty": "beginner"}
  },
  {
    "content": "FastAPI is a modern, fast web framework for building APIs with Python. It provides automatic API documentation, type checking, and high performance.",
    "metadata": {"source": "fastapi_docs", "category": "web_development", "difficulty": "intermediate"}
  },
  {
    "content": "Machine learning involves algorithms that can learn patterns from data without explicit programming. Common techniques include supervised, unsupervised, and reinforcement learning.",
    "metadata": {"source": "ml_handbook", "category": "ai_ml", "difficulty": "intermediate"}
  }
]
EOF
```

## 🧪 Step 3: Testing the Setup

### 3.1 Run Unit Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

### 3.2 Manual Testing Script
Create `test_setup.py`:
```python
import asyncio
import json
import os
from dotenv import load_dotenv
from src.core.config import Config
from src.core.system import AskFollowupsSystem

async def test_setup():
    """Test the basic system setup"""
    
    # Load environment
    load_dotenv()
    
    # Create config
    config = Config(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        confidence_threshold=0.7
    )
    
    print("🧪 Testing Ask Follow-ups System Setup")
    print("=" * 50)
    
    try:
        # Initialize system
        print("1. Initializing system...")
        system = AskFollowupsSystem(config)
        print("✅ System initialized successfully")
        
        # Load sample data
        print("2. Loading sample documents...")
        with open("data/sample_docs.json", "r") as f:
            sample_docs = json.load(f)
        
        system.vector_store.add_documents(sample_docs)
        print(f"✅ Loaded {len(sample_docs)} sample documents")
        
        # Test query processing
        print("3. Testing query processing...")
        test_queries = [
            "How do I learn programming?",  # Should trigger clarification
            "What is Python used for?",     # Should provide direct answer
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Test {i}: '{query}'")
            result = await system.process_query(query)
            
            print(f"   Action: {result['action']}")
            print(f"   Confidence: {result['confidence_scores'].get('overall_confidence', 0):.3f}")
            
            if result['action'] == 'ask_clarification':
                print(f"   Questions: {len(result['clarification_questions'])}")
            else:
                print(f"   Answer length: {len(result['answer'])} chars")
        
        print("\n✅ All tests passed! System is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_setup())
    exit(0 if success else 1)
```

Run the test:
```bash
python test_setup.py
```

## 🐳 Step 4: Docker Setup

### 4.1 Create Dockerfile
```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY .env.example ./.env

# Create necessary directories
RUN mkdir -p /app/data/chroma_db /app/logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 4.2 Create docker-compose.yml
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - CHROMA_DB_PATH=/app/data/chroma_db
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # Optional: Monitoring stack
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./deployment/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  redis_data:
  grafana_data:
```

### 4.3 Build and Run
```bash
# Create .env file with your actual values
cp .env.example .env
# Edit .env with your OpenAI API key

# Build and run with Docker Compose
docker-compose up --build

# Or run in detached mode
docker-compose up -d

# Check logs
docker-compose logs -f app
```

## 🌐 Step 5: GitHub Actions CI/CD

### 5.1 Create `.github/workflows/ci.yml`
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
        options: --health-cmd="redis-cli ping" --health-interval=10s --health-timeout=5s --health-retries=3

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('requirements*.txt') }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Code quality checks
      run: |
        # Format check
        black --check src/ tests/
        
        # Lint check  
        flake8 src/ tests/
        
        # Type check
        mypy src/
        
        # Security check
        bandit -r src/
    
    - name: Run tests
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        REDIS_URL: redis://localhost:6379
      run: |
        pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Login to GitHub Container Registry
      uses: docker/login-action@v3
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: |
          ghcr.io/${{ github.repository }}:latest
          ghcr.io/${{ github.repository }}:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy to production
      run: |
        echo "Deploy to your preferred platform here"
        # Add your deployment script
```

### 5.2 Repository Secrets Setup
In your GitHub repository, go to Settings → Secrets and variables → Actions, and add:

- `OPENAI_API_KEY`: Your OpenAI API key
- `DOCKER_USERNAME`: Docker Hub username (if using Docker Hub)
- `DOCKER_PASSWORD`: Docker Hub password
- Any other deployment-specific secrets

## 🛠️ Step 6: Local Development Workflow

### 6.1 Development Setup Script
Create `scripts/setup.sh`:
```bash
#!/bin/bash
set -e

echo "🚀 Setting up Ask Follow-ups System for development"

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your actual API keys"
fi

# Create data directories
echo "Creating data directories..."
mkdir -p data/chroma_db logs

# Load sample data
echo "Loading sample data..."
python scripts/load_sample_data.py

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run tests: pytest tests/"
echo "3. Start development server: uvicorn src.api.main:app --reload"
echo "4. Or run Streamlit demo: streamlit run examples/streamlit_app.py"
```

### 6.2 Make it executable and run
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

## 🚀 Step 7: Deployment Options

### 7.1 Local Development Server
```bash
# Activate environment
source venv/bin/activate

# Start API server with auto-reload
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Or start Streamlit demo
streamlit run examples/streamlit_app.py

# Or start Gradio interface
python examples/gradio_demo.py
```

### 7.2 Docker Deployment
```bash
# Build image
docker build -t ask-followups:latest .

# Run with Docker Compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### 7.3 Cloud Deployment Options

**Heroku:**
```bash
# Install Heroku CLI, then:
heroku create your-app-name
heroku config:set OPENAI_API_KEY=your_key_here
heroku container:push web
heroku container:release web
```

**Google Cloud Run:**
```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/your-project/ask-followups

# Deploy to Cloud Run
gcloud run deploy ask-followups \
  --image gcr.io/your-project/ask-followups \
  --platform managed \
  --region us-central1 \
  --set-env-vars OPENAI_API_KEY=your_key_here
```

**AWS ECS/Fargate:**
```bash
# Push to Amazon ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-west-2.amazonaws.com

docker tag ask-followups:latest your-account.dkr.ecr.us-west-2.amazonaws.com/ask-followups:latest
docker push your-account.dkr.ecr.us-west-2.amazonaws.com/ask-followups:latest

# Deploy using ECS CLI or CloudFormation
```

## 📊 Step 8: Monitoring Setup

### 8.1 Health Checks
```bash
# Check API health
curl http://localhost:8000/health

# Check metrics endpoint
curl http://localhost:8000/metrics
```

### 8.2 Grafana Dashboard Setup
```bash
# Access Grafana (if using docker-compose)
open http://localhost:3000
# Login: admin/admin

# Import dashboard configuration from deployment/monitoring/grafana-dashboard.json
```

## 🔄 Step 9: Development Workflow

### 9.1 Feature Development
```bash
# Create feature branch
git checkout -b feature/new-evaluation-metric

# Make changes
# ... edit files ...

# Test changes
pytest tests/
black src/ tests/
flake8 src/ tests/

# Commit and push
git add .
git commit -m "feat: add new evaluation metric"
git push origin feature/new-evaluation-metric

# Create pull request on GitHub
```

### 9.2 Release Process
```bash
# Update version in setup.py and pyproject.toml
# Update CHANGELOG.md

# Create release branch
git checkout -b release/v1.1.0

# Final testing
pytest tests/
docker-compose up --build -d
# Run integration tests

# Merge to main
git checkout main
git merge release/v1.1.0

# Tag release
git tag v1.1.0
git push origin main --tags

# GitHub Actions will automatically build and deploy
```

## 🎯 Step 10: Production Checklist

### 10.1 Security
- [ ] API keys stored securely (environment variables/secrets)
- [ ] Input validation and sanitization implemented
- [ ] Rate limiting configured
- [ ] HTTPS enabled for production
- [ ] CORS properly configured
- [ ] Authentication/authorization if needed

### 10.2 Performance
- [ ] Redis caching enabled
- [ ] Database connection pooling
- [ ] Async processing for heavy operations  
- [ ] Load balancing configured
- [ ] CDN for static assets (if any)

### 10.3 Monitoring
- [ ] Health checks implemented
- [ ] Prometheus metrics exposed
- [ ] Grafana dashboards configured
- [ ] Alerting rules set up
- [ ] Log aggregation (ELK stack or similar)

### 10.4 Reliability
- [ ] Database backups automated
- [ ] Disaster recovery plan
- [ ] Auto-scaling configured
- [ ] Circuit breakers for external APIs
- [ ] Graceful degradation on failures

## 🆘 Troubleshooting Common Issues

### Issue 1: Import Errors
```bash
# Solution: Install in development mode
pip install -e .

# Or add src to Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
```

### Issue 2: OpenAI API Errors
```bash
# Check API key
python -c "import openai; print('API key valid' if openai.api_key else 'No API key')"

# Test connection
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Issue 3: ChromaDB Permission Issues
```bash
# Fix permissions
sudo chown -R $USER:$USER data/
chmod -R 755 data/
```

### Issue 4: Docker Build Issues
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache
```

## 📚 Next Steps

1. **Customize the system** for your specific use case
2. **Add your documents** to the knowledge base
3. **Fine-tune thresholds** based on your evaluation results
4. **Set up monitoring** for production usage
5. **Integrate with your existing systems** using the provided examples

## 📞 Support and Community

- **Documentation**: Check the `docs/` directory
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Use GitHub Discussions for questions
- **Contributing**: See CONTRIBUTING.md for guidelines

---

**🎉 You're all set!** Your ask follow-ups system is ready for development and deployment. The system will automatically ask clarifying questions when confidence is low, leading to better user experiences and more accurate responses.
