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
