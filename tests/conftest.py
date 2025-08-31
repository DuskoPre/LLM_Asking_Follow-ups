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
