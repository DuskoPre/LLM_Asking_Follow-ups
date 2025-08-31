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
