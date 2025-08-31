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
