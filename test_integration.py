#!/usr/bin/env python3
"""
Test script to verify LLM backend integration.
This script tests both Ollama and Transformers backends.
"""

import json
import os
import sys
from typing import Dict, Any

def test_backend_integration():
    """Test the integration of both LLM backends."""
    
    # Create test data directory
    test_dir = "test_data"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a simple test prompt
    test_prompts = {
        "prompts": [
            {
                "id": "test_prompt_1",
                "text": "Write a Python function to calculate the factorial of a number."
            }
        ]
    }
    
    # Save test prompts
    test_file = os.path.join(test_dir, "test_prompts.json")
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_prompts, f, ensure_ascii=False, indent=2)
    
    print("Created test prompts file:", test_file)
    
    # Test imports
    try:
        from pipeline import process_single_prompt, ensure_dirs
        print("✓ Pipeline imports successful")
    except ImportError as e:
        print(f"✗ Pipeline import failed: {e}")
        return False
    
    try:
        from LLM_strip import query_model as ollama_query
        print("✓ Ollama backend import successful")
    except ImportError as e:
        print(f"✗ Ollama backend import failed: {e}")
    
    try:
        from LLM_transformers import query_model as transformers_query
        print("✓ Transformers backend import successful")
    except ImportError as e:
        print(f"✗ Transformers backend import failed: {e}")
    
    # Test prompt data structure
    test_prompt_data = {
        'id': 'test_integration',
        'text': 'Write a simple hello world function in Python.'
    }
    
    project_root = ensure_dirs()
    data_dir = os.path.join(project_root, "data")
    
    print("\nTesting backend integration...")
    print("=" * 50)
    
    # Test with Ollama backend (if available)
    try:
        print("\nTesting Ollama backend...")
        process_single_prompt(test_prompt_data, data_dir, backend="ollama")
        print("✓ Ollama backend test completed")
    except Exception as e:
        print(f"✗ Ollama backend test failed: {e}")
        print("  (This is expected if Ollama is not running)")
    
    # Test with Transformers backend (if available)
    try:
        print("\nTesting Transformers backend...")
        process_single_prompt(test_prompt_data, data_dir, backend="transformers", model_name="microsoft/DialoGPT-medium")
        print("✓ Transformers backend test completed")
    except Exception as e:
        print(f"✗ Transformers backend test failed: {e}")
        print("  (This is expected if transformers/torch are not installed)")
    
    print("\n" + "=" * 50)
    print("Integration test completed!")
    print("\nTo run the full pipeline:")
    print("  python pipeline.py --backend ollama test_data/test_prompts.json")
    print("  python pipeline.py --backend transformers --model microsoft/DialoGPT-medium test_data/test_prompts.json")
    
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    dependencies = {
        'requests': 'requests',
        'transformers': 'transformers', 
        'torch': 'torch',
        'googletrans': 'googletrans',
        'matplotlib': 'matplotlib',
        'numpy': 'numpy'
    }
    
    missing = []
    for name, module in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} (missing)")
            missing.append(name)
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install them with: pip install " + " ".join(missing))
        return False
    else:
        print("\n✓ All dependencies are installed!")
        return True


if __name__ == "__main__":
    print("LLM Backend Integration Test")
    print("=" * 40)
    
    # Check dependencies first
    deps_ok = check_dependencies()
    
    if deps_ok:
        # Run integration test
        test_backend_integration()
    else:
        print("\nPlease install missing dependencies before running the integration test.")

