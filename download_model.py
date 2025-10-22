#!/usr/bin/env python3
"""
Script to download the Qwen3-4B-Instruct-2507 model to the local models folder.
"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def download_model():
    """Download the model to local models folder."""
    
    # Create models directory if it doesn't exist
    models_dir = "./models"
    local_model_path = "./models/Qwen3-4B-Instruct-2507"
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created directory: {models_dir}")
    
    print("Downloading Qwen3-4B-Instruct-2507 model...")
    print("This may take a while (model is ~1.5GB)...")
    
    try:
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B-Instruct-2507",
            cache_dir=local_model_path
        )
        
        # Download model
        print("Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-4B-Instruct-2507",
            cache_dir=local_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        print(f"✅ Model successfully downloaded to: {local_model_path}")
        print(f"📁 Model files are now available locally")
        
        # Test the model
        print("\n🧪 Testing the downloaded model...")
        test_prompt = "def hello_world():\n    return \"Hello, World!\""
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"✅ Model test successful!")
            print(f"Response: {response[:100]}...")
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Starting model download...")
    success = download_model()
    
    if success:
        print("\n🎉 Model download completed successfully!")
        print("You can now run LLMv2.py with the local model.")
    else:
        print("\n💥 Model download failed. Please check your internet connection and try again.")
