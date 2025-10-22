# LLM Backend Usage Examples

This project now supports two LLM backends: Ollama and Transformers (Hugging Face).

## Backend Options

### 1. Ollama Backend (Default)
Uses local Ollama server running on `http://localhost:11434`

**Requirements:**
- Ollama installed and running
- Model pulled (e.g., `qwen3:8b`)

**Usage:**
```bash
# Using JSON file with Ollama (default)
python pipeline.py prompts_input.json

# Explicitly specify Ollama backend
python pipeline.py --backend ollama prompts_input.json

# Interactive mode with Ollama
python pipeline.py --backend ollama
```

### 2. Transformers Backend
Uses Hugging Face transformers models locally

**Requirements:**
- `transformers` library installed
- `torch` installed
- Sufficient GPU/CPU memory for the model

**Usage:**
```bash
# Using transformers backend with default model
python pipeline.py --backend transformers prompts_input.json

# Using specific transformers model
python pipeline.py --backend transformers --model microsoft/DialoGPT-medium prompts_input.json

# Using a different model
python pipeline.py --backend transformers --model gpt2 prompts_input.json

# Interactive mode with transformers
python pipeline.py --backend transformers --model microsoft/DialoGPT-medium
```

## Available Transformers Models

Some popular models you can use:

- `microsoft/DialoGPT-medium` (default)
- `microsoft/DialoGPT-large`
- `gpt2`
- `gpt2-medium`
- `facebook/blenderbot-400M-distill`
- `microsoft/DialoGPT-small`

## Model Selection Guidelines

### For Code Generation:
- **DialoGPT models**: Good for conversational code generation
- **GPT-2 models**: General purpose, decent for code
- **Code-specific models**: Look for models trained specifically on code

### For Performance:
- **Small models** (`-small`, `-medium`): Faster, less memory
- **Large models** (`-large`, `-xl`): Better quality, more memory

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

For GPU support with transformers:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Configuration

### Ollama Configuration
Edit `LLM_strip.py` to change:
- `OLLAMA_URL`: Default is `http://localhost:11434/api/generate`
- `MODEL`: Default is `qwen3:8b`

### Transformers Configuration
Edit `LLM_transformers.py` to change:
- Default model in `query_model()` function
- Device selection (`auto`, `cpu`, `cuda`)
- Generation parameters (temperature, max_length, etc.)

## Troubleshooting

### Ollama Issues:
- Ensure Ollama is running: `ollama serve`
- Check if model is pulled: `ollama list`
- Pull a model: `ollama pull qwen3:8b`

### Transformers Issues:
- Check GPU availability: `torch.cuda.is_available()`
- Monitor memory usage during model loading
- Try smaller models if memory is insufficient
- Use CPU if GPU memory is limited

## Performance Comparison

| Backend | Speed | Memory | Quality | Setup |
|---------|-------|--------|---------|-------|
| Ollama | Fast | Medium | High | Easy |
| Transformers | Variable | High | Variable | Medium |

Choose Ollama for:
- Quick setup
- Consistent performance
- Lower memory usage

Choose Transformers for:
- Offline usage
- Custom model fine-tuning
- Research/experimentation

