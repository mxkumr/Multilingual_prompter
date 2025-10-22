"""
Multilingual Prompter Pipeline

This pipeline processes prompts through translation, LLM generation, and analysis.

Usage:
    python pipeline.py prompts.json --model Qwen/Qwen3-4B-Instruct-2507
    python pipeline.py prompts.json  # Uses default model

Features:
    - Uses LLMv2.py with Hugging Face transformers models
    - Code extraction and retry logic
    - GPU optimization with automatic device mapping
    - Multi-language prompt translation
    - Tree-sitter code parsing and analysis
"""

import os
import sys
import json
import asyncio
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List


def ensure_dirs() -> str:
    project_root = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    return project_root


def ensure_prompt_dir(data_dir: str, prompt_id: str) -> str:
    """Create and return the directory path for a specific prompt."""
    prompt_dir = os.path.join(data_dir, prompt_id)
    os.makedirs(prompt_dir, exist_ok=True)
    return prompt_dir


def load_prompts_from_json(json_file: str) -> List[Dict[str, str]]:
    """Load prompts from JSON file."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'prompts' not in data:
            raise ValueError("JSON file must contain a 'prompts' key")
        
        prompts = []
        for prompt_data in data['prompts']:
            if not all(key in prompt_data for key in ['id', 'text']):
                raise ValueError("Each prompt must have 'id' and 'text' keys")
            prompts.append({
                'id': prompt_data['id'],
                'text': prompt_data['text']
            })
        
        return prompts
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file '{json_file}' not found")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")


def translate_prompt(prompt_text: str) -> Dict[str, Optional[str]]:
    # Delegate to Prompt_translation.translate_prompt (async) with TARGET_LANG_CODES
    from Prompt_translation import translate_prompt as pt_translate_prompt, TARGET_LANG_CODES

    try:
        return asyncio.run(pt_translate_prompt(prompt_text, TARGET_LANG_CODES))
    except RuntimeError:
        # Fallback if an event loop is already running
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(pt_translate_prompt(prompt_text, TARGET_LANG_CODES))


def query_llm_for_translations(translations: Dict[str, str], model_name: str = None) -> Dict[str, str]:
    """
    Query LLM for translations using LLMv2.py with transformers models.
    
    Args:
        translations: Dictionary of language -> prompt translations
        model_name: Model name (defaults to Qwen/Qwen3-4B-Instruct-2507)
    """
    outputs: Dict[str, str] = {}
    
    # Use LLMv2.py with transformers models
    from LLMv2 import generate_code_with_retry, extract_code_from_response
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    # Initialize model and tokenizer
    model_name = model_name or "Qwen/Qwen3-4B-Instruct-2507"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    def query_func(prompt):
        return generate_code_with_retry(tokenizer, model, prompt, device)
    
    for lang, prompt in translations.items():
        if not prompt:
            outputs[lang] = None
            continue
        try:
            print(f"Querying transformers LLM for {lang}...")
            start_perf = time.perf_counter()
            start_wall = time.time()
            try:
                result = query_func(prompt)
                end_wall = time.time()
                duration = time.perf_counter() - start_perf
                log_llm_duration(lang, start_wall, end_wall, duration, success=True)
                outputs[lang] = result
                print(f"Generated code for {lang}: {str(result)[:100]}...")
            except Exception as inner_e:
                end_wall = time.time()
                duration = time.perf_counter() - start_perf
                log_llm_duration(lang, start_wall, end_wall, duration, success=False, error_message=str(inner_e))
                print(f"LLM query failed for {lang}: {inner_e}")
                outputs[lang] = None
            time.sleep(1)
        except Exception as e:
            print(f"LLM query failed for {lang}: {e}")
            outputs[lang] = None
    return outputs


def parse_llm_outputs(outputs: Dict[str, str]) -> Dict[str, Any]:
    # Reuse parser.parse_code_files_with_multilang_parser
    from parser import parse_code_files_with_multilang_parser
    print("Parsing code snippets with Tree-sitter...")
    return parse_code_files_with_multilang_parser(outputs)


def visualize_language_distribution() -> None:
    # Reuse non_english.main to generate charts and summary
    import non_english
    print("Generating language charts...")
    non_english.main()


def setup_logger() -> None:
    """Configure logging to file for LLM runtimes."""
    logging.basicConfig(
        filename="data/llm_runtime.log",
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s\t%(levelname)s\t%(message)s",
    )
    # Reduce noise from HTTP client libraries and others
    for noisy in ("httpx", "urllib3", "requests", "googletrans"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_llm_duration(language: str, start_ts: float, end_ts: float, seconds: float, success: bool = True, error_message: Optional[str] = None) -> None:
    """Log start time, end time, and total duration (in minutes) for an LLM run."""
    start_iso = datetime.fromtimestamp(start_ts).isoformat(timespec="seconds")
    end_iso = datetime.fromtimestamp(end_ts).isoformat(timespec="seconds")
    minutes = seconds / 60.0
    if success:
        logging.info(
            f"lang={language}\tstart={start_iso}\tend={end_iso}\tduration_min={minutes:.3f}"
        )
    else:
        logging.error(
            f"lang={language}\tstart={start_iso}\tend={end_iso}\tduration_min={minutes:.3f}\terror={error_message}"
        )


def process_single_prompt(prompt_data: Dict[str, str], data_dir: str, model_name: str = None) -> None:
    """Process a single prompt through the entire pipeline."""
    prompt_id = prompt_data['id']
    prompt_text = prompt_data['text']
    
    print(f"\n{'='*60}")
    print(f"Processing Prompt ID: {prompt_id}")
    print(f"Using model: {model_name or 'Qwen/Qwen3-4B-Instruct-2507 (default)'}")
    print("Features: Code extraction, retry logic, GPU optimization")
    print(f"{'='*60}")
    
    # Create prompt-specific directory
    prompt_dir = ensure_prompt_dir(data_dir, prompt_id)
    
    # Normalize prompt formatting
    try:
        from Prompt_translation import normalize_text
        prompt_text = normalize_text(prompt_text)
    except Exception:
        # Fallback to original prompt if normalization import fails
        pass
    
    # 1) Translate
    print("Translating prompt to multiple languages...")
    translations = translate_prompt(prompt_text)
    translated_path = os.path.join(prompt_dir, "translated_prompts.json")
    with open(translated_path, "w", encoding="utf-8") as f:
        json.dump(translations, f, ensure_ascii=False, indent=2)
    print(f"Saved translations to {translated_path}")
    
    # 2) Query LLM (Transformers with LLMv2.py)
    llm_outputs = query_llm_for_translations(translations, model_name=model_name)
    llm_out_path = os.path.join(prompt_dir, "llm_output.json")
    with open(llm_out_path, "w", encoding="utf-8") as f:
        json.dump(llm_outputs, f, ensure_ascii=False, indent=2)
    print(f"Saved LLM outputs to {llm_out_path}")
    
    # 3) Parse
    parsed = parse_llm_outputs(llm_outputs)
    parsed_path = os.path.join(prompt_dir, "llm_parsed.json")
    with open(parsed_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)
    print(f"Saved parsed results to {parsed_path}")
    
    # 4) Visualize
    visualize_language_distribution_for_prompt(prompt_dir)
    print(f"Completed processing for prompt ID: {prompt_id}")


def visualize_language_distribution_for_prompt(prompt_dir: str) -> None:
    """Generate language charts for a specific prompt directory."""
    # Use non_english.run_visualization with per-prompt paths
    import non_english
    print("Generating language charts...")
    input_path = os.path.join(prompt_dir, "llm_parsed.json")
    charts_dir = os.path.join(prompt_dir, "language_charts")
    summary_out = os.path.join(prompt_dir, "non_english_summary.json")
    non_english.run_visualization(input_path, charts_dir, summary_out)


def main() -> None:
    project_root = ensure_dirs()
    data_dir = os.path.join(project_root, "data")
    setup_logger()

    # Parse command line arguments
    model_name = None
    
    # Simple argument parsing
    args = sys.argv[1:]
    if "--model" in args:
        model_idx = args.index("--model")
        if model_idx + 1 < len(args):
            model_name = args[model_idx + 1]
    
    # Filter out our custom arguments to find the JSON file
    json_file = None
    for arg in args:
        if not arg.startswith("--") and arg.endswith(".json"):
            json_file = arg
            break

    # Check for JSON input file
    if json_file:
        try:
            prompts = load_prompts_from_json(json_file)
            print(f"Loaded {len(prompts)} prompts from {json_file}")
            
            # Process each prompt
            for i, prompt_data in enumerate(prompts, 1):
                print(f"\nProcessing prompt {i}/{len(prompts)}")
                process_single_prompt(prompt_data, data_dir, model_name=model_name)
            
            print(f"\n{'='*60}")
            print("All prompts processed successfully!")
            print(f"Results saved in individual folders under: {data_dir}")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"Error processing JSON file: {e}")
            return
    else:
        # Fallback to single prompt input for backward compatibility
        prompt_text = input("Enter the base prompt (in English): ").strip()
        if not prompt_text:
            print("Empty prompt; aborting.")
            return

        # Create a single prompt data structure
        prompt_data = {
            'id': 'single_prompt',
            'text': prompt_text
        }
        
        process_single_prompt(prompt_data, data_dir, model_name=model_name)
        print("Pipeline complete.")


if __name__ == "__main__":
    main()


