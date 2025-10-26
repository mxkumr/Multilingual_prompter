from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import time
import re
import os
from datetime import datetime
import logging

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"  # Use Hugging Face default cache directory 

def extract_code_from_response(response_text):
    """Extract only the code from the LLM response, removing thinking/narrative.

    Strategy:
    1) Strip <think>...</think> blocks if present.
    2) Prefer fenced code blocks (any language). If multiple, return the first clean one.
    3) If no fences, try to detect code by finding the first code token and return from there.
    4) If nothing code-like is detected, return an empty string to avoid prose leakage.
    """
    if not response_text:
        return ""

    # Remove thinking process and common preambles
    response_text = re.sub(r"<think>[\s\S]*?</think>", "", response_text, flags=re.DOTALL)
    response_text = re.sub(r"(?i)^(thoughts?|reasoning|analysis)\s*:.*?$", "", response_text, flags=re.MULTILINE)
    
    # Remove repetitive patterns (like the output shows)
    response_text = re.sub(r"(That's the function\.|The code is correct\.|The function is correct\.|Yes, the function returns.*?Therefore, the code is correct\.)", "", response_text, flags=re.MULTILINE)
    response_text = re.sub(r"Sample Input:.*?Sample Output:.*?(?=\n|$)", "", response_text, flags=re.DOTALL)

    # Prefer fenced code blocks of any language
    fenced_blocks = re.findall(r"```[a-zA-Z0-9_+-]*\s*([\s\S]*?)\s*```", response_text)
    if fenced_blocks:
        # Choose the first clean block (not repetitive)
        for block in fenced_blocks:
            clean_block = block.strip()
            if clean_block and len(clean_block) > 10:  # Avoid empty or very short blocks
                return clean_block
        # If no clean block found, return the first one
        return fenced_blocks[0].strip() if fenced_blocks else ""

    # Heuristic fallback: try to start from first code-ish token
    code_start = re.search(r"(^|\n)\s*(def |class |import |from |@|if __name__ == ['\"]__main__['\"]:)", response_text)
    if code_start:
        candidate = response_text[code_start.start():].strip()
        # Truncate trailing non-code sections if they start with typical prose markers
        candidate = re.split(r"\n\s*(Explanation|Notes?|Output|Result|Example|That's|The code|The function|Yes, the function|Sample Input)\s*:|\n\s*#\s*End", candidate)[0]
        return candidate.strip()

    # No detectable code found; return empty to avoid dumping narrative into JSON
    return ""


def generate_code_with_retry(tokenizer, model, prompt, device, max_retries=3):
    """Generate code with retry logic for better reliability."""
    
    # First attempt with standard instruction
    instruction = (
        "You are to output ONLY Python code that solves the user's request. "
        "Respond with a single fenced block using ```python ... ```. "
        "Do not include any explanations, narration, or thinking outside the fence.\n\n"
    )
    
    for attempt in range(max_retries):
        try:
            full_prompt = instruction + prompt
            inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
            raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the output
            if raw_response.startswith(full_prompt):
                raw_response = raw_response[len(full_prompt):].strip()
            
            # Extract only the code
            code_only = extract_code_from_response(raw_response)
            
            # If we got code, return it
            if code_only.strip():
                print(f"  Successfully extracted code on attempt {attempt + 1}")
                return code_only
            
            # If no code detected and this isn't the last attempt, try with stricter instruction
            if attempt < max_retries - 1:
                if attempt == 0:
                    instruction = (
                        "Return ONLY a single fenced Python block with the final code. "
                        "Absolutely no prose. If you cannot, return an empty code block.\n\n"
                    )
                else:
                    instruction = (
                        "ONLY output Python code. No explanations, no examples, no testing. "
                        "Just the code in a fenced block: ```python\n[code here]\n```\n\n"
                    )
                print(f"  Retry attempt {attempt + 1} with stricter instruction...")
            
        except Exception as e:
            print(f"  Generation attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise e
    
    # If all attempts failed to produce code, return empty string
    print("  All attempts failed to produce clean code")
    return ""


def setup_logger():
    logging.basicConfig(
        filename="data/llm_runtime.log",
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s\t%(levelname)s\t%(message)s",
    )

def log_llm_duration(language, start_ts, end_ts, seconds, success=True, error_message=None):
    start_iso = datetime.fromtimestamp(start_ts).isoformat(timespec="seconds")
    end_iso = datetime.fromtimestamp(end_ts).isoformat(timespec="seconds")
    minutes = seconds / 60.0
    if success:
        logging.info(f"lang={language}\tstart={start_iso}\tend={end_iso}\tduration_min={minutes:.3f}")
    else:
        logging.error(f"lang={language}\tstart={start_iso}\tend={end_iso}\tduration_min={minutes:.3f}\terror={error_message}")

def main():
    setup_logger()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)

    # Load prompts
    with open("data/translated_prompts.json", "r", encoding="utf-8") as f:
        translations = json.load(f)

    llm_outputs = {}
    for lang, prompt in translations.items():
        if not prompt:
            llm_outputs[lang] = None
            continue

        print(f"Generating code for language: {lang}")
        start_perf = time.perf_counter()
        start_wall = time.time()
        try:
            # Use the new retry function for better reliability
            code_only = generate_code_with_retry(tokenizer, model, prompt, device)
            end_wall = time.time()
            duration = time.perf_counter() - start_perf
            log_llm_duration(lang, start_wall, end_wall, duration, success=True)
            llm_outputs[lang] = code_only
            print(f"Generated code: {code_only[:100]}...")
        except Exception as e:
            end_wall = time.time()
            duration = time.perf_counter() - start_perf
            log_llm_duration(lang, start_wall, end_wall, duration, success=False, error_message=str(e))
            print(f"Error generating code for {lang}: {e}")
            llm_outputs[lang] = None
        
        time.sleep(1)  # Be nice to your GPU, avoid overheating

    # Save results
    with open("data/llm_output.json", "w", encoding="utf-8") as f:
        json.dump(llm_outputs, f, ensure_ascii=False, indent=4)

    print("LLM processing completed. Results saved to data/llm_output.json")

if __name__ == "__main__":
    main()
