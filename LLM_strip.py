import json
import requests
import time
import re
import logging
from datetime import datetime

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"  # Change to your model name if needed


def extract_code_from_response(response_text):
    """Extract only the code from the LLM response, removing thinking/narrative.

    Strategy:
    1) Strip <think>...</think> blocks if present.
    2) Prefer fenced code blocks (any language). If multiple, return the longest.
    3) If no fences, try to detect code by finding the first code token and return from there.
    4) If nothing code-like is detected, return an empty string to avoid prose leakage.
    """
    if not response_text:
        return ""

    # Remove thinking process and common preambles
    response_text = re.sub(r"<think>[\s\S]*?</think>", "", response_text, flags=re.DOTALL)
    response_text = re.sub(r"(?i)^(thoughts?|reasoning|analysis)\s*:.*?$", "", response_text, flags=re.MULTILINE)

    # Prefer fenced code blocks of any language
    fenced_blocks = re.findall(r"```[a-zA-Z0-9_+-]*\s*([\s\S]*?)\s*```", response_text)
    if fenced_blocks:
        # Choose the longest block assuming it's the main solution
        longest = max((b.strip() for b in fenced_blocks), key=len, default="")
        return longest

    # Heuristic fallback: try to start from first code-ish token
    code_start = re.search(r"(^|\n)\s*(def |class |import |from |@|if __name__ == ['\"]__main__['\"]:)", response_text)
    if code_start:
        candidate = response_text[code_start.start():].strip()
        # Truncate trailing non-code sections if they start with typical prose markers
        candidate = re.split(r"\n\s*(Explanation|Notes?|Output|Result|Example)\s*:|\n\s*#\s*End", candidate)[0]
        return candidate.strip()

    # No detectable code found; return empty to avoid dumping narrative into JSON
    return ""


def query_model(prompt, model=MODEL):
    # Prepend strict instruction to enforce code-only output
    instruction = (
        "You are to output ONLY Python code that solves the user's request. "
        "Respond with a single fenced block using ```python ... ```. "
        "Do not include any explanations, narration, or thinking outside the fence.\n\n"
    )
    payload = {
        "model": model,
        "prompt": instruction + prompt,
        "stream": False,
        "options": {"num_gpu": 10}
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=(120, 36000))
        if not response.ok:
            raise RuntimeError(f"{response.status_code} {response.reason}: {response.text}")
        raw_response = response.json().get("response", "")
    except requests.RequestException as e:
        raise RuntimeError(f"Request failed: {e}")
    
    # Extract only the code
    code_only = extract_code_from_response(raw_response)
    # If still no code was detected, make one retry with an even stricter instruction
    if not code_only.strip():
        retry_instruction = (
            "Return ONLY a single fenced Python block with the final code. "
            "Absolutely no prose. If you cannot, return an empty code block.\n\n"
        )
        payload["prompt"] = retry_instruction + prompt
        response = requests.post(OLLAMA_URL, json=payload, timeout=(120, 36000))
        if response.ok:
            raw_response = response.json().get("response", "")
            code_only = extract_code_from_response(raw_response)
    return code_only


def setup_logger():
    """Configure logging to file for LLM runtimes."""
    logging.basicConfig(
        filename="data/llm_runtime.log",
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s\t%(levelname)s\t%(message)s",
    )


def log_llm_duration(language, start_ts, end_ts, seconds, success=True, error_message=None):
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


def main():
    setup_logger()
    # Load translated prompts
    with open("data/translated_prompts.json", "r", encoding="utf-8") as f:
        translations = json.load(f)

    llm_outputs = {}
    for lang, prompt in translations.items():
        if not prompt:
            llm_outputs[lang] = None
            continue
        print(f"Querying model for language: {lang}")
        start_perf = time.perf_counter()
        start_wall = time.time()
        try:
            llm_response = query_model(prompt)
            end_wall = time.time()
            duration = time.perf_counter() - start_perf
            log_llm_duration(lang, start_wall, end_wall, duration, success=True)
            llm_outputs[lang] = llm_response
            print(f"Generated code for {lang}: {llm_response[:100]}...")
        except Exception as e:
            end_wall = time.time()
            duration = time.perf_counter() - start_perf
            log_llm_duration(lang, start_wall, end_wall, duration, success=False, error_message=str(e))
            print(f"Error querying model for {lang}: {e}")
            llm_outputs[lang] = None
        time.sleep(1)  # Be nice to your LLM, avoid rate limits

    # Save LLM outputs
    with open("data/llm_output.json", "w", encoding="utf-8") as f:
        json.dump(llm_outputs, f, ensure_ascii=False, indent=4)
    
    print("LLM processing completed. Results saved to data/llm_output.json")


if __name__ == "__main__":
    main()
