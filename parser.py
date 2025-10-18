import os
import json
import tempfile
import shutil
import sys
from typing import Dict, Any


def parse_code_files_with_multilang_parser(code_by_lang: Dict[str, str]) -> Dict[str, Any]:
    """Write code strings to temp files, run Multi_language_parser on each, return results.

    Note: Multi_language_parser's language build uses relative paths, so we chdir into
    the Multi_language_parser directory during parsing, then restore the cwd.
    """
    project_root = os.path.abspath(os.path.dirname(__file__))
    mlp_dir = os.path.join(project_root, "Multi_language_parser")

    temp_dir = tempfile.mkdtemp(prefix="llm_code_")
    results: Dict[str, Any] = {}

    original_cwd = os.getcwd()
    try:
        # Prepare temp files (assume generated code is Python)
        lang_to_file = {}
        for lang, code in code_by_lang.items():
            if not code:
                continue
            file_path = os.path.join(temp_dir, f"{lang}.py")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
            lang_to_file[lang] = file_path

        if not lang_to_file:
            return {"success": False, "error": "No code snippets to parse"}

        # Ensure Python can import modules from Multi_language_parser and
        # switch CWD so relative paths in language_build work.
        if mlp_dir not in sys.path:
            sys.path.insert(0, mlp_dir)
        os.chdir(mlp_dir)

        # Import after sys.path update and chdir so language_build resolves correctly
        from File_parser import RepoElementParser  # type: ignore

        parser = RepoElementParser()
        for lang, file_path in lang_to_file.items():
            try:
                result = parser.parse_file(file_path)
                results[lang] = result
            except Exception as e:
                results[lang] = {"success": False, "file_path": file_path, "error": str(e)}

        return {"success": True, "results": results}

    finally:
        # Restore working directory and clean up temp dir
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    project_root = os.path.abspath(os.path.dirname(__file__))
    input_path = os.path.join(project_root, "data", "llm_output.json")
    output_path = os.path.join(project_root, "data", "llm_parsed.json")

    if not os.path.exists(input_path):
        print(f"Input not found: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        llm_outputs = json.load(f)

    if not isinstance(llm_outputs, dict):
        print("Invalid input format: expected a JSON object mapping language -> code string")
        sys.exit(1)

    result = parse_code_files_with_multilang_parser(llm_outputs)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if result.get("success"):
        print(f"Parsing complete. Results written to {output_path}")
    else:
        print(f"Parsing failed. Details written to {output_path}")


if __name__ == "__main__":
    main()


