import asyncio
import inspect
import json
import os
import re
from typing import Dict, List, Optional

from googletrans import Translator


TARGET_LANG_CODES: List[str] = [
    "en",
    "zh-CN",
    "hi",
    "es",
    "ar",
    "fr",
    "bn",
    "pt",
    "ru",
    "id",
    "ur",
    "de",
    "ja",
    "mr",
    "vi",
    "te",
    "ha",
    "tr",
]

LANG_CODE_TO_NAME: Dict[str, str] = {
    "en": "English",
    "zh-CN": "Mandarin Chinese",
    "hi": "Hindi",
    "es": "Spanish",
    "ar": "Standard Arabic",
    "fr": "French",
    "bn": "Bengali",
    "pt": "Portuguese",
    "ru": "Russian",
    "id": "Indonesian",
    "ur": "Urdu",
    "de": "Standard German",
    "ja": "Japanese",
    "mr": "Marathi",
    "vi": "Vietnamese",
    "te": "Telugu",
    "ha": "Hausa",
    "tr": "Turkish",
}


SENTENCE_PUNCT = re.compile(r'([.!?])(?!\s|$)')


def normalize_text(text: str) -> str:
    text = SENTENCE_PUNCT.sub(r'\1 ', text)
    return re.sub(r'\s{2,}', ' ', text).strip()


async def translate_prompt(prompt_text: str, language_codes: List[str]) -> Dict[str, Optional[str]]:
    translator = Translator(service_urls=["translate.googleapis.com"])
    translations: Dict[str, Optional[str]] = {}

    for code in language_codes:
        try:
            if code == "en":
                translations[code] = prompt_text
                lang_name = LANG_CODE_TO_NAME.get(code, code)
                print(f"{code} ({lang_name}): {translations[code]}")
                continue

            result = translator.translate(prompt_text, dest=code)
            if inspect.isawaitable(result):
                result = await result
            translations[code] = result.text
            lang_name = LANG_CODE_TO_NAME.get(code, code)
            print(f"{code} ({lang_name}): {translations[code]}")
        except Exception as exc:
            print(f"Error translating to {code}: {exc}")
            translations[code] = None

    return translations


def write_translations_to_json(translations: Dict[str, Optional[str]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(translations, f, ensure_ascii=False, indent=4)


def main() -> None:
    prompt_text = input("Enter the prompt to translate: ").strip()
    prompt_text = normalize_text(prompt_text)

    translations = asyncio.run(translate_prompt(prompt_text, TARGET_LANG_CODES))
    out_file = os.path.join("data", "translated_prompts.json")
    write_translations_to_json(translations, out_file)

    print(f"Wrote translations to {out_file}")
    


if __name__ == "__main__":
    main()
