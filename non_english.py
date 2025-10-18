import os
import sys
import json
from collections import defaultdict
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt


def ensure_mlp_on_path(project_root: str) -> str:
    mlp_dir = os.path.join(project_root, "Multi_language_parser")
    if mlp_dir not in sys.path:
        sys.path.insert(0, mlp_dir)
    return mlp_dir


def classify_text(text: str) -> str:
    """Return script label using Multi_language_parser.language_detection.classify_string."""
    try:
        from language_detection import classify_string  # type: ignore
        return classify_string(text).get("script", "Unknown")
    except Exception:
        # Fallback: simple ASCII check
        try:
            text.encode("ascii")
            return "English/ASCII"
        except Exception:
            return "Non-English"


def aggregate_counts(elements: Dict[str, List[str]]) -> Dict[str, Any]:
    categories = [
        "identifiers",
        "variables",
        "literals",
        "comments",
        "docstrings",
        "functions",
        "classes",
    ]

    overall = defaultdict(int)
    by_category = {cat: defaultdict(int) for cat in categories}

    for cat in categories:
        values = elements.get(cat, []) or []
        for value in values:
            script = classify_text(str(value))
            # Normalize to two buckets for overview; keep script names for detail
            bucket = "English/ASCII" if script == "English/ASCII" else script or "Non-English"
            overall[bucket] += 1
            by_category[cat][bucket] += 1

    return {
        "overall": dict(overall),
        "by_category": {k: dict(v) for k, v in by_category.items()},
    }


def plot_overall_pie(counts: Dict[str, int], out_path: str, title: str) -> None:
    if not counts:
        return
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    plt.figure(figsize=(6, 6))
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.axis("equal")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_category_bars(by_category: Dict[str, Dict[str, int]], out_path: str, title: str) -> None:
    categories = list(by_category.keys())
    # Collect all bucket names across categories
    all_buckets = sorted({b for v in by_category.values() for b in v.keys()}) or ["English/ASCII", "Non-English"]

    import numpy as np
    x = np.arange(len(categories))
    width = 0.8 / max(1, len(all_buckets))

    plt.figure(figsize=(12, 6))
    for i, bucket in enumerate(all_buckets):
        vals = [by_category.get(cat, {}).get(bucket, 0) for cat in categories]
        plt.bar(x + i * width - (len(all_buckets)-1) * width / 2, vals, width=width, label=bucket)

    plt.xticks(x, categories, rotation=30, ha="right")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_visualization(input_path: str, charts_dir: str, summary_out: Optional[str] = None) -> None:
    os.makedirs(charts_dir, exist_ok=True)

    if not os.path.exists(input_path):
        print(f"Input not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        parsed = json.load(f)

    results = parsed.get("results", {}) if isinstance(parsed, dict) else {}
    summary: Dict[str, Any] = {}

    for lang_key, item in results.items():
        if not item or not item.get("success"):
            continue
        elements = item.get("elements", {}) or {}
        counts = aggregate_counts(elements)
        summary[lang_key] = counts

        # Charts per language
        pie_path = os.path.join(charts_dir, f"{lang_key}_overall_pie.png")
        plot_overall_pie(counts["overall"], pie_path, f"Overall English vs Non-English: {lang_key}")

        bars_path = os.path.join(charts_dir, f"{lang_key}_by_category.png")
        plot_category_bars(counts["by_category"], bars_path, f"By Category: {lang_key}")

    # Save summary JSON
    if summary_out:
        with open(summary_out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary}, f, ensure_ascii=False, indent=2)

    print(f"Saved charts to: {charts_dir}")
    if summary_out:
        print(f"Saved summary to: {summary_out}")


def main() -> None:
    project_root = os.path.abspath(os.path.dirname(__file__))
    ensure_mlp_on_path(project_root)

    input_path = os.path.join(project_root, "data", "llm_parsed.json")
    charts_dir = os.path.join(project_root, "data", "language_charts")
    summary_out = os.path.join(project_root, "data", "non_english_summary.json")
    run_visualization(input_path, charts_dir, summary_out)


if __name__ == "__main__":
    main()



