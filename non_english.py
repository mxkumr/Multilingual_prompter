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
    
    # Create labels with both count and percentage
    def make_autopct(values):
        def autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return f'{pct:.1f}%\n({val})'
        return autopct
    
    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=labels, autopct=make_autopct(values), startangle=90)
    plt.axis("equal")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_category_bars(by_category: Dict[str, Dict[str, int]], out_path: str, title: str) -> None:
    categories = list(by_category.keys())
    # Collect all bucket names across categories
    all_buckets = sorted({b for v in by_category.values() for b in v.keys()}) or ["English/ASCII", "Non-English"]

    import numpy as np
    x = np.arange(len(categories))
    width = 0.8 / max(1, len(all_buckets))

    plt.figure(figsize=(14, 8))
    bars = []
    for i, bucket in enumerate(all_buckets):
        vals = [by_category.get(cat, {}).get(bucket, 0) for cat in categories]
        bar = plt.bar(x + i * width - (len(all_buckets)-1) * width / 2, vals, width=width, label=bucket)
        bars.append(bar)
        
        # Add count labels on top of bars
        for j, (bar_rect, val) in enumerate(zip(bar, vals)):
            if val > 0:  # Only show label if there's a value
                plt.text(bar_rect.get_x() + bar_rect.get_width()/2., bar_rect.get_height() + 0.1,
                        f'{val}', ha='center', va='bottom', fontsize=8)

    plt.xticks(x, categories, rotation=30, ha="right")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_detailed_summary_table(summary: Dict[str, Any]) -> str:
    """Create a detailed text summary table of the language analysis."""
    lines = []
    lines.append("=" * 80)
    lines.append("DETAILED LANGUAGE ANALYSIS SUMMARY")
    lines.append("=" * 80)
    
    for lang_key, data in summary.items():
        lines.append(f"\n[LANGUAGE] {lang_key.upper()}")
        lines.append("-" * 40)
        
        # Overall summary
        overall = data.get("overall", {})
        total_items = sum(overall.values())
        lines.append(f"Total Items: {total_items}")
        
        for script, count in sorted(overall.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_items * 100) if total_items > 0 else 0
            lines.append(f"  {script}: {count} items ({percentage:.1f}%)")
        
        # Category breakdown
        lines.append(f"\n[CATEGORY BREAKDOWN]:")
        by_category = data.get("by_category", {})
        for category, scripts in by_category.items():
            if scripts:  # Only show categories with data
                lines.append(f"  {category.title()}:")
                for script, count in sorted(scripts.items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"    {script}: {count}")
    
    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def run_visualization(input_path: str, charts_dir: str, summary_out: Optional[str] = None) -> None:
    os.makedirs(charts_dir, exist_ok=True)

    if not os.path.exists(input_path):
        print(f"Input not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        parsed = json.load(f)

    results = parsed.get("results", {}) if isinstance(parsed, dict) else {}
    summary: Dict[str, Any] = {}

    print(f"Processing {len(results)} languages...")
    
    for lang_key, item in results.items():
        if not item or not item.get("success"):
            print(f"Skipping {lang_key} (no successful parsing)")
            continue
            
        print(f"Processing {lang_key}...")
        elements = item.get("elements", {}) or {}
        counts = aggregate_counts(elements)
        summary[lang_key] = counts

        # Charts per language
        pie_path = os.path.join(charts_dir, f"{lang_key}_overall_pie.png")
        plot_overall_pie(counts["overall"], pie_path, f"Overall Script Distribution: {lang_key}")

        bars_path = os.path.join(charts_dir, f"{lang_key}_by_category.png")
        plot_category_bars(counts["by_category"], bars_path, f"Script Distribution by Category: {lang_key}")

    # Create overall comparison chart
    if summary:
        create_overall_comparison_chart(summary, charts_dir)

    # Create detailed summary
    detailed_summary = create_detailed_summary_table(summary)
    print("\n" + detailed_summary)

    # Save summary JSON with enhanced data
    if summary_out:
        enhanced_summary = {
            "summary": summary,
            "detailed_analysis": detailed_summary,
            "total_languages": len(summary),
            "generated_at": __import__('datetime').datetime.now().isoformat()
        }
        with open(summary_out, "w", encoding="utf-8") as f:
            json.dump(enhanced_summary, f, ensure_ascii=False, indent=2)

    print(f"\n[SUCCESS] Saved charts to: {charts_dir}")
    if summary_out:
        print(f"[SUCCESS] Saved enhanced summary to: {summary_out}")


def create_overall_comparison_chart(summary: Dict[str, Any], charts_dir: str) -> None:
    """Create an overall comparison chart across all languages."""
    import numpy as np
    
    languages = list(summary.keys())
    all_scripts = set()
    
    # Collect all unique scripts
    for data in summary.values():
        overall = data.get("overall", {})
        all_scripts.update(overall.keys())
    
    all_scripts = sorted(list(all_scripts))
    
    # Create data matrix
    data_matrix = []
    for lang in languages:
        overall = summary[lang].get("overall", {})
        row = [overall.get(script, 0) for script in all_scripts]
        data_matrix.append(row)
    
    data_matrix = np.array(data_matrix)
    
    # Create stacked bar chart
    plt.figure(figsize=(16, 10))
    x = np.arange(len(languages))
    width = 0.8
    
    bottom = np.zeros(len(languages))
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_scripts)))
    
    for i, script in enumerate(all_scripts):
        values = data_matrix[:, i]
        if np.any(values > 0):  # Only plot if there are values
            plt.bar(x, values, width, bottom=bottom, label=script, color=colors[i])
            
            # Add count labels on bars
            for j, (val, bot) in enumerate(zip(values, bottom)):
                if val > 0:
                    plt.text(j, bot + val/2, f'{int(val)}', ha='center', va='center', 
                            fontsize=8, fontweight='bold')
            
            bottom += values
    
    plt.xlabel('Languages')
    plt.ylabel('Count')
    plt.title('Script Distribution Across All Languages')
    plt.xticks(x, languages, rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    comparison_path = os.path.join(charts_dir, "overall_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[SUCCESS] Created overall comparison chart: {comparison_path}")


def main() -> None:
    project_root = os.path.abspath(os.path.dirname(__file__))
    ensure_mlp_on_path(project_root)

    input_path = os.path.join(project_root, "data", "llm_parsed.json")
    charts_dir = os.path.join(project_root, "data", "language_charts")
    summary_out = os.path.join(project_root, "data", "non_english_summary.json")
    run_visualization(input_path, charts_dir, summary_out)


if __name__ == "__main__":
    main()



