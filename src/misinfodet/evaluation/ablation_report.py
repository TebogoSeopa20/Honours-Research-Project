"""Aggregate ablation results into a comparison table (Markdown + LaTeX).

Reads every metrics JSON in outputs/predictions/ matching *_metrics.json and
builds the table for Section 3.5's ablation study.
"""
from __future__ import annotations

import json
from pathlib import Path


COLUMNS = ["accuracy", "macro_f1", "macro_precision", "macro_recall", "parse_failure_rate"]


def collect(pred_dir: str | Path) -> dict[str, dict]:
    results = {}
    for path in sorted(Path(pred_dir).glob("*_metrics.json")):
        results[path.stem.replace("_metrics", "")] = json.loads(path.read_text())
    return results


def to_markdown(results: dict[str, dict]) -> str:
    header = "| Configuration | " + " | ".join(COLUMNS) + " |"
    sep = "|" + "---|" * (len(COLUMNS) + 1)
    rows = [header, sep]
    for name, m in results.items():
        rows.append(
            f"| {name} | " + " | ".join(str(m.get(c, "-")) for c in COLUMNS) + " |"
        )
    return "\n".join(rows)


def to_latex(results: dict[str, dict]) -> str:
    lines = [
        "\\begin{table}[ht]\\centering",
        "\\caption{Ablation study results.}",
        "\\begin{tabular}{l" + "c" * len(COLUMNS) + "}",
        "\\toprule",
        "Configuration & " + " & ".join(c.replace("_", " ") for c in COLUMNS) + " \\\\",
        "\\midrule",
    ]
    for name, m in results.items():
        lines.append(
            name.replace("_", " ")
            + " & "
            + " & ".join(str(m.get(c, "-")) for c in COLUMNS)
            + " \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def main(pred_dir: str = "outputs/predictions", out_dir: str = "outputs") -> None:
    results = collect(pred_dir)
    if not results:
        print("No *_metrics.json files found in", pred_dir)
        return
    Path(out_dir, "ablation_table.md").write_text(to_markdown(results))
    Path(out_dir, "ablation_table.tex").write_text(to_latex(results))
    print(to_markdown(results))


if __name__ == "__main__":
    main()
