#!/usr/bin/env python3

"""Build a simple LaTeX table from the toy distillation result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RUN_ORDER = [
    ("01_true_ar", "True AR sampling"),
    ("02_true_ode", "True ODE sampling"),
    ("03_ce_only", "CE only"),
    ("04_kl_teacher_forced", "KL to teacher-forced target"),
    ("05_kl_exact", "KL to exact denoiser"),
    ("06_ce_plus_kl_exact", "CE + exact KL"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def load_latest_json(run_dir: Path) -> dict | None:
    candidates = sorted(run_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    with candidates[0].open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_metric(metrics: dict | None, key: str) -> str:
    if metrics is None or key not in metrics or metrics[key] is None:
        return "--"
    return f"{float(metrics[key]):.4f}"


def build_table(rows: list[tuple[str, dict | None]]) -> str:
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\begin{document}",
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"Method & Hard CE & TF KL & Exact KL & Token Acc & Exact Vel MSE & Seq CE & Seq KL & Seq TV \\",
        r"\midrule",
    ]

    for label, payload in rows:
        summary = None if payload is None else payload.get("summary_metrics")
        lines.append(
            " & ".join(
                [
                    label,
                    format_metric(summary, "hard_ce"),
                    format_metric(summary, "teacher_forced_kl"),
                    format_metric(summary, "exact_kl"),
                    format_metric(summary, "token_acc"),
                    format_metric(summary, "exact_velocity_mse"),
                    format_metric(summary, "sequence_cross_entropy"),
                    format_metric(summary, "sequence_kl_emp_to_teacher"),
                    format_metric(summary, "sequence_tv_to_teacher"),
                ]
            )
            + r" \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\caption{Toy distillation comparison across oracle baselines and learned student objectives. Lower is better for every metric except token accuracy.}",
            r"\end{table}",
            r"\end{document}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_path = Path(args.output)

    rows: list[tuple[str, dict | None]] = []
    for run_dir_name, fallback_label in RUN_ORDER:
        payload = load_latest_json(results_root / run_dir_name)
        label = fallback_label
        if payload is not None and payload.get("display_name"):
            label = str(payload["display_name"])
            if payload.get("kind") == "student" and payload.get("mode") == "ce_kl_exact":
                lambda_kl = payload.get("config", {}).get("lambda_kl", 1.0)
                label = f"{label} ($\\lambda={lambda_kl:g}$)"
        rows.append((label, payload))

    table_tex = build_table(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table_tex, encoding="utf-8")
    print(f"Wrote LaTeX table to {output_path}")


if __name__ == "__main__":
    main()
