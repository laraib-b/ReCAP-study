"""Build markdown tables from a study JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("study_json", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    payload = json.loads(args.study_json.read_text(encoding="utf-8"))
    lines = [
        "# Study results",
        "",
        f"- Profile: `{payload.get('profile')}`",
        f"- Dataset: `{payload.get('dataset')}`",
        f"- Seed: `{payload.get('seed')}`",
        f"- Warning: {payload.get('warning')}",
        "",
        "## Calibrations (train-only)",
        "",
        "| Variant | Target alarms/yr | Chosen h/σ | Train alarms/yr |",
        "|---|---:|---:|---:|",
    ]
    for name, cal in payload.get("calibrations", {}).items():
        lines.append(
            f"| {name} | {cal['target_alarms_per_year']} | {cal['threshold_scale']:.4f} | {cal['train_alarms_per_year']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Causal portfolio + detector outcomes",
            "",
            "| Variant | CR | SR | MDD | Test alarms/yr | Adaptations/yr | Library |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, result in payload.get("results", {}).items():
        m = result["metrics"]
        d = result["detector"]
        lines.append(
            "| {name} | {cr:.4f} | {sr:.4f} | {mdd:.4f} | {alarms:.4f} | {adapt:.4f} | {lib} |".format(
                name=name,
                cr=m["CR"],
                sr=m["SR"],
                mdd=m["MDD"],
                alarms=d["alarms_per_year"],
                adapt=result["adaptations_per_year"],
                lib=result["library_size"],
            )
        )

    synth = payload.get("synthetic_diagnostic")
    if synth:
        lines.extend(["", "## Synthetic-shift diagnostic", "", "| Label | Magnitude | Detected | Delay | Missed | FA before shift |", "|---|---:|---|---:|---|---:|"])
        for row in synth.get("results", []):
            lines.append(
                f"| {row['label']} | {row['injected_magnitude']} | {row['detected']} | {row['detection_delay']} | {row['missed']} | {row['false_alarms_before_shift']} |"
            )

    text = "\n".join(lines) + "\n"
    out = args.out or args.study_json.with_suffix(".md")
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
