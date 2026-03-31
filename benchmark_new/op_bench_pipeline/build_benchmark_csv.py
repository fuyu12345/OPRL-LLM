#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass
class BuildConfig:
    predictions_csv: str
    responses_csv: str
    prediction_column: str
    output_dir: str
    output_name: str
    source_column: str = "source"


def parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser(description="Convert judged rows into OP-Bench scoring CSVs.")
    parser.add_argument("--predictions_csv", required=True, help="Prediction CSV produced by OP-Bench judge.")
    parser.add_argument("--responses_csv", required=True, help="Original 60-response CSV containing question source.")
    parser.add_argument(
        "--prediction_column",
        required=True,
        help="Column in predictions_csv containing the 1-5 judge scores.",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for scored CSV outputs.")
    parser.add_argument("--output_name", required=True, help="Base output filename for the full scored CSV.")
    parser.add_argument("--source_column", default="source", help="Source column name in responses_csv.")
    args = parser.parse_args()
    return BuildConfig(**vars(args))


def build_scored_frame(config: BuildConfig) -> pd.DataFrame:
    pred_path = Path(config.predictions_csv).expanduser().resolve()
    resp_path = Path(config.responses_csv).expanduser().resolve()
    if not pred_path.exists():
        raise FileNotFoundError(f"predictions_csv not found: {pred_path}")
    if not resp_path.exists():
        raise FileNotFoundError(f"responses_csv not found: {resp_path}")

    pred = pd.read_csv(pred_path)
    if config.prediction_column not in pred.columns:
        raise ValueError(f"Prediction column not found: {config.prediction_column}")

    responses = pd.read_csv(resp_path)
    required = {"question_id", config.source_column}
    missing = sorted(required - set(responses.columns))
    if missing:
        raise ValueError(f"responses_csv is missing columns: {missing}")

    responses = responses[["question_id", config.source_column]].drop_duplicates()
    pred["representation_rating"] = pd.to_numeric(pred[config.prediction_column], errors="coerce")
    if pred["representation_rating"].isna().all():
        raise ValueError("All representation_rating values are NaN after conversion.")

    scored = pred.merge(responses, on="question_id", how="left")
    return scored


def save_outputs(scored: pd.DataFrame, config: BuildConfig) -> tuple[Path, Path, Path, Path]:
    out_dir = Path(config.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    full_path = out_dir / config.output_name
    stem = full_path.stem
    suffix = full_path.suffix or ".csv"
    modelslant_path = out_dir / f"{stem}_modelslant{suffix}"
    prism_path = out_dir / f"{stem}_prism{suffix}"
    meta_path = out_dir / f"{stem}.meta.json"

    scored.to_csv(full_path, index=False)
    if config.source_column in scored.columns:
        scored[scored[config.source_column] == "modelslant"].to_csv(modelslant_path, index=False)
        scored[scored[config.source_column] == "prism"].to_csv(prism_path, index=False)
    else:
        pd.DataFrame(columns=scored.columns).to_csv(modelslant_path, index=False)
        pd.DataFrame(columns=scored.columns).to_csv(prism_path, index=False)

    payload = {
        "config": asdict(config),
        "rows": int(len(scored)),
        "questions": int(scored["question_id"].nunique()),
        "modelslant_rows": int((scored[config.source_column] == "modelslant").sum())
        if config.source_column in scored.columns
        else 0,
        "prism_rows": int((scored[config.source_column] == "prism").sum())
        if config.source_column in scored.columns
        else 0,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return full_path, modelslant_path, prism_path, meta_path


def main() -> None:
    config = parse_args()
    scored = build_scored_frame(config)
    full_path, modelslant_path, prism_path, meta_path = save_outputs(scored, config)
    print(f"[save] full={full_path}")
    print(f"[save] modelslant={modelslant_path}")
    print(f"[save] prism={prism_path}")
    print(f"[save] meta={meta_path}")
    print(f"[summary] rows={len(scored)} questions={scored['question_id'].nunique()}")


if __name__ == "__main__":
    main()
