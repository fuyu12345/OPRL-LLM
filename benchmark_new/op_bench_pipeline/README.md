# OP-Bench Evaluation Pipeline

This folder provides an anonymized, minimal pipeline for evaluating model outputs with OP-Bench.

The pipeline has four stages:

1. Generate one response for each of the 60 OP-Bench questions.
2. Expand those 60 responses to the official `user x question` evaluation rows.
3. Run an LLM judge to predict row-level representation ratings.
4. Convert the judged rows into OP-Bench scoring inputs and run the OP-Bench scoring script.

## Data Assumptions

The generated response CSV must contain at least:

- `question_id`
- `question`
- `response`

The official OP-Bench data source must contain at least:

- `user`
- `question_id`
- `question`
- `freeresponse`
- `selection_text`
- `selection_position`
- `Age`
- `Sex`
- `Ethnicity simplified`
- `U.s. political affiliation`
- `cluster_kmeans`

The response CSV may also include a `source` column with values such as `modelslant` or `prism`. If present, it will be used later to create split-specific scored files.

## Step 1: Expand 60 Responses to Judge Rows

Use `prepare_judge_rows.py` to merge the 60 generated responses with the official OP-Bench user-question rows.

Example:

```bash
python benchmark_new/op_bench_pipeline/prepare_judge_rows.py \
  --responses_csv path/to/responses.csv \
  --official_data_path path/to/official_overtonbench_rows.csv \
  --output_dir path/to/judge_inputs \
  --output_name model_overton60_full_rows.csv \
  --model_name model_name
```

If `--official_data_path` is omitted, the script will try to load the official Hugging Face dataset (`elinorpd/overtonbench`, split `full`).

## Step 2: Run the OP-Bench Judge

Run the OP-Bench judge pipeline on the expanded judge rows. This step is performed in the OP-Bench codebase itself.

For the original OP-Bench prompt:

```bash
python src/prompting_pipeline/prediction.py \
  --client <judge_client> \
  --model <judge_model> \
  --prompt fs+fr \
  --fewshot_source full \
  --data path/to/judge_inputs/model_overton60_full_rows.csv
```

For the modified multi-perspective judge prompt:

```bash
python src/prompting_pipeline/prediction.py \
  --client <judge_client> \
  --model <judge_model> \
  --prompt multi-perspective \
  --data path/to/judge_inputs/model_overton60_full_rows.csv
```

The judge output must contain a numeric prediction column such as:

- `openrouter_fs+fr_avg`
- `openrouter_multi-perspective_avg`

## Step 3: Convert Judge Output to OP-Bench Scoring Inputs

Use `build_benchmark_csv.py` to map the predicted judge scores into OP-Bench's expected `representation_rating` column and to create split-specific scoring files.

Example:

```bash
python benchmark_new/op_bench_pipeline/build_benchmark_csv.py \
  --predictions_csv path/to/predictions.csv \
  --responses_csv path/to/responses.csv \
  --prediction_column openrouter_fs+fr_avg \
  --output_dir path/to/benchmark_inputs \
  --output_name model_scored.csv
```

This writes:

- `model_scored.csv`
- `model_scored_modelslant.csv`
- `model_scored_prism.csv`

## Step 4: Compute OP-Bench Scores

Run the OP-Bench scoring script on the generated scoring inputs:

```bash
python src/benchmark_overton_pipeline.py \
  --data path/to/benchmark_inputs/model_scored.csv \
  --cluster_col cluster_kmeans \
  --weighted \
  --outdir path/to/benchmark_outputs/model_full
```

Repeat for `model_scored_modelslant.csv` and `model_scored_prism.csv` if separate split scores are needed.

## How OP-Bench Computes the Final Score

For each question and opinion cluster, OP-Bench averages the row-level judge scores across users in that cluster. A cluster is counted as covered if its average score is at least the threshold `tau` (default `4`).

The question-level OP score is the proportion of covered clusters for that question. The final benchmark score is the average of these question-level scores.

Since OP-Bench contains 45 Prism questions and 15 ModelSlant questions, the `Full` score is not the simple average of the two split scores. It is the weighted average over all 60 questions:

```text
Full = (45 * Prism + 15 * ModelSlant) / 60
```

## Judge Prompt Variants

### Original OP-Bench Judge

The original OP-Bench setup scores whether a single user would feel that the entire response represents their own view.

### Modified Multi-Perspective Judge

The modified prompt changes the criterion from whole-response single-user alignment to perspective inclusion. It asks whether the response contains at least one clear and meaningful perspective that matches the user's expressed view.

This change should be reported as an adjusted evaluation setting rather than as a direct replacement for the original OP-Bench definition.

The full template is provided in `PROMPT_TEMPLATES.md`.
