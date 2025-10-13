#!/usr/bin/env bash
# Run LLM-as-Judge with Batch API

export OPENAI_API_KEY=""              
export JUDGE_MODEL="gpt-4.1"           
export INPUT_CSV="/path/to/test_5p_gen.csv"
export LOGDIR="/path/to/benchmark_new/llm_eval/logs"
export OUTDIR="/path/to/benchmark_new/llm_eval/output"
export N_ROWS=300                       # total rows to evaluate
export CHUNK_SIZE=50                    # rows per batch job
export MAX_OUTPUT_TOKENS=450            # model output limit
python gpt.py