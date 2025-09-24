import os
import re
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_DIR = "/your_path/verl/output_model/hf_models/hf_grpo-Llama-3.2-3B-Instruct-im-rewardscaledown-unique-12k"

# Directories and file patterns for loop
REFERENCE_DIR = "/your_path/verl/benchmark/dataset"
OUTPUT_DIR = "generated_outputs"
LABELS = [ "5p","6p","7p","8p","9p","10p","gt10p" ]

BATCH_SIZE = 1
MAX_NEW = 1200
BASE_TEMP = 0.7
BASE_TOPP = 0.9
MAX_ATTEMPTS = 4

def wrap_prompt(original: str) -> str:
    return f"Provide a structured comprehensive analysis and your opinions on this topic: {original}\n\n"

STRICT_FORMAT_RE = re.compile(
    r"(?is)<core\s+perspectives>\s*(?P<core>.+?)\s*</core\s+perspectives>\s*<summary>\s*(?P<sum>.+?)\s*</summary>"
)
def extract_blocks(text: str):
    if not isinstance(text, str):
        return None, None
    m = STRICT_FORMAT_RE.search(text)
    if not m:
        return None, None
    core = m.group("core").strip()
    summ = m.group("sum").strip()
    if core and summ:
        return core, summ
    return None, None


tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto", trust_remote_code=True)
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    model.config.pad_token_id = tokenizer.eos_token_id

def make_model_input(raw: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": raw}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return raw

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    batch_size=BATCH_SIZE,
)

# Main Loop over datasets 
os.makedirs(OUTPUT_DIR, exist_ok=True)

for label in LABELS:
    TEST_CSV = os.path.join(REFERENCE_DIR, f"test_{label}.csv")
    OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"qwen2.5-1.5b-im-{label}.csv")

    df_test = pd.read_csv(TEST_CSV, usecols=["prompt"])
    prompts = df_test["prompt"].astype(str).tolist()
    N = len(prompts)
    print(f"\n=== Processing {label} ===")
    print(f"[INFO] Loaded {N} prompts from {TEST_CSV}")

    if os.path.exists(OUTPUT_CSV):
        df_out = pd.read_csv(OUTPUT_CSV)
        for col in ["prompt", "answer_p", "answer_s", "ok"]:
            if col not in df_out.columns:
                df_out[col] = "" if col != "ok" else 0
        out_map = {row["prompt"]: row for _, row in df_out.iterrows()}
        answer_p = [str(out_map.get(p, {}).get("answer_p", "")) for p in prompts]
        answer_s = [str(out_map.get(p, {}).get("answer_s", "")) for p in prompts]
        ok_flags = [int(out_map.get(p, {}).get("ok", 0)) for p in prompts]
        print(f"[RESUME] Loaded existing results: {sum(ok_flags)} done, {N - sum(ok_flags)} remaining.")
    else:
        answer_p = ["" for _ in range(N)]
        answer_s = ["" for _ in range(N)]
        ok_flags = [0 for _ in range(N)]
        print("[RESUME] No previous output found; starting fresh.")

    def save_progress():
        df_save = pd.DataFrame({
            "prompt": prompts,
            "answer_p": answer_p,
            "answer_s": answer_s,
            "ok": ok_flags,
        })
        df_save.to_csv(OUTPUT_CSV, index=False)

    remaining = [i for i in range(N) if ok_flags[i] == 0]

    for attempt in range(MAX_ATTEMPTS):
        if not remaining:
            break
        temperature = BASE_TEMP
        top_p = BASE_TOPP
        print(f"\n[ATTEMPT {attempt+1}/{MAX_ATTEMPTS}] Remaining: {len(remaining)} | temp={temperature:.2f}, top_p={top_p:.2f}")

        todo = remaining.copy()
        newly_success_global = set()

        for start in range(0, len(todo), BATCH_SIZE):
            batch_idx = todo[start:start+BATCH_SIZE]
            if not batch_idx:
                continue
            batch_prompts = [prompts[i] for i in batch_idx]
            wrapped_inputs = [make_model_input(wrap_prompt(p)) for p in batch_prompts]
            if not wrapped_inputs:
                continue

            batch_outputs = generator(
                wrapped_inputs,
                max_new_tokens=MAX_NEW,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                return_full_text=False,
            )

            newly_success = []
            for pos_in_batch, (prompt_i, out_list) in enumerate(zip(batch_prompts, batch_outputs)):
                idx = batch_idx[pos_in_batch]
                text = out_list[0]["generated_text"] if out_list else ""
                print(f"\n=== Prompt idx {idx} ===")
                print(f"Prompt: {prompt_i}")
                print(f"--- Model output (first 500 chars) ---\n{text}\n")

                core, summ = extract_blocks(text)
                if core is not None and summ is not None:
                    answer_p[idx] = core
                    answer_s[idx] = summ
                    ok_flags[idx] = 1
                    newly_success.append(idx)
                    print("--- Extracted <core perspectives> and <summary>")
                else:
                    print("[WARN] Failed to extract valid blocks; will retry.")

            if newly_success:
                newly_success_global.update(newly_success)
            save_progress()

        if newly_success_global:
            remaining = [i for i in remaining if i not in newly_success_global]
        print(f"[ATTEMPT {attempt+1}] Success so far: {N - len(remaining)} / {N}")

    if remaining:
        print(f"\n[FINAL] Could not extract {len(remaining)} samples after {MAX_ATTEMPTS} attempts for {label}.")
    else:
        print(f"\n[FINAL] Successfully extracted all samples for {label}.")

    save_progress()
    print(f"Saved progress to {OUTPUT_CSV}")
    print(f"OK rows: {sum(ok_flags)} / {N}")
