"""Generate MiniMax-M2.5 training data from open-perfectblend questions.

Downloads prompts from mlabonne/open-perfectblend and regenerates responses
using MiniMax-M2.5 via the DeepInfra API. Outputs JSONL suitable for
TorchSpec Eagle3 training with chat_template=minimax-m2.

Usage:
  python examples/data/generate_minimax_data.py \
    --output examples/data/minimax_m25_train.jsonl \
    --num-samples 500 \
    --max-workers 32

  # Eval set
  python examples/data/generate_minimax_data.py \
    --output examples/data/minimax_m25_eval.jsonl \
    --num-samples 64 --offset 500 --max-workers 16
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

API_URL = "http://di-slc-193.cloud.deepinfra.com:9000/v1/chat/completions"
API_KEY = os.environ.get(
    "DEEPINFRA_API_KEY", ""
)
MODEL = os.environ.get("MODEL", "")  # leave empty for direct endpoints
MAX_TOKENS = 16000
REQUEST_TIMEOUT = 600  # MiniMax-M2.5 reasoning can be slow


def load_prompts(num_samples: int, offset: int = 0, seed: int = 42) -> list[dict]:
    """Load and sample prompts from the dataset."""
    from datasets import load_dataset

    print(f"Loading dataset mlabonne/open-perfectblend...")
    ds = load_dataset("mlabonne/open-perfectblend", split="train")
    print(f"Dataset size: {len(ds)}")

    # Deterministic shuffle
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    selected = indices[offset : offset + num_samples]
    prompts = []
    for idx in selected:
        row = ds[idx]
        conversations = row["conversations"]
        # Extract user messages (format uses 'from'/'value' keys)
        user_msgs = []
        for msg in conversations:
            role = msg.get("from", msg.get("role", ""))
            content = msg.get("value", msg.get("content", ""))
            if role in ("human", "user"):
                user_msgs.append(content)

        if user_msgs:
            prompts.append({
                "idx": idx,
                "source": row.get("source", ""),
                "user_messages": user_msgs,
                # For multi-turn, keep all turns up to last user msg
                "full_conversation": conversations,
            })

    print(f"Selected {len(prompts)} prompts (offset={offset})")
    return prompts


def call_api(messages: list[dict], max_retries: int = 3) -> dict | None:
    """Call the DeepInfra API with retries."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40
    }
    if MODEL:
        payload["model"] = MODEL

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT
            )
            if resp.status_code == 429:
                wait = 2 ** attempt + random.random() * 2
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            message = choice["message"]
            return {
                "content": message.get("content", ""),
                "reasoning_content": message.get("reasoning_content", ""),
                "finish_reason": choice.get("finish_reason", ""),
            }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt + random.random())
            else:
                print(f"  API error after {max_retries} retries: {e}", file=sys.stderr)
                return None
    return None


def generate_sample(prompt: dict, sample_id: int) -> dict | None:
    """Generate a single training sample.

    Stores the raw API response as the assistant message — no reformatting.
    This is what the eagle draft model should learn to predict.
    """
    user_content = prompt["user_messages"][0]
    messages = [{"role": "user", "content": user_content}]

    result = call_api(messages)
    if result is None:
        return None

    content = result["content"] or ""
    reasoning = result["reasoning_content"] or ""

    # Concatenate reasoning + content exactly as the model produced them
    assistant_content = ""
    if reasoning:
        assistant_content += reasoning
    if content:
        assistant_content += content

    if len(assistant_content.strip()) < 10:
        return None

    conversations = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    return {
        "id": f"minimax_{sample_id:06d}",
        "conversations": conversations,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate MiniMax-M2.5 training data")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--append", action="store_true", help="Append to existing file")
    args = parser.parse_args()

    prompts = load_prompts(args.num_samples, args.offset, args.seed)

    # Skip already-generated samples if appending
    existing = 0
    if args.append and os.path.exists(args.output):
        with open(args.output) as f:
            existing = sum(1 for _ in f)
        prompts = prompts[existing:]
        print(f"Appending to {args.output} ({existing} existing, {len(prompts)} remaining)")

    if not prompts:
        print("Nothing to generate.")
        return

    succeeded = 0
    failed = 0
    t0 = time.time()
    write_lock = __import__("threading").Lock()

    print(f"Generating {len(prompts)} samples with {args.max_workers} workers...")

    mode = "a" if args.append else "w"
    with open(args.output, mode) as outfile, \
         ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(generate_sample, p, existing + i): i
            for i, p in enumerate(prompts)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                sample = future.result()
                if sample is not None:
                    with write_lock:
                        outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        outfile.flush()
                    succeeded += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                print(f"  Error on sample {idx}: {e}", file=sys.stderr)

            done = succeeded + failed
            if done % 10 == 0 or done == len(prompts):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(
                    f"  Progress: {done}/{len(prompts)} "
                    f"({succeeded} ok, {failed} failed) "
                    f"[{rate:.1f} samples/s, {elapsed:.0f}s elapsed]"
                )

    elapsed = time.time() - t0
    total = existing + succeeded
    print(f"\nDone: {succeeded} new samples written to {args.output} ({total} total)")
    print(f"Failed: {failed}, Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
