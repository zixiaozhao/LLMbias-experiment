import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

from src.prompts.bias_bank import bias_suffix_for_filename, normalize_bias_type


def _parse_bias_type(value: str) -> str:
    try:
        return normalize_bias_type(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def score(args):
    splits = load_dataset("mattymchen/codejudgebench", args.task).keys()
    canonical_bias_type = normalize_bias_type(getattr(args, "bias_type", "none"))
    bias_suffix = bias_suffix_for_filename(canonical_bias_type)
    output_dir = Path(args.output_dir)

    all_score = []
    all_difficulty = []
    for s in splits:
        model_id = args.model_name.rstrip('/').split('/')[-1]
        base = f"{model_id}_{args.task}-{s}"
        filepath = output_dir / f"{base}{bias_suffix}.jsonl"
        if filepath.exists():
            with filepath.open('r', encoding='utf-8') as f:
                result = [json.loads(line) for line in f.readlines()]
            score = [item['pred'] == item['label'] for item in result]
            all_score.extend(score)
            all_difficulty.extend([item['difficulty'] for item in result])
            print(filepath, f"{np.mean(score)*100:.2f}")
        else:
            print("MISSING", filepath)

    print(f"================== {args.task} ==================")
    easy_score = [x for x, d in zip(all_score, all_difficulty) if d == "easy"]
    med_score = [x for x, d in zip(all_score, all_difficulty) if d == "medium"]
    hard_score = [x for x, d in zip(all_score, all_difficulty) if d == "hard"]
    print("Easy", f"{np.mean(easy_score)*100:.2f}")
    print("Medium", f"{np.mean(med_score)*100:.2f}")
    print("Hard", f"{np.mean(hard_score)*100:.2f}")

    print("Micro Avg", f"{np.mean(all_score)*100:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--task", choices=["codegen", "coderepair", "testgen"], required=True
    )
    parser.add_argument("--bias_type", type=_parse_bias_type, default="none")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    score(args)
