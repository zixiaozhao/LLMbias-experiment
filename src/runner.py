import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from src.prompts.bias_bank import (
    bias_side,
    bias_suffix_for_filename,
    normalize_bias_type,
)
from src.result_store import JsonlResultStore
from src.tasks import get_task


DATASET_NAME = "mattymchen/codejudgebench"
ORIGINAL = "original"
REVERSED = "reversed"
VALID_ORIENTATIONS = {ORIGINAL, REVERSED}


def _parse_bias_type(value: str) -> str:
    try:
        return normalize_bias_type(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise argparse.ArgumentTypeError(
            "run_id may contain only letters, digits, dot, underscore, and hyphen."
        )
    return value


def build_parser(default_output_dir: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--task", choices=["codegen", "coderepair", "testgen"], required=True
    )
    parser.add_argument("--split", type=str, default="all")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_id", type=_parse_run_id, default="default")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Process at most this many samples per split (intended for smoke tests).",
    )
    parser.add_argument("--bias_type", type=_parse_bias_type, default="none")
    parser.add_argument("--output_dir", type=str, default=default_output_dir)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start a new partial result and atomically replace any existing final file.",
    )
    return parser


def _sample_id(item: Dict, task_name: str, split: str, sample_index: int) -> str:
    for field in ("sample_id", "problem_id", "question_id", "id"):
        value = item.get(field)
        if value is not None and str(value):
            return str(value)
    return f"{task_name}:{split}:{sample_index}"


def process_batch(
    indexed_batch: Sequence[Tuple[int, Dict]],
    task_class,
    model,
    *,
    orientation: str,
    model_name: str,
    task_name: str,
    split: str,
    canonical_bias_type: str,
    requested_generation_config: Dict,
    run_id: str = "default",
) -> List[Dict]:
    if orientation not in VALID_ORIENTATIONS:
        raise ValueError(f"Unknown orientation: {orientation}")

    reverse = orientation == REVERSED
    samples = [
        task_class.from_dict({**item, "reverse": reverse})
        for _, item in indexed_batch
    ]
    judgments = model.judge(samples)
    if len(judgments) != len(indexed_batch):
        raise RuntimeError(
            f"Model returned {len(judgments)} judgments for {len(indexed_batch)} samples."
        )

    positive_position = "B" if reverse else "A"
    canonical_bias_side = bias_side(canonical_bias_type)
    records = []
    for (sample_index, item), judgment in zip(indexed_batch, judgments):
        record = dict(judgment)
        record.update(
            {
                "dataset": DATASET_NAME,
                "sample_id": _sample_id(item, task_name, split, sample_index),
                "sample_index": sample_index,
                "model_name": model_name,
                "task": task_name,
                "split": split,
                "orientation": orientation,
                "label": positive_position,
                "positive_position": positive_position,
                "difficulty": item["difficulty"],
                "bias_type": canonical_bias_type,
                "bias_side": canonical_bias_side,
                "requested_generation_config": dict(requested_generation_config),
                "run_id": run_id,
                "bias_on_positive": (
                    None
                    if canonical_bias_side is None
                    else canonical_bias_side == positive_position
                ),
            }
        )
        records.append(record)
    return records


def run_evaluation(args, *, orientations: Iterable[str]) -> None:
    orientations = tuple(orientations)
    if not orientations or any(item not in VALID_ORIENTATIONS for item in orientations):
        raise ValueError(f"Invalid orientations: {orientations}")
    if len(set(orientations)) != len(orientations):
        raise ValueError("Orientations must be unique.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if args.max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max_samples must be greater than zero when provided.")

    canonical_bias_type = normalize_bias_type(args.bias_type)
    args.bias_type = canonical_bias_type
    requested_generation_config = {
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }

    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME, args.task)
    split_names = list(dataset.keys()) if args.split == "all" else [args.split]
    task_class = get_task(args.task)

    output_dir = Path(args.output_dir)
    model_id = args.model_name.rstrip("/").split("/")[-1]
    suffix = bias_suffix_for_filename(canonical_bias_type)
    canonical_bias_side = bias_side(canonical_bias_type)

    pending_plans = []
    for split in split_names:
        split_data = dataset[split]
        if args.max_samples is not None:
            split_data = split_data.select(
                range(min(args.max_samples, len(split_data)))
            )
        limit_suffix = (
            f"-limit{args.max_samples}" if args.max_samples is not None else ""
        )
        output_file = (
            output_dir
            / f"{model_id}_{args.task}-{split}{limit_suffix}{suffix}.jsonl"
        )
        context = {
            "dataset": DATASET_NAME,
            "model_name": args.model_name,
            "task": args.task,
            "split": split,
            "bias_type": canonical_bias_type,
            "bias_side": canonical_bias_side,
            "requested_generation_config": requested_generation_config,
            "run_id": args.run_id,
        }
        store = JsonlResultStore(
            output_file,
            sample_count=len(split_data),
            orientations=orientations,
            expected_context=context,
        )
        if store.prepare(overwrite=args.overwrite):
            print(f"COMPLETE {output_file}")
        else:
            pending_plans.append((split, split_data, store))

    if not pending_plans:
        return

    from tqdm import tqdm

    from src.models.factory import ModelFactory

    model = ModelFactory.get_model(args.model_name)(args)
    for split, split_data, store in pending_plans:
        total = len(split_data) * len(orientations)
        with tqdm(
            total=total,
            initial=len(store.completed),
            desc=f"{args.task}/{split}",
            unit="judgment",
        ) as progress:
            for start in range(0, len(split_data), args.batch_size):
                indices = range(start, min(start + args.batch_size, len(split_data)))
                for orientation in orientations:
                    pending_indices = [
                        index
                        for index in indices
                        if not store.is_completed(index, orientation)
                    ]
                    if not pending_indices:
                        continue
                    indexed_batch = [
                        (index, split_data[index]) for index in pending_indices
                    ]
                    records = process_batch(
                        indexed_batch,
                        task_class,
                        model,
                        orientation=orientation,
                        model_name=args.model_name,
                        task_name=args.task,
                        split=split,
                        canonical_bias_type=canonical_bias_type,
                        requested_generation_config=requested_generation_config,
                        run_id=args.run_id,
                    )
                    store.append(records)
                    progress.update(len(records))

        store.finalize()
        print(f"WROTE {store.final_path}")
