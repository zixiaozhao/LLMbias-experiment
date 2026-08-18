# LLMbias

Experiment code for measuring bias in LLM-as-a-judge evaluations on
[CodeJudgeBench](https://github.com/hongcha0/CodeJudgeBench).

This repository intentionally contains only the files needed to run and score
the experiments. It does not include cluster-specific
automation, model weights, datasets, logs, or generated results.

## Requirements

- Linux with an NVIDIA GPU
- Python 3.12 (the version used for the verified experiment environment)
- A CUDA setup compatible with PyTorch and vLLM
- Enough GPU memory for the selected judge model

Install the Python dependencies in a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first run downloads the selected model and the
`mattymchen/codejudgebench` dataset from Hugging Face unless they are already
cached.

## Run an experiment

Run a small smoke test first:

```bash
python run.py \
  --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
  --task codegen \
  --split gemini_2.5_pro \
  --max_samples 2 \
  --output_dir outputs
```

`run.py` evaluates both the original and reversed A/B answer order. Progress is
saved after every batch in a `.jsonl.partial` file and resumes automatically.
Use `--overwrite` only when you want to restart an existing result.

Run the same experiment with a bias cue:

```bash
python run.py \
  --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
  --task codegen \
  --bias_type authority:A \
  --output_dir outputs
```

Supported bias names are `authority`, `verbosity`, `modelname`, `bandwagon`,
`distraction`, `finalonly`, `sentiment`, `diversity`, `cot`, `selfenhance`, and
`refined`. Append `:A` or `:B` to choose the side; omitting the side defaults to
`:A`. Use `none` for the unbiased baseline.

The supported tasks are `codegen`, `coderepair`, and `testgen`. Run
`python run.py --help` for all generation and resume options.

## Score results

Use the same model, task, bias, and output directory used for generation:

```bash
python -m src.score \
  --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
  --task codegen \
  --bias_type authority:A \
  --output_dir outputs
```

## Attribution

This project contains modified code from
[CodeJudgeBench](https://github.com/hongcha0/CodeJudgeBench), which is licensed
under Apache License 2.0. The modifications add configurable bias cues,
orientation-aware execution, resumable atomic result writing, and experiment
metadata.

If you use the benchmark, cite:

```bibtex
@article{jiang2025codejudgebench,
  title   = {CodeJudgeBench: Benchmarking LLM-as-a-Judge for Coding Tasks},
  author  = {Hongchao Jiang and Yiming Chen and Yushi Cao and Hung-yi Lee and Robby T. Tan},
  year    = {2025},
  journal = {arXiv preprint arXiv:2507.10535}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
