from src.runner import ORIGINAL, REVERSED, build_parser, run_evaluation


if __name__ == "__main__":
    parser = build_parser(default_output_dir="outputs")
    run_evaluation(parser.parse_args(), orientations=(ORIGINAL, REVERSED))
