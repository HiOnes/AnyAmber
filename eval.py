from args import get_args
from runners import egat_runner, match_runner, range_runner
from runners.end2end_runner import evaluate_end2end


def main():
    args = get_args()
    args.mode = "eval"
    if args.task == "match":
        return match_runner.evaluate(args)
    if args.task == "range":
        return range_runner.evaluate(args)
    if args.task == "egat":
        return egat_runner.evaluate(args)
    if args.task == "end2end":
        return evaluate_end2end(args)
    raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
