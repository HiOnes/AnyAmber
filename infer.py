from args import get_args
from runners import match_runner
from runners.end2end_runner import infer_end2end
from runners.infer_runner import infer_egat


def main():
    args = get_args()
    args.mode = "infer"
    if args.task == "egat":
        return infer_egat(args)
    if args.task == "match":
        return match_runner.infer(args)
    if args.task == "end2end":
        return infer_end2end(args)
    raise ValueError("infer.py supports --task egat, match, or end2end.")


if __name__ == "__main__":
    main()
