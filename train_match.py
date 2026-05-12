from args import get_args
from runners.match_runner import train


if __name__ == "__main__":
    args = get_args()
    args.task = "match"
    args.mode = "train"
    train(args)
