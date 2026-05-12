from args import get_args
from runners.range_runner import train


if __name__ == "__main__":
    args = get_args()
    args.task = "range"
    args.mode = "train"
    train(args)
