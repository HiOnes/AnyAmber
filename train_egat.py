from args import get_args
from runners.egat_runner import train


if __name__ == "__main__":
    args = get_args()
    args.task = "egat"
    args.mode = "train"
    train(args)
