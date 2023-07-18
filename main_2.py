import argparse
from trainer import train_model, test_model
import torch

def main(args):
    if args.mode == "train":
        train_model()
    elif args.mode == "test":
        pass
    elif args.mode == "train_test":
        pass

if __name__=="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test", "train_test"], required=True)
    args = parser.parse_args()
    main(args)