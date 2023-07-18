import argparse
from trainer import train_model, test_model

def main(args):
    if args.mode == "train":
        train_model()
    elif args.mode == "test":
        pass
    elif args.mode == "train_test":
        pass

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test", "train_test"], required=True)
    args = parser.parse_args()
    main(args)