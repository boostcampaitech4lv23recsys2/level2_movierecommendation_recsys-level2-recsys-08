import argparse
from util import inference
import warnings
warnings.filterwarnings('ignore')

def main(args):
    model_path = args.model_path
    topk = int(args.top_k)
    model_name = model_path[8:-4].split('-')[0]

    inference(model_name=model_name, topk=topk, model_path=model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--top_k", default = 10, type=int)
    args = parser.parse_args()

    main(args)