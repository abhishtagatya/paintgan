import argparse

from algorithm.gatys import Gatys
from algorithm.ada_in import AdaIN
from algorithm.cyclegan import CycleGAN

from evaluate.deception_score import DeceptionScore

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Score Model")
    parser.add_argument('--metric', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--eval-dir', type=str, required=False)
    parser.add_argument('--eval-dataset', type=str, required=False)
    parser.add_argument('--checkpoint', type=str, required=False)

    args = parser.parse_args()

    if args.metric == 'deception':
        d_score = DeceptionScore(
            model_name=args.model,
            eval_dataset_file=args.eval_dataset,
            checkpoint=args.checkpoint
        )
        d_score.score(args.eval_dir)
