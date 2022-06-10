import argparse

import tensorflow as tf

from algorithm.ada_in import AdaIN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Model")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--style', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=False)
    parser.add_argument('--steps-per-epoch', type=int, required=False)
    parser.add_argument('--batch-size', type=int, required=False)

    args = parser.parse_args()

    if args.model == 'adain':
        model = AdaIN(
            content_dir=args.content,
            style_dir=args.style,
            epochs=args.epochs,
            steps_per_epochs=args.steps_per_epoch,
            batch_size=args.batch_size,
        )
        model.train()