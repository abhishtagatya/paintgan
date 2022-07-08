import argparse

import tensorflow as tf

from algorithm.gatys import Gatys
from algorithm.ada_in import AdaIN
from algorithm.cyclegan import CycleGAN
from algorithm.pgan import PaintGAN
from algorithm.pgan_abt import PaintGAN_Ablation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Model")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--style', type=str, required=True)

    parser.add_argument('--epochs', type=int, required=False)
    parser.add_argument('--steps-per-epoch', type=int, required=False)
    parser.add_argument('--batch-size', type=int, required=False)
    parser.add_argument('--buffer-size', type=int, required=False)
    parser.add_argument('--learning-rate', type=int, required=False)

    parser.add_argument('--content-domain', type=str, required=False)
    parser.add_argument('--style-domain', type=str, required=False)
    parser.add_argument('--max-set', type=int, required=False)

    args = parser.parse_args()

    # Mapping
    model = args.model
    content = args.content
    style = args.style
    epochs = args.epochs if args.epochs is not None else 1
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch is not None else 1
    batch_size = args.batch_size if args.batch_size is not None else 1

    if args.model == 'adain':
        model = AdaIN(
            content_dir=args.content,
            style_dir=args.style,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
        )
        model.train()

    if args.model == 'cyclegan':
        buffer_size = args.buffer_size if args.buffer_size is not None else 1
        content_domain = args.content_domain if args.content_domain is not None else ''
        style_domain = args.style_domain if args.style_domain is not None else ''
        max_set = args.max_set if args.max_set is not None else 0

        model = CycleGAN(
            content_dir=args.content,
            style_dir=args.style,
            domain=args.style_domain,
            epochs=args.epochs,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            max_set=max_set
        )

        model.train()

    if args.model == 'gatys':
        model = Gatys(
            content_dir=args.content,
            style_dir=args.style,
            epochs=args.epochs,
        )

        model.train()

    if args.model == 'pgan':
        content_domain = args.content_domain if args.content_domain is not None else ''
        style_domain = args.style_domain if args.style_domain is not None else ''
        max_set = args.max_set if args.max_set is not None else 0

        model = PaintGAN(
            content_dir=args.content,
            style_dir=args.style,
            domain=args.style_domain,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_set=max_set
        )

        model.train()

    if args.model == 'pgan_abt':
        model = PaintGAN_Ablation(
            content_dir=args.content,
            style_dir=args.style,
            domain=args.style_domain,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )

        model.train()
