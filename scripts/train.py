import argparse

import tensorflow as tf

from algorithm.gatys import Gatys
from algorithm.ada_in import AdaIN
from algorithm.cyclegan import CycleGAN
from algorithm.discogan import DiscoGAN
from algorithm.pgan_abt import PaintGAN_Ablation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
        Train Models from Scratch
    """)
    parser.add_argument('--model', help='model name', type=str, required=True)
    parser.add_argument('--content', help='content directory / file', type=str, required=True)
    parser.add_argument('--style', help='style directory / file', type=str, required=True)

    parser.add_argument('--epochs', help='epochs to train', type=int, required=False)
    parser.add_argument('--steps-per-epoch', help='steps per epoch', type=int, required=False)
    parser.add_argument('--batch-size', help='batch size', type=int, required=False)
    parser.add_argument('--buffer-size', help='buffer size', type=int, required=False)
    parser.add_argument('--learning-rate', help='learning rate', type=float, required=False)

    parser.add_argument('--content-domain', help='content domain (for PDPM)', type=str, required=False)
    parser.add_argument('--style-domain', help='style domain (for PDPM)', type=str, required=False)
    parser.add_argument('--max-set', help='max set for dataset', type=int, required=False)

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

    if args.model == 'discogan':
        content_domain = args.content_domain if args.content_domain is not None else ''
        style_domain = args.style_domain if args.style_domain is not None else ''
        max_set = args.max_set if args.max_set is not None else 0

        model = DiscoGAN(
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
