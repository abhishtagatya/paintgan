import argparse

from algorithm.gatys import Gatys
from algorithm.ada_in import AdaIN
from algorithm.cyclegan import CycleGAN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference Model")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--style', type=str, required=False)
    parser.add_argument('--checkpoint', type=str, required=False)
    parser.add_argument('--save-file', type=str, required=False)

    args = parser.parse_args()

    if args.model == 'adain':
        model = AdaIN(
            checkpoint=args.checkpoint,
            mode='inference'
        )
        model.evaluate(
            content=args.content,
            style=args.style,
            save_filename=args.save_file
        )

    if args.model == 'cyclegan':
        model = CycleGAN(
            checkpoint=args.checkpoint,
            mode='inference'
        )

        model.evaluate(
            content=args.content,
            save_filename=args.save_file
        )

    if args.model == 'gatys':
        model = Gatys(
            content_dir=args.content,
            style_dir=args.style,
            epochs=10,
            mode='inference'
        )

        model.evaluate(
            save_filename=args.save_file
        )
