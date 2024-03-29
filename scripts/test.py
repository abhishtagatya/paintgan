import argparse

from algorithm.gatys import Gatys
from algorithm.ada_in import AdaIN
from algorithm.cyclegan import CycleGAN
from algorithm.discogan import DiscoGAN
from algorithm.pgan_abt import PaintGAN_Ablation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference Model")
    parser.add_argument('--model', help='model name', type=str, required=True)
    parser.add_argument('--content', help='content directory / file', type=str, required=True)
    parser.add_argument('--style', help='style directory / file', type=str, required=False)
    parser.add_argument('--checkpoint', help='checkpoint directory / file', type=str, required=False)
    parser.add_argument('--save-file', help='save file name', type=str, required=False)

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

    if args.model == 'discogan':
        model = DiscoGAN(
            checkpoint=args.checkpoint,
            mode='inference'
        )

        model.evaluate(
            content=args.content,
            save_filename=args.save_file
        )

    if args.model == 'pgan_abt':
        model = PaintGAN_Ablation(
            checkpoint=args.checkpoint,
            mode='inference'
        )

        model.evaluate(
            content=args.content,
            save_filename=args.save_file
        )
