import argparse
import os.path

from tqdm import tqdm

from algorithm.gatys import Gatys
from algorithm.ada_in import AdaIN
from algorithm.cyclegan import CycleGAN
from algorithm.discogan import DiscoGAN
from algorithm.pgan_abt import PaintGAN_Ablation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Model")
    parser.add_argument('--model', help='model name', type=str, required=True)
    parser.add_argument('--content-dir', help='content directory', type=str, required=True)
    parser.add_argument('--style', help='style directory / file', type=str, required=False)
    parser.add_argument('--checkpoint', help='checkpoint directory / file', type=str, required=False)
    parser.add_argument('--save-file', help='save file name', type=str, required=False)

    args = parser.parse_args()

    if args.model == 'adain':
        model = AdaIN(
            checkpoint=args.checkpoint,
            mode='evaluate'
        )

        if not os.path.exists(args.content_dir) or not os.path.exists(args.style):
            raise FileNotFoundError(f"Content Directory or Style is not Found : {args.content_dir}, {args.style}")

        eval_paths = os.listdir(args.content_dir)
        style_path = args.style

        for eval_img in tqdm(eval_paths):
            model.evaluate(os.path.join(args.content_dir, eval_img), style_path, save_filename=args.save_file)

    if args.model == 'cyclegan':
        model = CycleGAN(
            checkpoint=args.checkpoint,
            mode='evaluate'
        )

        if not os.path.exists(args.content_dir):
            raise FileNotFoundError(f"Content Directory is not Found : {args.content_dir}")

        eval_paths = os.listdir(args.content_dir)

        for eval_img in tqdm(eval_paths):
            model.evaluate(
                os.path.join(args.content_dir, eval_img),
                save_filename=args.save_file
            )

    if args.model == 'gatys':
        if not os.path.exists(args.content_dir) or not os.path.exists(args.style):
            raise FileNotFoundError(f"Content Directory or Style is not Found : {args.content_dir}, {args.style}")

        eval_paths = os.listdir(args.content_dir)
        style_path = args.style

        for eval_img in tqdm(eval_paths):
            model = Gatys(
                content_dir=os.path.join(args.content_dir, eval_img),
                style_dir=style_path,
                epochs=10,
                mode='evaluate'
            )

            model.evaluate(
                save_filename=args.save_file
            )

    if args.model == 'discogan':
        model = DiscoGAN(
            checkpoint=args.checkpoint,
            mode='evaluate'
        )

        if not os.path.exists(args.content_dir):
            raise FileNotFoundError(f"Content Directory is not Found : {args.content_dir}")

        eval_paths = os.listdir(args.content_dir)

        for eval_img in tqdm(eval_paths):
            model.evaluate(
                os.path.join(args.content_dir, eval_img),
                save_filename=args.save_file
            )

    if args.model == 'pgan_abt':
        model = PaintGAN_Ablation(
            checkpoint=args.checkpoint,
            mode='evaluate'
        )

        if not os.path.exists(args.content_dir):
            raise FileNotFoundError(f"Content Directory is not Found : {args.content_dir}")

        eval_paths = os.listdir(args.content_dir)

        for eval_img in tqdm(eval_paths):
            model.evaluate(
                os.path.join(args.content_dir, eval_img),
                save_filename=args.save_file
            )







