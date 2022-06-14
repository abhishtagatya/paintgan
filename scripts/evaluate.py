import argparse
import os.path

from tqdm import tqdm

from algorithm.gatys import Gatys
from algorithm.ada_in import AdaIN
from algorithm.cyclegan import CycleGAN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Model")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--content-dir', type=str, required=True)
    parser.add_argument('--style', type=str, required=False)
    parser.add_argument('--checkpoint', type=str, required=False)
    parser.add_argument('--save-file', type=str, required=False)

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





