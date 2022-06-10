import argparse

from algorithm.ada_in import AdaIN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Model")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--style', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=False)
    parser.add_argument('--save-file', type=str, required=False)

    args = parser.parse_args()

    if args.model == 'adain':
        model = AdaIN(
            checkpoint=args.checkpoint
        )
        model.evaluate(
            content=args.content,
            style=args.style,
            save_filename=args.save_file
        )
