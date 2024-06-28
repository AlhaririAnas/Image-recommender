import torch
import pickle
from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from resources.generator import ImageGenerator
from resources.metadata_reader import create_database, get_metadata, save_metadata_in_database
from resources.similarity import get_similarities


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("-m", "--metadata", action="store_true")

parser.add_argument("-s", "--similarity", action="store_true")

parser.add_argument("-p", "--path", action="store")

parser.add_argument("-d", "--device", type=str, default=None, help="Device to use: cuda or cpu")

parser.add_argument("--pkl_file", action="store", default="similarities.pkl")

parser.add_argument("--checkpoint", type=int, default=100)

args = parser.parse_args()


def run(args):
    img_gen = ImageGenerator(args.path).image_generator()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    create_database()
    similarities = defaultdict()
    id = 0
    for img in tqdm(img_gen, total=447375):
        id += 1
        if args.metadata:
            metadata = get_metadata(img)
            save_metadata_in_database(metadata)
        if args.similarity:
            similarities[id] = get_similarities(img, args)
            if id % args.checkpoint:
                with open(args.pkl_file, "wb") as f:
                    pickle.dump(similarities, f)


if __name__ == "__main__":
    run(args)
