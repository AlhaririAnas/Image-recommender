import os
import torch
import pickle
from tqdm import tqdm
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from resources.generator import ImageGenerator
from resources.metadata_reader import (
    create_database,
    get_metadata,
    save_metadata_in_database,
    get_last_entry,
    get_filename_from_id,
)
from resources.similarity import get_similarities
from app.app import app, start_app


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("-m", "--metadata", action="store_true")

parser.add_argument("-s", "--similarity", action="store_true")

parser.add_argument("-p", "--path", action="store", default="D:/")

parser.add_argument("-d", "--device", type=str, default=None, help="Device to use: cuda or cpu")

parser.add_argument("--pkl_file", action="store", default="similarities.pkl")

parser.add_argument("--checkpoint", type=int, default=100)

parser.add_argument("--debug", action="store_true")

args = parser.parse_args()


def run(args):
    """
    Runs the main image processing pipeline based on the provided arguments.

    Args:
        args: The arguments containing device information, file paths, and processing options.

    Returns:
        None
    """
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    create_database()
    try:
        with open(args.pkl_file, "rb") as f:
            similarities = pickle.load(f)
    except FileNotFoundError:
        similarities = defaultdict()
        similarities[1] = None

    last_db_id = get_last_entry()
    last_sim_id = max(similarities.keys())
    id = min(last_sim_id, last_db_id)

    if id == 0:
        starting_path = None
    else:
        starting_path = os.join(args.path, get_filename_from_id(id))
    img_gen = ImageGenerator(args.path).image_generator(starting_path=starting_path)

    for img in tqdm(img_gen, total=447584, initial=id):
        id += 1
        if args.metadata and last_db_id < id:
            metadata = get_metadata(img)
            save_metadata_in_database(metadata)
        if args.similarity:
            try:
                similarities[id] = get_similarities(img, args)
            except OSError:
                continue
            if id % args.checkpoint == 0:
                with open(args.pkl_file, "wb") as f:
                    pickle.dump(similarities, f)
    with open(args.pkl_file, "wb") as f:
        pickle.dump(similarities, f)


def create_and_save_clustering_model(vectors, vector_ids, filename, clusters, method="kmeans"):
    """
    Creates and saves a clustering model based on the input vectors.

    Parameters:
        vectors (array): The input vectors for clustering.
        vector_ids (array): The IDs corresponding to the input vectors.
        filename (str): The name of the file to save the clustering model.
        clusters (int): The number of clusters to create.
        method (str, optional): The clustering method to use, defaults to "kmeans".

    Returns:
        None
    """
    scaler = StandardScaler()
    vectors_scaled = scaler.fit_transform(vectors)

    if method == "dbscan":
        model = DBSCAN(eps=0.5, min_samples=5)
    elif method == "kmeans":
        model = KMeans(n_clusters=clusters, random_state=0)
    else:
        raise ValueError("Unsupported clustering method")

    model.fit(vectors_scaled)

    with open(filename, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "vector_ids": vector_ids}, f)
    print(f"Clustering model saved to {filename}")


def load_pkl_files():
    """
    A function to load pickle files containing similarities, color clusters, and embedding clusters.
    If the files are not found, it creates the clusters using the create_and_save_clustering_model function.
    """
    print("Loading similarities from pickle file...")
    try:
        with open(args.pkl_file, "rb") as f:
            similarities = pickle.load(f)
            app.config["SIMILARITIES"] = similarities
    except FileNotFoundError:
        raise ValueError("No similarities found! Run the script with the -s flag.")
    if os.path.exists("color_cluster.pkl"):
        with open("color_cluster.pkl", "rb") as f:
            app.config["COLOR_CLUSTER"] = pickle.load(f)
    else:
        print("No color cluster found. Creating...")
        create_and_save_clustering_model(
            [similarities[v][0] for v in similarities.keys()],
            [v for v in similarities.keys()],
            filename="color_cluster.pkl",
            clusters=41,
        )
    if os.path.exists("embedding_cluster.pkl"):
        with open("embedding_cluster.pkl", "rb") as f:
            app.config["EMBEDDING_CLUSTER"] = pickle.load(f)
    else:
        print("No embedding cluster found. Creating...")
        create_and_save_clustering_model(
            [similarities[v][1] for v in similarities.keys()],
            [v for v in similarities.keys()],
            filename="embedding_cluster.pkl",
            clusters=45,
        )

    print("Done!")


if __name__ == "__main__":
    if args.metadata or args.similarity:
        run(args)
    else:
        app.config["ARGS"] = args
        load_pkl_files()
        if args.debug:
            app.run(debug=True)
        else:
            start_app()
