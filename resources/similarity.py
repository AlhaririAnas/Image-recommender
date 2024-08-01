import pickle
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity as cos_similarity
from scipy.spatial import distance
from resources.color_vector import color_histogram
from resources.embedding import inception_v3
from resources.yolo import get_yolo_classes, mean_iou


def euclidean_distance(v1, v2):
    return distance.euclidean(v1, v2)


def manhattan_distance(v1, v2):
    return distance.cityblock(v1, v2)


def cosine_similarity(v1, v2):
    return cos_similarity([v1], [v2])[0][0]


def get_similarities(img, args):
    histogram = color_histogram(np.array(img))
    features = inception_v3(img, args.device)
    yolo_classes = get_yolo_classes(img)

    return [histogram, features, yolo_classes]


def get_most_similar(img_paths, args, similarity_measures):
    color_similarities = []
    embedding_similarities = []
    yolo_similarities = []

    for img_path in img_paths:
        img = Image.open(img_path)
        similarities = get_similarities(img, args)
        if "color" in similarity_measures:
            color_similarities.append(similarities[0])
        if "embedding" in similarity_measures:
            embedding_similarities.append(similarities[1])
        if "yolo" in similarity_measures:
            yolo_similarities.append(similarities[2])

    with open(args.pkl_file, "rb") as f:
        all_similarities = pickle.load(f)

    color_distances = {}
    embedding_distances = {}
    yolo_distances = {}

    for image_id, vectors in all_similarities.items():
        if "color" in similarity_measures:
            color_distance = np.mean([euclidean_distance(similarity, vectors[0]) for similarity in color_similarities])
            color_distances[image_id] = color_distance
        if "embedding" in similarity_measures:
            embedding_distance = np.mean(
                [euclidean_distance(similarity, vectors[1]) for similarity in embedding_similarities]
            )
            embedding_distances[image_id] = embedding_distance
        if "yolo" in similarity_measures:
            yolo_distance = np.mean([mean_iou(similarity, vectors[2]) for similarity in yolo_similarities])
            yolo_distances[image_id] = yolo_distance

    color_most_similar = sorted(color_distances, key=color_distances.get)[:5] if "color" in similarity_measures else []
    embedding_most_similar = (
        sorted(embedding_distances, key=embedding_distances.get)[:5] if "embedding" in similarity_measures else []
    )
    yolo_most_similar = (
        sorted(yolo_distances, key=yolo_distances.get, reverse=True)[:5] if "yolo" in similarity_measures else []
    )

    return color_most_similar, embedding_most_similar, yolo_most_similar
