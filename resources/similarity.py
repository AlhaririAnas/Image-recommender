from itertools import chain
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity as cos_similarity
from scipy.spatial import distance
from resources.color_vector import color_histogram
from resources.embedding import inception_v3
from resources.yolo import get_yolo_classes, mean_iou
from resources.timeit import timeit


def euclidean_distance(v1, v2):
    return distance.euclidean(v1, v2)

def manhattan_distance(v1, v2):
    return distance.cityblock(v1, v2)

def cosine_similarity(v1, v2):
    return cos_similarity([v1], [v2])[0][0]

@timeit
def get_similarities(img, args):
    histogram = color_histogram(np.array(img))
    features = inception_v3(img, args.device)
    yolo_classes = get_yolo_classes(img)

    return [histogram, features, yolo_classes]

@timeit
def get_most_similar(
    img_paths, args, similarity_measures, distance_measure, all_similarities, color_clusters, embedding_clusters
):
    similarities = []
    yolo_similarities = []

    color_distances = {}
    embedding_distances = {}
    yolo_distances = {}

    distance_func = {"euclidean": euclidean_distance, "manhattan": manhattan_distance, "cosine": cosine_similarity}[
        distance_measure
    ]

    for img_path in img_paths:
        img = Image.open(img_path)
        sim = get_similarities(img, args)
        similarities.append(sim)
        if "yolo" in similarity_measures:
            yolo_similarities.append(sim[2])

    if "color" in similarity_measures:
        color_similarity = np.mean([similarity[0] for similarity in similarities], axis=0)
        model, scaler, vector_ids = color_clusters["model"], color_clusters["scaler"], color_clusters["vector_ids"]
        color_similarity_scaled = scaler.transform([color_similarity])
        cluster_label = model.predict(color_similarity_scaled)[0]
        same_cluster_ids = [vector_ids[i] for i, label in enumerate(model.labels_) if label == cluster_label]
        for image_id in same_cluster_ids:
            color_distance = np.mean([distance_func(all_similarities[image_id][0], color_similarity)])
            color_distances[image_id] = color_distance

    if "embedding" in similarity_measures:
        embedding_similarity = np.mean([similarity[1] for similarity in similarities], axis=0)
        model, scaler, vector_ids = (
            embedding_clusters["model"],
            embedding_clusters["scaler"],
            embedding_clusters["vector_ids"],
        )
        embedding_similarity_scaled = scaler.transform([embedding_similarity])
        cluster_label = model.predict(embedding_similarity_scaled)[0]
        same_cluster_ids = [vector_ids[i] for i, label in enumerate(model.labels_) if label == cluster_label]
        for image_id in same_cluster_ids:
            embedding_distance = np.mean([distance_func(all_similarities[image_id][1], embedding_similarity)])
            embedding_distances[image_id] = embedding_distance

    if "yolo" in similarity_measures:
        yolo_classes = list(chain.from_iterable(yolo_sim.keys() for yolo_sim in yolo_similarities))
        for image_id, vectors in all_similarities.items():
            if any(item in list(vectors[2].keys()) for item in yolo_classes):
                yolo_distance = np.mean([mean_iou(similarity, vectors[2]) for similarity in yolo_similarities])
                yolo_distances[image_id] = yolo_distance
    print(len(color_distances), len(embedding_distances), len(yolo_distances))
    color_most_similar = sorted(color_distances, key=color_distances.get)[:5] if "color" in similarity_measures else []
    embedding_most_similar = (
        sorted(embedding_distances, key=embedding_distances.get)[:5] if "embedding" in similarity_measures else []
    )
    yolo_most_similar = (
        sorted(yolo_distances, key=yolo_distances.get, reverse=True)[:5] if "yolo" in similarity_measures else []
    )

    return color_most_similar, embedding_most_similar, yolo_most_similar
