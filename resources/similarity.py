import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos_similarity
from scipy.spatial import distance
from resources.color_vector import color_histogram
from resources.embedding import inception_v3
from resources.yolo import get_yolo_classes

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
