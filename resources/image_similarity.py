from scipy.spatial import distance
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity as cos_similarity

def euclidean_distance(v1, v2):
    return distance.euclidean(v1, v2)

def manhattan_distance(v1, v2):
    return distance.cityblock(v1, v2)

def cosine_similarity(v1, v2):
    return cos_similarity([v1], [v2])[0][0]

def jaccard_similarity(v1, v2):
    return jaccard_score(v1, v2, average='weighted')

def hamming_distance(v1, v2):
    return distance.hamming(v1, v2)
