from resources.similarity import *
import numpy as np

v1 = np.array([0.1, 0.2, 0.4, 0.6])
v2 = np.array([0.9, 0.6, 0.3, 0])


def test_euclidean_distance():
    result = euclidean_distance(v1, v2)
    expected = np.sqrt(np.sum((v1 - v2) ** 2))
    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_manhattan_distance():
    result = manhattan_distance(v1, v2)
    expected = np.sum(np.abs(v1 - v2))
    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_cosine_similarity():
    result = cosine_similarity(v1, v2)

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    expected = dot_product / (norm_v1 * norm_v2)
    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"
