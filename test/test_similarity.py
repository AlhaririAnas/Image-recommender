import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image
from resources.similarity import get_most_similar, euclidean_distance, manhattan_distance, cosine_similarity


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


@pytest.fixture
def mock_image_open():
    with patch("PIL.Image.open") as mock_open:
        mock_image = MagicMock(spec=Image.Image)
        mock_open.return_value = mock_image
        yield mock_open


@pytest.fixture
def mock_get_similarities():
    with patch("resources.similarity.get_similarities") as mock_get_similarities:
        mock_get_similarities.return_value = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6]),
            {0: [[10, 20, 30, 40]]},
        ]
        yield mock_get_similarities


def test_get_most_similar(mock_image_open, mock_get_similarities):
    img_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]
    args = MagicMock()
    similarity_measures = ["color", "embedding", "yolo"]
    distance_measure = "euclidean"
    all_similarities = {
        1: [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6]), {0: [[10, 20, 30, 40]]}],
        2: [np.array([0.2, 0.3, 0.4]), np.array([0.5, 0.6, 0.7]), {1: [[50, 60, 70, 80]]}],
    }
    color_clusters = {"model": MagicMock(), "scaler": MagicMock(), "vector_ids": [1, 2]}
    embedding_clusters = {"model": MagicMock(), "scaler": MagicMock(), "vector_ids": [1, 2]}

    color_clusters["model"].predict.return_value = [0]
    embedding_clusters["model"].predict.return_value = [0]

    color_clusters["model"].labels_ = [0, 0]
    embedding_clusters["model"].labels_ = [0, 0]

    color_clusters["scaler"].transform.return_value = [[0.1, 0.2, 0.3]]
    embedding_clusters["scaler"].transform.return_value = [[0.4, 0.5, 0.6]]

    result = get_most_similar(
        img_paths, args, similarity_measures, distance_measure, all_similarities, color_clusters, embedding_clusters
    )

    assert len(result) == 3
    assert len(result[0]) <= 5
    assert len(result[1]) <= 5
    assert len(result[2]) <= 5
