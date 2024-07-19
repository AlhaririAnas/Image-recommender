import pytest
from unittest.mock import MagicMock, patch
from resources.yolo import get_yolo_classes, calculate_iou, mean_iou

# Mocking the YOLO model
@pytest.fixture
def mock_yolo_model():
    with patch("resources.yolo.model", autospec=True) as mock_model:
        yield mock_model

def test_get_yolo_classes(mock_yolo_model):
    img = "dummy_image"

    mock_boxes = [
        MagicMock(cls=0, xyxy=[(10, 20, 30, 40)]),
        MagicMock(cls=1, xyxy=[(50, 60, 70, 80)]),
    ]
    mock_yolo_model.return_value = [MagicMock(boxes=mock_boxes)]

    results = get_yolo_classes(img)

    assert results == {
        0: [[10, 20, 30, 40]],
        1: [[50, 60, 70, 80]],
    }

def test_calculate_iou():
    box1 = [10, 20, 30, 40]
    box2 = [20, 30, 40, 50]
    expected_iou = 1 / 7
    assert calculate_iou(box1, box2) == pytest.approx(expected_iou)

    box3 = [10, 20, 30, 40]
    box4 = [10, 20, 30, 40]
    expected_iou = 1.0
    assert calculate_iou(box3, box4) == expected_iou

    box5 = [0, 0, 10, 10]
    box6 = [20, 20, 30, 30]
    expected_iou = 0.0
    assert calculate_iou(box5, box6) == expected_iou

def test_mean_iou():
    dict1 = {
        0: [[10, 20, 30, 40]],
        1: [[50, 60, 70, 80]],
    }
    dict2 = {
        0: [[10, 20, 30, 40]],
        1: [[55, 65, 75, 85]],
    }
    expected_mean_iou = (1.0 + calculate_iou([50, 60, 70, 80], [55, 65, 75, 85])) / 2
    assert mean_iou(dict1, dict2) == pytest.approx(expected_mean_iou)

    dict3 = {
        0: [[0, 0, 10, 10]],
    }
    dict4 = {
        0: [[20, 20, 30, 30]],
    }
    expected_mean_iou = 0.0
    assert mean_iou(dict3, dict4) == expected_mean_iou

    dict5 = {}
    dict6 = {
        0: [[10, 20, 30, 40]],
    }
    expected_mean_iou = 0.0
    assert mean_iou(dict5, dict6) == expected_mean_iou

if __name__ == "__main__":
    pytest.main()
