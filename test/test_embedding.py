import pytest
import torch
from PIL import Image
from unittest.mock import patch
from resources.embedding import prepocess, inception_v3


def create_dummy_image(mode="RGB"):
    img = Image.new(mode, (299, 299), color=(255))
    return img


# Test prepocess function
def test_prepocess_valid_image():
    img = create_dummy_image()
    processed_img = prepocess(img)
    assert processed_img is not None
    assert processed_img.shape == (1, 3, 299, 299)


def test_prepocess_grayscale_image():
    img = create_dummy_image(mode="L")
    processed_img = prepocess(img)
    assert processed_img is not None
    assert processed_img.shape == (1, 3, 299, 299)


def test_prepocess_invalid_input():
    invalid_input = "not an image"
    processed_img = prepocess(invalid_input)
    assert processed_img is None


# Mock for torch model to avoid actual model loading and computation
class MockModel(torch.nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()

    def forward(self, x):
        return torch.ones((1, 2048, 8, 8))


# Test inception_v3 function
@pytest.fixture
def mock_model():
    with patch("torchvision.models.inception_v3", return_value=MockModel()) as _fixture:
        yield _fixture


def test_inception_v3_valid_image(mock_model):
    img = create_dummy_image()
    device = "cpu"
    features = inception_v3(img, device)
    assert features is not None
    assert features.shape == (2048 * 8 * 8,)


def test_inception_v3_invalid_device(mock_model):
    img = create_dummy_image()
    invalid_device = "invalid_device"
    with pytest.raises(Exception):
        features = inception_v3(img, invalid_device)
