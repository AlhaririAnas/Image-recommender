import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models.inception import Inception_V3_Weights


def prepocess(img):
    """
    A function that preprocesses an image for the Inception v3 model.

    Parameters:
    img: input image to be preprocessed.

    Returns:
    Preprocessed image ready for model input.
    """
    try:
        transform = transforms.Compose([transforms.Resize(299), transforms.CenterCrop(299), transforms.ToTensor()])
        img = transform(img)

        if img.unsqueeze(0).shape[1] == 1:  # If the photo consists of grayscale => triple it
            img = torch.cat([img, img, img], dim=0)

        return img.unsqueeze(0)

    except Exception as e:
        print(f"Error loading image {img}: {e}")
        return None


def inception_v3(img, device):
    """
    This function runs an Inception v3 model on the input image to extract features and returns a flattened numpy array.

    Parameters:
    img: the input image to be processed.
    device: the device (cpu or gpu) on which to run the model.

    Returns:
    A flattened numpy array of extracted features from the input image.
    """
    model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval()
    with torch.no_grad():
        model.to(device)
        img = prepocess(img).to(device)
        return model(img).cpu().numpy().flatten()
