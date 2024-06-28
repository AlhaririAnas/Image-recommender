import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models.inception import Inception_V3_Weights


def prepocess(img):
    try:
        transform = transforms.Compose([transforms.Resize(299), transforms.CenterCrop(299), transforms.ToTensor()])
        img = transform(img)
        return img.unsqueeze(0)
    except Exception as e:
        print(f"Error loading image {img}: {e}")
        return None


def inception_v3(img, device):
    model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval()
    with torch.no_grad():
        model.to(device)
        img = prepocess(img)
        return model(img).cpu().numpy().flatten()
