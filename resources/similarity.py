from resources.embedding import inception_v3
from resources.yolo import get_yolo_classes

### import other similarity measure methods


def get_similarities(img, args):
    features = inception_v3(img, args.device)
    yolo_classes = get_yolo_classes(img)
    ### other similarities

    return [features, yolo_classes]
