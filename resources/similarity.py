import pickle
from resources.embedding import inception_v3

### import other similarity measure methods


def get_similarities(img, args):
    features = inception_v3(img, args.device)
    ### other similarities

    return [features]
