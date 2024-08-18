from ultralytics import YOLO


model = YOLO("yolov8s.pt")


def get_yolo_classes(img):
    """
    Extracts classes and bounding boxes from a YOLO model prediction on the input image.

    Parameters:
    img: input image for YOLO prediction.

    Returns:
    results: a dictionary containing classes as keys and lists of bounding boxes as values.
    """
    results = {}
    boxes = model(img, verbose=False)[0].boxes

    for box in boxes:
        object = int(box.cls)
        if object not in results.keys():
            results[object] = [list(map(int, box.xyxy[0]))]
        else:
            results[object].append(list(map(int, box.xyxy[0])))
    return results


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Parameters:
        box1 (list): List of 4 integers representing the coordinates of the first bounding box.
        box2 (list): List of 4 integers representing the coordinates of the second bounding box.

    Returns:
        float: The IoU value between the two bounding boxes.
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    iou = inter_area / union_area
    return iou


def mean_iou(dict1, dict2):
    """
    Calculate the mean Intersection over Union (IoU) for detected objects between two images.

    Parameters:
        dict1, dict2: dictionaries
        Each dictionary has labels as keys and lists of bounding boxes as values.

    Returns:
        float: Mean IoU value.
    """

    iou_list = [
        calculate_iou(box1, box2) for label in dict1 if label in dict2 for box1 in dict1[label] for box2 in dict2[label]
    ]

    if not iou_list:
        return 0.0

    total = sum(len(value_list) for value_list in dict1.values())
    matches = sum(len(value_list) == len(dict2[key]) if key in dict2 else 0 for key, value_list in dict1.items())

    match_ratio = matches / total if total > 0 else 0
    mean_iou = sum(iou_list) / len(iou_list) * match_ratio

    return mean_iou


if __name__ == "__main__":
    import pickle

    with open("similarities.pkl", "rb") as f:
        metadata = pickle.load(f)
        input_image = metadata[2][1]

    most_similar_image = (0.0, 0)
    for k in range(2, 22):
        mean_iou_value = mean_iou(input_image, metadata[k][1])
        if mean_iou_value > most_similar_image[0]:
            most_similar_image = (mean_iou_value, k)

    print("Mean IoU:", most_similar_image)
