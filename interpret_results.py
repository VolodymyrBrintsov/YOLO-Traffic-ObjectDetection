import numpy as np
def get_outputs(output_annotation, score_threshold):
    # Convert the output annotation to a list of bounding boxes, scores, and classes
    boxes, scores, pred_classes = [], [], []
    for y in range(7):
        for x in range(7):
            if output_annotation[y, x, 4] >= score_threshold:
                boxes.append(output_annotation[y, x, :4])  # Assuming the boxes are in (x1, y1, x2, y2) format
                scores.append(output_annotation[y, x, 4])
                pred_classes.append(output_annotation[y, x, 5:])

    # Convert lists to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)
    pred_classes = np.array(pred_classes)
    return boxes, scores, pred_classes

def non_max_suppression(boxes, scores, pred_classes, score_threshold=0.5, iou_threshold=0.5):
    # Filter boxes based on the score threshold
    mask = scores >= score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    pred_classes = pred_classes[mask]

    # Sort boxes by scores in descending order
    sorted_indices = np.argsort(scores)[::-1]
    boxes = boxes[sorted_indices]
    scores = scores[sorted_indices]
    pred_classes = pred_classes[sorted_indices]

    selected_boxes = []
    selected_scores = []
    selected_classes = []

    while len(boxes) > 0:
        selected_boxes.append(boxes[0])
        selected_scores.append(scores[0])
        selected_classes.append(pred_classes[0])

        # Calculate Intersection over Union (IoU)
        iou = _calculate_iou(boxes[0], boxes[1:])

        # Keep boxes with IoU less than the threshold
        mask = iou <= iou_threshold
        boxes = boxes[1:][mask]
        scores = scores[1:][mask]
        pred_classes = pred_classes[1:][mask]

    return np.array(selected_boxes), np.array(selected_scores), np.array(selected_classes)

def _calculate_iou(box, boxes):
    # Calculate Intersection area
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Calculate Union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection

    # Calculate IoU
    iou = intersection / union

    return iou