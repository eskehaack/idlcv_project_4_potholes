import numpy as np

from utils.read_xml import read_content
from utils.selective_search import selective_search

def iou(boxA, boxB):
    """
    Computes IoU between two bounding boxes.
    Boxes are in [xmin, ymin, xmax, ymax].
    """

    # Coordinates of intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute intersection area
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    # Areas of A and B
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Avoid division by zero
    if areaA == 0 or areaB == 0:
        return 0.0

    # IoU
    return interArea / float(areaA + areaB - interArea)

def compute_iou_matrix(gt_boxes, prop_boxes):
    """
    Computes IoU for all GT boxes vs all proposal boxes.
    
    gt_boxes: list of [xmin, ymin, xmax, ymax]
    prop_boxes: list of [xmin, ymin, xmax, ymax]
    
    Returns:
        iou_matrix  (len(gt) x len(props))
        best_ious_per_gt
    """

    iou_matrix = np.zeros((len(gt_boxes), len(prop_boxes)))

    for i, gt in enumerate(gt_boxes):
        for j, prop in enumerate(prop_boxes):
            iou_matrix[i, j] = iou(gt, prop)

    best_ious = iou_matrix.max(axis=1) if len(prop_boxes) > 0 else np.zeros(len(gt_boxes))

    return iou_matrix, best_ious

def test_boxes(annotation, test_proposals=[100, 200, 500, 1000, 1500, 2000, 3000]):
    # Read XML GT boxes
    filename, gt_boxes = read_content(annotation)

    # Get proposals
    img_path = "/dtu/datasets1/02516/potholes/images/" + filename
    rects = selective_search(img_path, show=False)

    # rects are (x, y, w, h) → convert to xmin, ymin, xmax, ymax
    prop_boxes = [
        [x, y, x + w, y + h]
        for (x, y, w, h) in rects
    ]

    for N in test_proposals:
        _, best_iou = compute_iou_matrix(gt_boxes, prop_boxes[:N])
        print(f"Proposals: {N:4d} | Mean best IoU: {best_iou.mean():.3f}")