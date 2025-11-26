import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from utils.iou import iou
from utils.read_xml import read_content


def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove overlapping bounding boxes.
    
    Args:
        boxes: list of bounding boxes in format [xmin, ymin, xmax, ymax]
        scores: list of confidence scores for each box
        iou_threshold: IoU threshold for suppression (default 0.5)
    
    Returns:
        keep_indices: list of indices of boxes to keep after NMS
    """
    if len(boxes) == 0:
        return []
    
    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Sort boxes by scores in descending order
    sorted_indices = np.argsort(scores)[::-1]
    
    keep_indices = []
    
    while len(sorted_indices) > 0:
        # Take the box with highest score
        current_idx = sorted_indices[0]
        keep_indices.append(current_idx)
        
        if len(sorted_indices) == 1:
            break
            
        # Calculate IoU between current box and remaining boxes
        remaining_indices = sorted_indices[1:]
        ious = []
        
        for idx in remaining_indices:
            iou_val = iou(boxes[current_idx], boxes[idx])
            ious.append(iou_val)
        
        # Remove boxes with IoU greater than threshold
        ious = np.array(ious)
        keep_mask = ious < iou_threshold
        sorted_indices = remaining_indices[keep_mask]
    
    return keep_indices


def load_image_with_boxes(annotation_path, images_dir="/dtu/datasets1/02516/potholes/images"):
    """
    Load an image and its bounding boxes from XML annotation.
    
    Args:
        annotation_path: path to XML annotation file
        images_dir: directory containing the images
    
    Returns:
        image: loaded image (BGR format)
        boxes: list of bounding boxes
        filename: name of the image file
    """
    filename, boxes = read_content(annotation_path)
    img_path = os.path.join(images_dir, filename)
    
    image = cv2.imread(img_path)
    if image is None:
        raise RuntimeError(f"Failed to load image at {img_path}")
    
    return image, boxes, filename


def apply_nms_to_image(annotation_path, scores=None, iou_threshold=0.5, 
                      images_dir="/dtu/datasets1/02516/potholes/images",
                      figsize=(15, 8), save_path=None):
    """
    Load an image with bounding boxes and apply NMS, then visualize results.
    
    Args:
        annotation_path: path to XML annotation file
        scores: list of confidence scores for each box (if None, uses dummy scores)
        iou_threshold: IoU threshold for NMS
        images_dir: directory containing images
        figsize: figure size for visualization
        save_path: path to save the result image (optional)
    
    Returns:
        kept_boxes: list of boxes after NMS
        kept_indices: indices of kept boxes
    """
    # Load image and boxes
    image, boxes, filename = load_image_with_boxes(annotation_path, images_dir)
    
    if len(boxes) == 0:
        print("No bounding boxes found in annotation")
        return [], []
    
    # Generate dummy scores if not provided (decreasing confidence)
    if scores is None:
        scores = [1.0 - i * 0.1 for i in range(len(boxes))]
        scores = [max(0.1, score) for score in scores]  # Ensure minimum score
    
    # Apply NMS
    keep_indices = non_max_suppression(boxes, scores, iou_threshold)
    kept_boxes = [boxes[i] for i in keep_indices]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Original image with all boxes
    img_original = image.copy()
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(img_original, (int(xmin), int(ymin)), (int(xmax), int(ymax)), 
                     (0, 0, 255), 2)  # Red color
        # Add score text
        score_text = f"{scores[i]:.2f}"
        cv2.putText(img_original, score_text, (int(xmin), int(ymin)-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Image after NMS
    img_nms = image.copy()
    for i, idx in enumerate(keep_indices):
        box = boxes[idx]
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(img_nms, (int(xmin), int(ymin)), (int(xmax), int(ymax)), 
                     (0, 255, 0), 2)  # Green color
        # Add score text
        score_text = f"{scores[idx]:.2f}"
        cv2.putText(img_nms, score_text, (int(xmin), int(ymin)-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Convert BGR to RGB for matplotlib
    img_original_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    img_nms_rgb = cv2.cvtColor(img_nms, cv2.COLOR_BGR2RGB)
    
    # Display images
    ax1.imshow(img_original_rgb)
    ax1.set_title(f'Before NMS: {len(boxes)} boxes')
    ax1.axis('off')
    
    ax2.imshow(img_nms_rgb)
    ax2.set_title(f'After NMS (IoU={iou_threshold}): {len(kept_boxes)} boxes')
    ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    print(f"Original boxes: {len(boxes)}")
    print(f"Boxes after NMS: {len(kept_boxes)}")
    print(f"Removed boxes: {len(boxes) - len(kept_boxes)}")
    
    return kept_boxes, keep_indices


def apply_nms_with_custom_boxes(image, boxes, scores, iou_threshold=0.5, 
                               figsize=(15, 8), title_prefix="Custom"):
    """
    Apply NMS to custom boxes and visualize results.
    
    Args:
        image: input image (BGR format)
        boxes: list of bounding boxes [xmin, ymin, xmax, ymax]
        scores: list of confidence scores
        iou_threshold: IoU threshold for NMS
        figsize: figure size for visualization
        title_prefix: prefix for plot titles
    
    Returns:
        kept_boxes: list of boxes after NMS
        kept_indices: indices of kept boxes
    """
    if len(boxes) != len(scores):
        raise ValueError("Number of boxes must match number of scores")
    
    # Apply NMS
    keep_indices = non_max_suppression(boxes, scores, iou_threshold)
    kept_boxes = [boxes[i] for i in keep_indices]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Original image with all boxes
    img_original = image.copy()
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(img_original, (int(xmin), int(ymin)), (int(xmax), int(ymax)), 
                     (0, 0, 255), 2)  # Red color
        # Add score text
        score_text = f"{scores[i]:.2f}"
        cv2.putText(img_original, score_text, (int(xmin), int(ymin)-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Image after NMS
    img_nms = image.copy()
    for i, idx in enumerate(keep_indices):
        box = boxes[idx]
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(img_nms, (int(xmin), int(ymin)), (int(xmax), int(ymax)), 
                     (0, 255, 0), 2)  # Green color
        # Add score text
        score_text = f"{scores[idx]:.2f}"
        cv2.putText(img_nms, score_text, (int(xmin), int(ymin)-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Convert BGR to RGB for matplotlib
    img_original_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    img_nms_rgb = cv2.cvtColor(img_nms, cv2.COLOR_BGR2RGB)
    
    # Display images
    ax1.imshow(img_original_rgb)
    ax1.set_title(f'{title_prefix} - Before NMS: {len(boxes)} boxes')
    ax1.axis('off')
    
    ax2.imshow(img_nms_rgb)
    ax2.set_title(f'{title_prefix} - After NMS (IoU={iou_threshold}): {len(kept_boxes)} boxes')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return kept_boxes, keep_indices


# Example usage and test functions
def test_nms_example():
    """
    Test NMS with synthetic overlapping boxes.
    """
    # Create a dummy image
    image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Define overlapping boxes with scores
    boxes = [
        [100, 100, 200, 200],  # High score box
        [110, 110, 210, 210],  # Overlapping with first (should be suppressed)
        [150, 150, 250, 250],  # Partially overlapping
        [300, 100, 400, 200],  # Non-overlapping box
        [105, 105, 195, 195],  # Highly overlapping with first (should be suppressed)
    ]
    
    scores = [0.9, 0.7, 0.8, 0.6, 0.5]
    
    print("Testing NMS with synthetic boxes:")
    print(f"Input boxes: {len(boxes)}")
    print(f"Scores: {scores}")
    
    kept_boxes, keep_indices = apply_nms_with_custom_boxes(
        image, boxes, scores, iou_threshold=0.5, title_prefix="Test"
    )
    
    print(f"Kept indices: {keep_indices}")
    print(f"Kept boxes: {kept_boxes}")
    
    return kept_boxes, keep_indices


if __name__ == "__main__":
    # Run the test example
    test_nms_example()