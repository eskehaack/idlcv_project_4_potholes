#!/usr/bin/env python3
"""
Test script for Non-Maximum Suppression (NMS) functionality.
This script demonstrates how to use the NMS implementation with images and bounding boxes.
"""

import sys
import os
import numpy as np

# Add the current directory to the path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.nms import (
        non_max_suppression, 
        apply_nms_to_image, 
        apply_nms_with_custom_boxes, 
        test_nms_example
    )
    print("Successfully imported NMS functions")
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure opencv-python is installed: pip install opencv-python")
    sys.exit(1)


def demo_nms_basic():
    """
    Demonstrate basic NMS functionality with simple overlapping boxes.
    """
    print("\n" + "="*50)
    print("DEMO 1: Basic NMS with overlapping boxes")
    print("="*50)
    
    # Define some overlapping bounding boxes
    boxes = [
        [50, 50, 150, 150],    # Box 1 - high confidence
        [60, 60, 160, 160],    # Box 2 - overlaps with Box 1
        [200, 200, 300, 300], # Box 3 - separate, medium confidence  
        [55, 55, 145, 145],    # Box 4 - heavily overlaps with Box 1
        [210, 210, 310, 310], # Box 5 - overlaps with Box 3
    ]
    
    scores = [0.9, 0.7, 0.8, 0.6, 0.5]
    
    print(f"Original boxes: {len(boxes)}")
    print(f"Scores: {scores}")
    
    # Apply NMS with different thresholds
    for threshold in [0.3, 0.5, 0.7]:
        keep_indices = non_max_suppression(boxes, scores, iou_threshold=threshold)
        kept_boxes = [boxes[i] for i in keep_indices]
        
        print(f"\nNMS with IoU threshold {threshold}:")
        print(f"  Kept indices: {keep_indices}")
        print(f"  Kept {len(kept_boxes)}/{len(boxes)} boxes")


def demo_nms_with_xml(annotation_path=None):
    """
    Demonstrate NMS with actual XML annotation file.
    """
    print("\n" + "="*50)
    print("DEMO 2: NMS with XML annotation")
    print("="*50)
    
    if annotation_path is None:
        print("No annotation path provided. Skipping XML demo.")
        print("To test with actual annotations, provide path to XML file.")
        return
    
    if not os.path.exists(annotation_path):
        print(f"Annotation file not found: {annotation_path}")
        return
    
    try:
        # Apply NMS to image from XML annotation
        kept_boxes, keep_indices = apply_nms_to_image(
            annotation_path, 
            images_dir="potholes/images",
            iou_threshold=0.5,
            save_path="nms_result.png"
        )
        print(f"Successfully applied NMS to {annotation_path}")
        print(f"Result saved to: nms_result.png")
        
    except Exception as e:
        print(f"Error processing XML annotation: {e}")


def demo_synthetic_image():
    """
    Create a synthetic image with overlapping boxes and apply NMS.
    """
    print("\n" + "="*50)
    print("DEMO 3: NMS with synthetic image")
    print("="*50)
    
    try:
        # This will create a test image and demonstrate NMS visually
        test_nms_example()
        print("Synthetic image NMS demo completed successfully!")
        
    except Exception as e:
        print(f"Error in synthetic demo: {e}")


def demo_nms_with_npy(npy_path, images_dir="potholes/images"):
    """
    Demonstrate NMS with proposals from .npy file.
    """
    print("\n" + "="*50)
    print("DEMO 4: NMS with .npy proposals")
    print("="*50)
    
    if not os.path.exists(npy_path):
        print(f"Proposals file not found: {npy_path}")
        return

    try:
        # Load proposals
        boxes = np.load(npy_path)
        print(f"Loaded {len(boxes)} proposals from {npy_path}")
        
        # Infer image path
        # filename is like "potholes0_props.npy" -> image is "potholes0.png"
        basename = os.path.basename(npy_path)
        # Try to find the matching image
        # Strategy 1: Replace _props.npy with .png
        image_name = basename.replace("_props.npy", ".png")
        image_path = os.path.join(images_dir, image_name)
        
        if not os.path.exists(image_path):
             # Strategy 2: Just take the part before the first underscore if it looks like potholesX
             prefix = basename.split('_')[0] 
             image_name = prefix + ".png"
             image_path = os.path.join(images_dir, image_name)
        
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            print(f"Searched for: {image_name} in {images_dir}")
            return
            
        # Load image
        import cv2
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return
            
        # Generate dummy scores
        # Random scores between 0.0 and 1.0
        scores = np.random.uniform(0.0, 1.0, size=len(boxes))
        
        # Apply NMS
        kept_boxes, keep_indices = apply_nms_with_custom_boxes(
            image, boxes, scores, iou_threshold=0.1, title_prefix="NPY Proposals"
        )
        
        print(f"Original boxes: {len(boxes)}")
        print(f"Boxes after NMS: {len(kept_boxes)}")
        print(f"Removed boxes: {len(boxes) - len(kept_boxes)}")
        
        print(f"Successfully applied NMS to {npy_path}")
        
    except Exception as e:
        print(f"Error processing .npy file: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Main function to run all NMS demonstrations.
    """
    print("Non-Maximum Suppression (NMS) Test Suite")
    print("========================================")
    
    # Demo 1: Basic NMS functionality
    # demo_nms_basic()
    
    # Demo 2: NMS with XML (if available)
    # You can modify this path to point to your actual XML annotation files
    # xml_path = "potholes/annotations/potholes0.xml"  # Set to actual path if you have XML annotations
    # demo_nms_with_xml(xml_path)
    
    # Demo 3: NMS with synthetic image (visual demonstration)
    # demo_synthetic_image()
    
    # Demo 4: NMS with .npy proposals
    npy_path = "P4_1/labeled_proposals/potholes0_props.npy"
    demo_nms_with_npy(npy_path)
    
    print("\n" + "="*50)
    print("All NMS demos completed!")
    print("="*50)
    
    # Additional usage examples
    print("\nUsage Examples:")
    print("1. Basic NMS on boxes and scores:")
    print("   from utils.nms import non_max_suppression")
    print("   keep_indices = non_max_suppression(boxes, scores, iou_threshold=0.5)")
    
    print("\n2. NMS with XML annotation:")
    print("   from utils.nms import apply_nms_to_image")
    print("   kept_boxes, indices = apply_nms_to_image('path/to/annotation.xml')")
    
    print("\n3. NMS with custom image and boxes:")
    print("   from utils.nms import apply_nms_with_custom_boxes")
    print("   kept_boxes, indices = apply_nms_with_custom_boxes(image, boxes, scores)")


if __name__ == "__main__":
    main()