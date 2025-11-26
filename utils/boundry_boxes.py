import os
import cv2
import matplotlib.pyplot as plt
from utils.read_xml import read_content

def draw_boxes_on_image(
    annotation,
    images_dir="/dtu/datasets1/02516/potholes/images",
    color=(0, 0, 0),
    thickness=1,
    figsize=(12, 8),
    show=True,
    save_path=None,
):
    """
    annotation: path to the XML annotation file
    images_dir: directory where the corresponding images are stored
    """

    name, boxes = read_content(annotation)

    img_path = os.path.join(images_dir, name)
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"cv2 failed to read image at {img_path}")

    img_draw = img.copy()
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(img_draw, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness)
        # optional: label with index
        cv2.putText(img_draw, str(i+1), (int(xmin)+3, int(ymin)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    img_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)

    if show:
        plt.figure(figsize=figsize)
        plt.axis('off')
        plt.imshow(img_rgb)
        plt.show()

    if save_path:
        cv2.imwrite(save_path, img_draw)