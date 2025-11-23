import cv2
import matplotlib.pyplot as plt

def selective_search(image_path, color=(0, 0, 0), thickness=1, figsize=(12, 8), show=True, save_path=None):

    img = cv2.imread(f"{image_path}")
    if img is None:
        raise RuntimeError(f"cv2 failed to read image at {image_path}")
    
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()

    print(f"Total Number of Region Proposals: {len(rects)}")
    img_draw = img.copy()
    for i, rect in enumerate(rects[:100]):  # Draw only first 100 proposals
        x, y, w, h = rect
        cv2.rectangle(img_draw, (x, y), (x + w, y + h), color, thickness)
    
    img_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)

    if show:
        plt.figure(figsize=figsize)
        plt.axis('off')
        plt.imshow(img_rgb)
        plt.show()

    if save_path:
        cv2.imwrite(save_path, img_draw)
    
    return rects