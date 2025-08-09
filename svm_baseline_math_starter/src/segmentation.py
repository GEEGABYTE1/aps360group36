import numpy as np
from skimage import filters, measure, morphology

def segment_components(img, min_area=20):
    """
    img: 2D np.array (grayscale 0..1)
    returns: list of (y1, x1, y2, x2) bounding boxes
    """
    th = img > filters.threshold_otsu(img)
    th = morphology.remove_small_objects(th, min_size=min_area)
    lbl = measure.label(th)
    boxes = []
    for r in measure.regionprops(lbl):
        if r.area < min_area: 
            continue
        y1, x1, y2, x2 = r.bbox
        boxes.append((y1, x1, y2, x2))
    boxes.sort(key=lambda b: (b[1], b[0]))  # sort by x then y
    return boxes
