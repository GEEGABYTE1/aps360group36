import numpy as np
from skimage.feature import hog
from skimage.measure import moments_hu
from skimage.transform import resize

def _geometry_feats(crop):
    h, w = crop.shape
    area = h * w
    ink = (crop > 0.5).sum() / area if area > 0 else 0.0
    ys, xs = np.nonzero(crop > 0.5)
    if len(xs) == 0:
        cx = cy = 0.0
        ar = 0.0
    else:
        cx = xs.mean() / max(w, 1)
        cy = ys.mean() / max(h, 1)
        ar = w / max(h, 1)
    return np.array([h, w, ar, ink, cx, cy], dtype=np.float32)

def _zoning(crop, rows=6, cols=6):
    h, w = crop.shape
    zs = []
    for r in range(rows):
        for c in range(cols):
            y1 = int(r * h / rows); y2 = int((r+1) * h / rows)
            x1 = int(c * w / cols); x2 = int((c+1) * w / cols)
            cell = crop[y1:y2, x1:x2]
            zs.append((cell > 0.5).mean() if cell.size else 0.0)
    return np.array(zs, dtype=np.float32)

def extract_features(crop, crop_size=48, hog_ppc=(8,8), hog_cpb=(2,2), hog_bins=9, zoning=(6,6)):
    # Early guards to avoid NaNs/inf in resize
    if crop is None or crop.size == 0:
        raise ValueError("empty crop")
    h0, w0 = crop.shape[:2]
    if h0 < 2 or w0 < 2:
        raise ValueError("degenerate crop")

    # Pad to square before resizing
    if crop.shape[0] != crop.shape[1]:
        m = max(crop.shape)
        pad_y = (m - crop.shape[0]) // 2
        pad_x = (m - crop.shape[1]) // 2
        pad = ((pad_y, m - crop.shape[0] - pad_y), (pad_x, m - crop.shape[1] - pad_x))
        crop = np.pad(crop, pad, mode='constant')

    crop = resize(crop, (crop_size, crop_size), anti_aliasing=True)
    # normalize to [0,1]
    denom = (crop.max() - crop.min())
    crop = (crop - crop.min()) / (denom + 1e-8)

    geom = _geometry_feats(crop)
    hu = moments_hu(crop).astype(np.float32)
    h = hog(crop, orientations=hog_bins, pixels_per_cell=hog_ppc, cells_per_block=hog_cpb, feature_vector=True)
    z = _zoning(crop, rows=zoning[0], cols=zoning[1])
    return np.concatenate([geom, hu, h.astype(np.float32), z], axis=0)
