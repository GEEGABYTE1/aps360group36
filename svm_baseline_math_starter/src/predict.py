import argparse, os, json
import numpy as np
from joblib import load
from tqdm import tqdm
from skimage.io import imread
from .segmentation import segment_components
from .features import extract_features
from .config import Config

def predict_on_dir(model_path, images_dir):
    clf = load(model_path)
    out = {}
    for fname in tqdm(sorted(os.listdir(images_dir))):
        if not fname.lower().endswith((".png",".jpg",".jpeg")): 
            continue
        img = imread(os.path.join(images_dir, fname), as_gray=True)
        boxes = segment_components(img, min_area=Config.min_area)
        preds = []
        for (y1,x1,y2,x2) in boxes:
            crop = img[y1:y2, x1:x2]
            feat = extract_features(crop, crop_size=Config.crop_size)[None, :]
            proba = clf.predict_proba(feat)[0]
            label = clf.classes_[np.argmax(proba)]
            preds.append({"bbox":[int(y1),int(x1),int(y2),int(x2)], "label":str(label), "score":float(np.max(proba))})
        out[fname] = preds
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    preds = predict_on_dir(args.model, args.images_dir)
    with open(args.out, "w") as f:
        json.dump(preds, f, indent=2)
    print("Saved", args.out)
