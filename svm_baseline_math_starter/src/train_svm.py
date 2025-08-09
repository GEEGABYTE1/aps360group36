import argparse, json, os, numpy as np
from tqdm import tqdm
from joblib import dump
from skimage.io import imread
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from .features import extract_features
from .config import Config

def load_samples(labels_path, images_dir):
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    X, y = [], []
    skipped = 0

    for img_id, ann in tqdm(labels.items(), desc="Loading"):
        img_path = os.path.join(images_dir, img_id)
        if not os.path.exists(img_path):
            skipped += 1
            continue

        img = imread(img_path, as_gray=True)
        H, W = img.shape[:2]

        for obj in ann:
            # clamp bbox to image bounds + validate
            y1, x1, y2, x2 = obj["bbox"]

            y1 = max(0, min(int(y1), H - 1))
            y2 = max(0, min(int(y2), H))
            x1 = max(0, min(int(x1), W - 1))
            x2 = max(0, min(int(x2), W))

            if y2 <= y1 or x2 <= x1:  # degenerate after clamping
                skipped += 1
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                skipped += 1
                continue

            try:
                feat = extract_features(crop, crop_size=Config.crop_size)
            except Exception:
                skipped += 1
                continue

            X.append(feat)
            y.append(obj["label"])

    if skipped:
        print(f"[warn] skipped {skipped} invalid crops")

    if len(X) == 0:
        raise ValueError("No valid training samples after bbox validation.")

    return np.stack(X), np.array(y)

def main(args):
    X, y = load_samples(args.labels, args.images_dir)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
        ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True))
    ])

    # Compact, high-value grid; verbose shows per-fold progress
    param_grid = {"svm__C":[1,3,10], "svm__gamma":[1e-3,3e-3,1e-2]}
    clf = GridSearchCV(pipe, param_grid, scoring="f1_macro", cv=5, n_jobs=-1, verbose=1)
    clf.fit(X, y)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dump(clf.best_estimator_, args.out)
    print("Saved model to", args.out)
    print("Best params:", clf.best_params_)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
