# run_pipeline.py
# End-to-end pipeline for the SVM baseline:
#  1) Normalize CROHME lists (handles 1-col or tabbed lines)
#  2) InkML -> PNG (train/valid/test)
#  3) Generate gt_latex.json for each split
#  4) Train SVM (progress bar, compact grid)
#  5) Predict + Reconstruct + Evaluate (valid + test)
#
# Usage:
#   .\.venv\Scripts\Activate
#   python run_pipeline.py

import os
import sys
import json
import traceback
import subprocess
from pathlib import Path

# ---------- BASE PATHS (auto) ----------
BASE = Path(__file__).resolve().parent
PY = sys.executable

# Try to auto-detect the renderer script name
CANDIDATE_RENDERERS = ["prep_all_from_inkml.py", "prep_inkml_to_pngs.py"]
RENDERER = None
for name in CANDIDATE_RENDERERS:
    p = BASE / name
    if p.exists():
        RENDERER = str(p)
        break

# Required helper for gt generation
SVM_BASELINE_MATH = BASE / "svm_baseline_math.py"

# Inputs
INKML_ROOT = str(BASE / "data")  # parent of 'crohme2019'
LIST_TRAIN = str(BASE / "crohme2019_train.txt")
LIST_VALID = str(BASE / "crohme2019_valid.txt")
LIST_TEST  = str(BASE / "crohme2019_test.txt")

# Normalized 1-column lists (auto-created next to originals)
LIST_TRAIN_1C = str(BASE / "crohme2019_train_1col.txt")
LIST_VALID_1C = str(BASE / "crohme2019_valid_1col.txt")
LIST_TEST_1C  = str(BASE / "crohme2019_test_1col.txt")

# Outputs
OUT_ROOT   = str(BASE / "data")
TRAIN_PNGS = str(BASE / "data" / "train" / "pngs")
VALID_PNGS = str(BASE / "data" / "valid" / "pngs")
TEST_PNGS  = str(BASE / "data" / "test" / "pngs")

GT_TRAIN = str(BASE / "data" / "train" / "gt_latex.json")
GT_VALID = str(BASE / "data" / "valid" / "gt_latex.json")
GT_TEST  = str(BASE / "data" / "test" / "gt_latex.json")

MODEL_PATH   = str(BASE / "models" / "svm_baseline.joblib")
VALID_COMPS  = str(BASE / "data" / "valid" / "preds.json")
TEST_COMPS   = str(BASE / "data" / "test" / "preds.json")
VALID_LATEX  = str(BASE / "data" / "valid" / "latex_pred.json")
TEST_LATEX   = str(BASE / "data" / "test" / "latex_pred.json")
VALID_REPORT = str(BASE / "reports" / "valid_report.json")
TEST_REPORT  = str(BASE / "reports" / "test_report.json")

# ---------- HELPERS ----------
def print_env():
    print("=== ENV CHECK ===")
    print("Working dir:", os.getcwd())
    print("Script dir  :", BASE)
    print("Interpreter :", PY)
    try:
        import numpy as np
        import sklearn, joblib, skimage, PIL, tqdm
        print("numpy       :", np.__version__)
        print("scikit-learn:", sklearn.__version__)
        print("joblib      :", joblib.__version__)
        print("scikit-image:", skimage.__version__)
        print("Pillow      :", PIL.__version__)
        print("tqdm        :", tqdm.__version__)
    except Exception as e:
        print("[warn] Could not import one or more libs:", e)
    print("="*60)

def sh(cmd, check=True):
    print("\n>>>", " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    print(f"[cmd exit] returncode={res.returncode}")
    if check and res.returncode != 0:
        raise SystemExit(f"Command failed ({res.returncode}): {' '.join(cmd)}")
    return res.returncode

def require_file(path_str, hint=None):
    p = Path(path_str)
    print(f"[check] file exists? {p}  -> {p.exists()}")
    if not p.exists():
        msg = f"Required file not found: {p}"
        if hint:
            msg += f"\nHint: {hint}"
        raise SystemExit(msg)

def ensure_dirs():
    for p in [TRAIN_PNGS, VALID_PNGS, TEST_PNGS, Path(MODEL_PATH).parent, Path(VALID_REPORT).parent]:
        Path(p).mkdir(parents=True, exist_ok=True)
    print("[ok] ensured output dirs")

def normalize_list(in_path, out_path):
    require_file(in_path, "Make sure the CROHME list exists in your project root.")
    lines = []
    with open(in_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Accept either "rel\tlabel" or just "rel" (ignore any extra whitespace)
            rel = line.split('\t', 1)[0].split()[0]
            lines.append(rel)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[ok] wrote 1-column list: {out_path}  ({len(lines)} items)")

# ---------- STEPS ----------
def sanity_checks():
    print("=== SANITY CHECKS ===")
    print("[check] renderer candidates:", CANDIDATE_RENDERERS)
    print("[check] renderer chosen     :", RENDERER)
    if RENDERER is None:
        print("[files in base dir]:", [p.name for p in BASE.iterdir()])
        raise SystemExit("No renderer found. Place one of the candidates in the project root shown above.")
    print("[check] svm_baseline_math.py:", SVM_BASELINE_MATH.exists())
    if not SVM_BASELINE_MATH.exists():
        raise SystemExit(f"Missing required script: {SVM_BASELINE_MATH}")
    print("[check] list files:")
    print("  -", LIST_TRAIN, Path(LIST_TRAIN).exists())
    print("  -", LIST_VALID, Path(LIST_VALID).exists())
    print("  -", LIST_TEST,  Path(LIST_TEST).exists())
    print("="*60)

def render_pngs():
    print("\n=== 1) Normalize list files ===")
    normalize_list(LIST_TRAIN, LIST_TRAIN_1C)
    normalize_list(LIST_VALID, LIST_VALID_1C)
    normalize_list(LIST_TEST,  LIST_TEST_1C)

    print("\n=== 2) Render PNGs from InkML (train/valid/test) ===")
    # Show one resolved example
    sample_rel = None
    with open(LIST_TRAIN_1C, "r", encoding="utf-8") as f:
        for line in f:
            sample_rel = line.strip()
            if sample_rel:
                break
    if sample_rel:
        full = Path(INKML_ROOT) / sample_rel
        print("[sanity] example InkML path:", full, "exists?", full.exists())
        if not full.exists():
            print("[warn] Example file not found. INKML_ROOT must be the PARENT of 'crohme2019'")
            print("       INKML_ROOT =", INKML_ROOT)

    sh([PY, RENDERER,
        "--inkml_root", INKML_ROOT,
        "--train_list", LIST_TRAIN_1C,
        "--valid_list", LIST_VALID_1C,
        "--test_list",  LIST_TEST_1C,
        "--out_root", OUT_ROOT,
        "--write_train_labels"
    ])

def gen_gt():
    print("\n=== 3) Generate gt_latex.json (train/valid/test) ===")
    sh([PY, str(SVM_BASELINE_MATH), LIST_TRAIN_1C, GT_TRAIN])
    sh([PY, str(SVM_BASELINE_MATH), LIST_VALID_1C, GT_VALID])
    sh([PY, str(SVM_BASELINE_MATH), LIST_TEST_1C,  GT_TEST])

def train_svm_with_progress():
    """
    Compact, high-value grid (C in {1,3,10}, gamma in {1e-3,3e-3,1e-2}) with
    3-fold CV and a progress bar. ~27 fits total.
    """
    print("\n=== 4) Train SVM on train split (progress bar) ===")
    import numpy as np
    from joblib import dump
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold, ParameterGrid
    from sklearn.base import clone
    from sklearn.metrics import f1_score
    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(x, **kwargs): return x  # fallback

    # Reuse the project’s loader
    sys.path.insert(0, str(BASE))
    from src.train_svm import load_samples

    lbls = Path(OUT_ROOT) / "train" / "labels.json"
    require_file(str(lbls), "Re-run step 2 with --write_train_labels to create labels.json.")

    print("[load] labels:", lbls)
    print("[load] images:", TRAIN_PNGS)
    X, y = load_samples(str(lbls), TRAIN_PNGS)
    print("[data] X:", X.shape, " y:", y.shape, " classes:", len(set(y)))

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
        ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True))
    ])

    param_grid = {"svm__C":[1,3,10], "svm__gamma":[1e-3,3e-3,1e-2]}
    grid = list(ParameterGrid(param_grid))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    best_score, best_params = -1.0, None
    print(f"[search] {len(grid)} configs × 3-fold CV = {len(grid)*3} fits")
    for params in tqdm(grid, desc="Grid", unit="cfg"):
        model = clone(pipe).set_params(**params)
        scores = []
        for tr, va in cv.split(X, y):
            model.fit(X[tr], y[tr])
            pred = model.predict(X[va])
            scores.append(f1_score(y[va], pred, average="macro"))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score, best_params = mean_score, params
            print(f"[best so far] {best_params}  f1_macro={best_score:.4f}")

    print(f"[best] params={best_params}  f1_macro={best_score:.4f}")
    best_model = clone(pipe).set_params(**best_params)
    best_model.fit(X, y)
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    dump(best_model, MODEL_PATH)
    print("[ok] saved model:", MODEL_PATH)

def predict_and_eval(split):
    print(f"\n=== 5) Predict + reconstruct + evaluate ({split}) ===")
    if split == "valid":
        imgs, comps, latex, gt, report = VALID_PNGS, VALID_COMPS, VALID_LATEX, GT_VALID, VALID_REPORT
    else:
        imgs, comps, latex, gt, report = TEST_PNGS, TEST_COMPS, TEST_LATEX, GT_TEST, TEST_REPORT

    sh([PY, "-m", "src.predict",
        "--images_dir", imgs,
        "--model", MODEL_PATH,
        "--out", comps
    ])
    sh([PY, "-m", "src.reconstruct",
        "--components", comps,
        "--out", latex
    ])
    sh([PY, "-m", "src.eval_tools",
        "--gt", gt,
        "--pred", latex,
        "--report", report
    ])

def main():
    print_env()
    os.chdir(BASE)
    sanity_checks()
    ensure_dirs()
    render_pngs()
    gen_gt()
    train_svm_with_progress()
    predict_and_eval("valid")
    predict_and_eval("test")
    print("\n=== DONE ===")
    print("Model:", MODEL_PATH)
    print("Reports:")
    print(" -", VALID_REPORT)
    print(" -", TEST_REPORT)

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print("\n[EXIT]", e)
    except Exception:
        print("\n[UNCAUGHT ERROR] Traceback follows:")
        traceback.print_exc()
