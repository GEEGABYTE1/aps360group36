# SVM Baseline for Handwritten Math (CROHME)

A minimal, well-structured baseline for symbol classification with an SVM and rule-based expression reconstruction.

## Quick start
```bash
# create env (example with conda or venv)
pip install -r requirements.txt

# 1) Prepare data (rendered 256x256 PNGs live in ./data/images)
#    labels.json maps image_id -> list of {bbox:[y1,x1,y2,x2], label:str}
#    (You can generate this from your InkML pipeline.)

# 2) Train SVM on cropped symbols
python -m src.train_svm --labels data/labels.json --images_dir data/images --out models/svm_baseline.joblib

# 3) Predict + reconstruct LaTeX on a set
python -m src.predict --images_dir data/images --model models/svm_baseline.joblib --out data/preds.json
python -m src.reconstruct --components data/preds.json --out data/latex_pred.json

# 4) Evaluate
python -m src.eval_tools --gt data/gt_latex.json --pred data/latex_pred.json --report reports/report.json
```

## Structure
- `src/segmentation.py` – connected-components / stroke clustering stubs
- `src/features.py` – Hu moments, HOG, zoning, geometry
- `src/train_svm.py` – sklearn Pipeline + GridSearchCV
- `src/predict.py` – segment → crop → features → predict per component
- `src/reconstruct.py` – rule-based superscripts/subscripts/fractions
- `src/eval_tools.py` – symbol- and expression-level metrics (BLEU)

Fill in the TODOs and plug into your existing InkML renderer.
