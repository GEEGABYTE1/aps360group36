#!/usr/bin/env bash
set -e
python -m src.predict --images_dir data/images --model models/svm_baseline.joblib --out data/preds.json
python -m src.reconstruct --components data/preds.json --out data/latex_pred.json
python -m src.eval_tools --gt data/gt_latex.json --pred data/latex_pred.json --report reports/report.json
