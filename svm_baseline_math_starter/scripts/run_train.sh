#!/usr/bin/env bash
set -e
python -m src.train_svm --labels data/labels.json --images_dir data/images --out models/svm_baseline.joblib
