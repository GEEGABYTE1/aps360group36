Important files

run_pipeline.py normalizes lists, renders PNGs, builds GT, trains the symbol classifier (SVC), makes predictions and evaluates

prep_all_from_inkml.py converts CROHME inkml to pngs and labels.json

svm_baseline_math.py builds gt_latex.json from the CROHME lists

src/config.py contains thresholds & sizes

src/features.py is a feature extractor

src/segmentation.py — connected-component segmentation (I don't remember ever calling or using this but also I pulled an allnighter so my memory might be hazy)

src/train_svm.py trains the SVM 

src/predict.py, src/reconstruct.py, src/eval_tools.py — inference, LaTeX reconstruction, metrics

crohme2019_train.txt, crohme2019_valid.txt, crohme2019_test.txt are all CROHME file lists (repo root)



How to use/construct an SVC:
place inkml dataset in data/crohme2019/<split>/
<split> = train, valid or test. 


Create an enviornment in vs code by pasting this into the debugger/terminal

python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt    # or:
pip install numpy scikit-learn joblib tqdm scikit-image pillow


Then, run the full pipeline:
python run_pipeline.py

finally, train the SVM
python -m src.train_svm `
  --labels ".\data\train\labels.json" `
  --images_dir ".\data\train\pngs" `
  --out ".\models\svm_baseline.joblib"
