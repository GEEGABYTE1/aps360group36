import argparse, json, numpy as np
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

def bleu_score(reference, hypothesis, n=4):
    # Tiny BLEU for tokens split by space
    def ngrams(seq, k):
        return Counter([tuple(seq[i:i+k]) for i in range(max(0,len(seq)-k+1))])
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    weights = [0.25,0.25,0.25,0.25][:n]
    score = 0.0
    for k, w in zip(range(1, n+1), weights):
        ref_ng = ngrams(ref_tokens, k); hyp_ng = ngrams(hyp_tokens, k)
        overlap = sum((ref_ng & hyp_ng).values())
        total = max(sum(hyp_ng.values()), 1)
        p_k = overlap / total
        score += w * (np.log(p_k + 1e-12))
    bp = 1.0 if len(hyp_tokens) > len(ref_tokens) else np.exp(1 - len(ref_tokens)/max(len(hyp_tokens),1))
    return float(bp * np.exp(score))

def eval_expressions(gt_map, pred_map):
    exact = 0; bleus = []
    for k, gt in gt_map.items():
        pred = pred_map.get(k, "")
        if pred.strip() == gt.strip():
            exact += 1
        bleus.append(bleu_score(gt, pred))
    exact_match = exact / max(len(gt_map),1)
    return {"exact_match": exact_match, "bleu": float(np.mean(bleus))}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    with open(args.gt,"r") as f: gt = json.load(f)
    with open(args.pred,"r") as f: pd = json.load(f)

    rep = eval_expressions(gt, pd)
    with open(args.report,"w") as f: json.dump(rep, f, indent=2)
    print(rep)
