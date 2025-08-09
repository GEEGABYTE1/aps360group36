import argparse, json
import numpy as np
from .config import Config

def _center(box):
    y1,x1,y2,x2 = box
    return ( (y1+y2)/2.0, (x1+x2)/2.0 ), (y2-y1, x2-x1)

def reconstruct(preds_for_image, cfg: Config = Config()):
    # Very simple greedy reconstruction: LEFT->RIGHT with super/sub attachment
    # preds_for_image: list of dicts with keys "bbox", "label"
    # returns LaTeX string
    # Sort by x then y
    items = sorted(preds_for_image, key=lambda o: (o["bbox"][1], o["bbox"][0]))
    out = ""
    used = set()
    for i, obj in enumerate(items):
        if i in used: 
            continue
        box = obj["bbox"]; label = obj["label"]
        (cy, cx), (h, w) = _center(box)
        base = label
        # search for super/sub among following few tokens
        sup, sub = None, None
        for j in range(i+1, min(i+6, len(items))):
            if j in used: 
                continue
            b2 = items[j]["bbox"]; (cy2, cx2), (h2, w2) = _center(b2)
            dx = cx2 - cx; dy = cy - cy2
            if dx < -cfg.tau_x_right * w: 
                continue
            if dy > cfg.tau_sup_y * h and abs(cx2 - cx) < 1.2*w:
                sup = items[j]["label"]; used.add(j)
            if -dy > cfg.tau_sub_y * h and abs(cx2 - cx) < 1.2*w:
                sub = items[j]["label"]; used.add(j)
        if sup and sub:
            out += f"{base}^{{{sup}}}_{{{sub}}} "
        elif sup:
            out += f"{base}^{{{sup}}} "
        elif sub:
            out += f"{base}_{{{sub}}} "
        else:
            out += f"{base} "
    return out.strip()

def main(args):
    with open(args.components, "r") as f:
        comps = json.load(f)
    out = {}
    for img_id, preds in comps.items():
        out[img_id] = reconstruct(preds, Config())
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved", args.out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--components", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
