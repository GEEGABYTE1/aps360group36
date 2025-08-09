# prep_all_from_inkml.py
# Render CROHME InkML to PNGs and (optionally) write per-symbol labels.json
# Usage (called by run_pipeline.py):
#   python prep_all_from_inkml.py --inkml_root <path/to/data> \
#     --train_list crohme2019_train_1col.txt \
#     --valid_list crohme2019_valid_1col.txt \
#     --test_list  crohme2019_test_1col.txt \
#     --out_root   data \
#     --write_train_labels

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

# -------------------------
# Namespace-agnostic parser
# -------------------------
def _parse_traces(root) -> Dict[str, List[Tuple[float, float]]]:
    traces = {}
    for tr in root.findall(".//{*}trace"):
        tid = tr.get("id")
        if tid is None or tr.text is None:
            continue
        pts: List[Tuple[float, float]] = []
        # CROHME points are comma-separated; each chunk is "x y" (optional time)
        for chunk in tr.text.strip().split(","):
            toks = chunk.strip().replace(",", " ").split()
            if len(toks) >= 2:
                try:
                    x = float(toks[0]); y = float(toks[1])
                    pts.append((x, y))
                except Exception:
                    pass
        if pts:
            traces[tid] = pts
    return traces

import xml.etree.ElementTree as ET
from pathlib import Path

def parse_crohme_symbols_or_whole(inkml_path):
    """
    If symbol annotations exist, return them.
    Otherwise, return one entry for the whole expression.
    """
    tree = ET.parse(inkml_path)
    root = tree.getroot()

    # Map trace id -> points
    traces = {}
    for tr in root.findall(".//{*}trace"):
        tid = tr.get("id")
        pts = []
        for chunk in tr.text.strip().split(","):
            xy = chunk.strip().replace(",", " ").split()
            if len(xy) >= 2:
                try:
                    x = float(xy[0]); y = float(xy[1])
                    pts.append((x, y))
                except:
                    pass
        traces[tid] = pts

    # Try per-symbol traceGroups
    out = []
    for tg in root.findall(".//{*}annotationXML//{*}traceGroup"):
        label = None
        for ann in tg.findall(".//{*}annotation"):
            if (ann.get("type") or "").lower() == "truth":
                label = (ann.text or "").strip()
        tvs = tg.findall(".//{*}traceView")
        if not label or not tvs:
            continue
        xs, ys = [], []
        for tv in tvs:
            ref = tv.get("traceDataRef") or tv.get("traceViewRef") or tv.get("traceRef")
            if ref and ref in traces:
                for (x, y) in traces[ref]:
                    xs.append(x); ys.append(y)
        if xs and ys:
            out.append({"label": label, "bbox": [int(min(ys)), int(min(xs)), int(max(ys)), int(max(xs))]})

    # If we found symbols, return them
    if out:
        return out

    # Otherwise: Whole-expression fallback
    global_label = None
    for ann in root.findall(".//{*}annotation"):
        if (ann.get("type") or "").lower() in {"truth", "latex"}:
            global_label = (ann.text or "").strip()
            break

    # Gather all points from all traces
    xs, ys = [], []
    for pts in traces.values():
        for (x, y) in pts:
            xs.append(x); ys.append(y)

    if xs and ys:
        return [{"label": global_label or "EXPR",
                 "bbox": [int(min(ys)), int(min(xs)), int(max(ys)), int(max(xs))]}]
    else:
        return []


# -------------------------
# Rendering
# -------------------------
IMG_SIZE = 256
MARGIN   = 8   # pixels around content
STROKE_W = 2

def _collect_all_points(traces: Dict[str, List[Tuple[float,float]]]) -> Tuple[List[float], List[float]]:
    xs, ys = [], []
    for pts in traces.values():
        for (x, y) in pts:
            xs.append(x); ys.append(y)
    return xs, ys

def _render_inkml_to_png(inkml_path: Path, out_png: Path) -> bool:
    """
    Renders the whole expression (all traces) to a 256x256 PNG.
    Returns True if something was drawn, else False.
    """
    root = ET.parse(inkml_path).getroot()
    traces = _parse_traces(root)
    if not traces:
        return False

    xs, ys = _collect_all_points(traces)
    if not xs or not ys:
        return False

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    w = maxx - minx
    h = maxy - miny
    if w <= 0 or h <= 0:
        return False

    # scale to fit into (IMG_SIZE - 2*MARGIN)^2
    avail = IMG_SIZE - 2 * MARGIN
    s = min(avail / w, avail / h)
    # center within canvas
    offx = (IMG_SIZE - s * w) / 2.0
    offy = (IMG_SIZE - s * h) / 2.0

    def to_px(p):
        x, y = p
        X = int(round(offx + (x - minx) * s))
        Y = int(round(offy + (y - miny) * s))
        return X, Y

    img = Image.new("L", (IMG_SIZE, IMG_SIZE), 255)
    drw = ImageDraw.Draw(img)

    for pts in traces.values():
        if len(pts) == 1:
            x, y = to_px(pts[0])
            drw.ellipse([x-1, y-1, x+1, y+1], fill=0)
        else:
            px = [to_px(p) for p in pts]
            drw.line(px, fill=0, width=STROKE_W, joint="curve")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    return True

# -------------------------
# I/O helpers
# -------------------------
def _read_list(path: Path) -> List[str]:
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            # allow "rel\tlabel" or just "rel"
            rel = s.split("\t", 1)[0].split()[0]
            lines.append(rel)
    return lines

def _process_split(inkml_root: Path, list_path: Path, out_dir: Path,
                   collect_labels: bool = False) -> Dict[str, List[Dict]]:
    """
    Render all files in 'list_path' into 'out_dir' (as PNGs).
    If collect_labels=True, also extract per-symbol labels and return a dict:
        { "<png_name>": [ {"label":..., "bbox":[y1,x1,y2,x2]}, ... ], ... }
    """
    rels = _read_list(list_path)
    labels_map: Dict[str, List[Dict]] = {}
    rendered = 0
    with_syms = 0
    total_syms = 0

    for rel in rels:
        inkml_path = inkml_root / rel
        if not inkml_path.exists():
            print(f"[warn] missing InkML: {inkml_path}")
            continue
        png_name = Path(rel).with_suffix(".png").name
        out_png = out_dir / png_name

        ok = _render_inkml_to_png(inkml_path, out_png)
        if ok:
            rendered += 1

            if collect_labels:
                try:
                    syms = parse_crohme_symbols_or_whole(inkml_path)
                except Exception:
                    syms = []
                if syms:
                    labels_map[png_name] = syms
                    with_syms += 1
                    total_syms += len(syms)

    print(f"[ok] rendered {rendered} images → {out_dir}")
    if collect_labels:
        print(f"[ok] labels: images_with_symbols={with_syms}  total_symbols={total_syms}")
    return labels_map

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inkml_root", required=True, help="Parent directory of 'crohme2019'")
    ap.add_argument("--train_list", required=True)
    ap.add_argument("--valid_list", required=True)
    ap.add_argument("--test_list",  required=True)
    ap.add_argument("--out_root",   required=True)
    ap.add_argument("--write_train_labels", action="store_true")
    args = ap.parse_args()

    inkml_root = Path(args.inkml_root)
    out_root   = Path(args.out_root)

    # Output dirs
    train_pngs = out_root / "train" / "pngs"
    valid_pngs = out_root / "valid" / "pngs"
    test_pngs  = out_root / "test" / "pngs"
    train_pngs.mkdir(parents=True, exist_ok=True)
    valid_pngs.mkdir(parents=True, exist_ok=True)
    test_pngs.mkdir(parents=True, exist_ok=True)

    # Render
    labels_map = _process_split(inkml_root, Path(args.train_list), train_pngs,
                                collect_labels=args.write_train_labels)
    _process_split(inkml_root, Path(args.valid_list), valid_pngs, collect_labels=False)
    _process_split(inkml_root, Path(args.test_list),  test_pngs,  collect_labels=False)

    # Save labels.json if requested
    if args.write_train_labels:
        labels_path = out_root / "train" / "labels.json"
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels_map, f, indent=2)
        print(f"[ok] wrote {labels_path}  images_with_symbols={len(labels_map)} "
              f" total_symbols={sum(len(v) for v in labels_map.values())}")

if __name__ == "__main__":
    main()
