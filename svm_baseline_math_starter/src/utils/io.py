import json, os
from skimage.io import imread

def load_image(path):
    return imread(path, as_gray=True)

def read_json(path):
    with open(path, "r") as f: return json.load(f)

def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)
