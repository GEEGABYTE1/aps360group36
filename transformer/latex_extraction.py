import os
import xml.etree.ElementTree as ET
import re

def tokenize_latex(s):
    # Remove $ if present
    s = s.replace('$', '')
    # Regex: LaTeX commands, braces, symbols, single digits, letters
    pattern = r'(\\[a-zA-Z]+|[{}^_=+\-*/\[\](),]|[0-9]|[a-zA-Z])'
    tokens = re.findall(pattern, s)
    return tokens

def extract_latex_from_inkml(inkml_path):
    try:
        tree = ET.parse(inkml_path)
        root = tree.getroot()
        for ann in root.findall('.//{http://www.w3.org/2003/InkML}annotation'):
            if ann.attrib.get('type') == 'truth':
                return ann.text
    except ET.ParseError as e:
        print(f"Parse error in {inkml_path}: {e}")
    except Exception as e:
        print(f"Error in {inkml_path}: {e}")
    return None

def build_vocab(inkml_dir):
    vocab = set()
    for root, _, files in os.walk(inkml_dir):
        for fname in files:
            if fname.endswith('.inkml'):
                path = os.path.join(root, fname)
                latex = extract_latex_from_inkml(path)
                if latex:
                    tokens = tokenize_latex(latex)
                    vocab.update(tokens)
    # Add special tokens
    special = ["<pad>", "<bos>", "<eos>", "<unk>"]
    vocab = special + sorted(vocab)
    return vocab

if __name__ == "__main__":
    inkml_dir = "transformer/data/crohme2019"  # adjust as needed
    vocab = build_vocab(inkml_dir)
    # Save to file
    with open("transformer/data/vocab.txt", "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")
    print(f"Vocab size: {len(vocab)}")
""" 
# Example usage:
inkml_file = 'transformer/data/crohme2019/test/UN19wb_1121_em_1194.inkml'
latex = extract_latex_from_inkml(inkml_file)
print(latex)  # Should print something like: $\frac{149}{84}$
print(tokenize_latex(latex))  # Should print tokens like ['\\frac', '{', '149', '}', '{', '84', '}']

inkml_dir = 'transformer/data/crohme2019/'

arr = os.walk(inkml_dir)
for root, _, files in arr:
    print(root, _, files)  """