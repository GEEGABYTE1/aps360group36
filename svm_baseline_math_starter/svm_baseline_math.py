import json, sys, re
from pathlib import Path

RELS = {"Right","NoRel","Sup","Sub","Inside","Above","Below"}

FUNC_LIMITS = {"\\sum","\\int","\\lim"}
ALIASES = {
    # normalize a few tokens that sometimes appear as words
    "COMMA": ",",
    "\\leq": "\\leq",
    "\\geq": "\\geq",
    "\\lt": "\\lt",
    "\\gt": "\\gt",
    "\\pm": "\\pm",
    "\\ldots": "\\ldots",
}

def to_latex_token(tok: str) -> str:
    # map aliases and leave LaTeX commands + symbols as-is
    if tok in ALIASES:
        return ALIASES[tok]
    return tok

def consume_sup_sub(seq, i, base):
    """Handle base Rel arg -> base^{arg} or base_{arg}."""
    rel = seq[i+1] if i+1 < len(seq) else None
    arg = seq[i+2] if i+2 < len(seq) else None
    if rel not in ("Sup","Sub") or arg is None:
        return to_latex_token(base), i+1
    arg = to_latex_token(arg)
    if rel == "Sup":
        out = f"{to_latex_token(base)}^{{{arg}}}"
    else:
        out = f"{to_latex_token(base)}_{{{arg}}}"
    # optionally skip a trailing relation tag if present (e.g., NoRel/Right)
    j = i+3
    if j < len(seq) and seq[j] in RELS:
        j += 1
    return out, j

def consume_limits(seq, i, op):
    """Handle \sum/\int/\lim with Below/Above (any order, optional both)."""
    lower = None
    upper = None
    j = i+1
    # try to read one or two (Below/Above) blocks
    for pass_k in range(2):
        if j+1 < len(seq) and seq[j] in ("Below","Above"):
            rel = seq[j]
            arg = to_latex_token(seq[j+1])
            if rel == "Below": lower = arg if lower is None else lower
            else: upper = arg if upper is None else upper
            j += 2
            if j < len(seq) and seq[j] in RELS:
                j += 1
        else:
            break
    piece = to_latex_token(op)
    if lower is not None: piece += f"_{{{lower}}}"
    if upper is not None: piece += f"^{{{upper}}}"
    return piece, j

def consume_sqrt(seq, i):
    """Handle \sqrt Inside arg -> \sqrt{arg}"""
    if i+2 >= len(seq):  # not enough room
        return to_latex_token(seq[i]), i+1
    if seq[i+1] != "Inside":
        return to_latex_token(seq[i]), i+1
    arg = to_latex_token(seq[i+2])
    j = i+3
    if j < len(seq) and seq[j] in RELS:
        j += 1
    return f"\\sqrt{{{arg}}}", j

def try_fraction_window(seq, i):
    """
    Heuristic: detect '- Above <num ...> ... - Below <den ...>' pattern
    starting at current position. Return (latex, new_index) or (None, i).
    """
    # require a '-' token followed by 'Above'
    if not (i < len(seq) and seq[i] == "-" and i+1 < len(seq) and seq[i+1] == "Above"):
        return None, i
    j = i + 2
    num_tokens = []
    # collect numerator until we hit another '-' + 'Below' or end/stop token
    while j < len(seq):
        if j+1 < len(seq) and seq[j] == "-" and seq[j+1] == "Below":
            j += 2
            break
        if seq[j] in RELS:
            # stop on a relation that clearly exits the fraction context
            if seq[j] in ("Right","NoRel"):
                j += 1
                continue
        num_tokens.append(to_latex_token(seq[j]))
        j += 1
    # collect denominator
    den_tokens = []
    while j < len(seq):
        if seq[j] in RELS and seq[j] in ("Right","NoRel"):
            j += 1
            # allow trailing Right/NoRel then stop fraction
            break
        den_tokens.append(to_latex_token(seq[j]))
        j += 1
    if not num_tokens or not den_tokens:
        return None, i
    num = " ".join(num_tokens).strip()
    den = " ".join(den_tokens).strip()
    return f"\\frac{{{num}}}{{{den}}}", j

def line_to_latex(rhs: str) -> str:
    # Tokenize by whitespace, keep order: sym REL sym REL ...
    parts = rhs.strip().split()
    out = []
    i = 0
    while i < len(parts):
        tok = parts[i]
        # FRACTION heuristic first (if a bar-like '-' starts an Above/Below pair)
        if tok == "-" and i+1 < len(parts) and parts[i+1] == "Above":
            frac, j = try_fraction_window(parts, i)
            if frac:
                out.append(frac); i = j; continue
        # LIMIT operators
        if tok in FUNC_LIMITS and i+1 < len(parts) and parts[i+1] in ("Below","Above"):
            piece, j = consume_limits(parts, i, tok)
            out.append(piece); i = j; continue
        # sqrt
        if tok == "\\sqrt" and i+1 < len(parts) and parts[i+1] == "Inside":
            piece, j = consume_sqrt(parts, i)
            out.append(piece); i = j; continue
        # Sup/Sub
        if i+1 < len(parts) and parts[i+1] in ("Sup","Sub"):
            piece, j = consume_sup_sub(parts, i, tok)
            out.append(piece); i = j; continue
        # Plain token; optionally skip a trailing relation tag
        out.append(to_latex_token(tok))
        i += 1
        if i < len(parts) and parts[i] in ("Right","NoRel"):
            i += 1
    # compact some spacing around braces/parentheses
    text = " ".join(out)
    text = re.sub(r"\s+([,)\]])", r"\\1", text)
    text = re.sub(r"([(\\[{])\s+", r"\\1", text)
    return text.strip()

def convert_file(list_path: Path, out_json: Path):
    mapping = {}
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:  # header lines like "inkml   w"
                continue
            path, rhs = line.split("\t", 1)
            img_name = Path(path).with_suffix(".png").name
            mapping[img_name] = line_to_latex(rhs)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"Wrote {out_json} with {len(mapping)} entries.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_crohme_to_latex.py <crohme_list.txt> <out.json>")
        sys.exit(1)
    convert_file(Path(sys.argv[1]), Path(sys.argv[2]))
