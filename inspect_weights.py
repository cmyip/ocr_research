"""Classify every rec_NN.bin in weights/ and check model <-> label/sample pairings.

Pairing rule: files are packed in SDK order, so a model's label files and sample
input are the non-model files that follow it, up to the next model.
A label file "matches" when its line count equals a model output's class dim.
"""
import os
import re
import struct
import sys

import numpy as np
import onnx
import MNN

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


def classify(data):
    if data == b"dummy":
        return "dummy"
    if b"pytorch" in data[:32] and data[:2] == b"\x08\x08":
        return "onnx"
    if b"MNN" in data and data[:4] in (b"\x20\x00\x00\x00", b"\x1c\x00\x00\x00"):
        return "mnn"
    if len(data) > 24 and struct.unpack("<3f", data[4:16]) == (3.0, 112.0, 96.0):
        return "floatblob"
    if decode_text(data) is not None:
        return "text"
    return "raw"


def decode_text(data):
    for enc in ("utf-8", "cp1252"):
        try:
            text = data.decode(enc)
        except UnicodeDecodeError:
            continue
        if all(c.isprintable() or c in "\r\n\t" for c in text):
            return text
    # mostly-UTF-8 files with a few malformed bytes (e.g. rec_01)
    text = data.decode("utf-8", errors="replace")
    bad = sum(not (c.isprintable() or c in "\r\n\t") or c == "�" for c in text)
    return text if bad < len(text) * 0.001 else None


def mnn_io(path):
    net = MNN.Interpreter(path)
    sess = net.createSession()
    ins = {k: tuple(v.getShape()) for k, v in net.getSessionInputAll(sess).items()}
    outs = {k: tuple(v.getShape()) for k, v in net.getSessionOutputAll(sess).items()}
    return ins, outs


def onnx_io(path):
    g = onnx.load(path, load_external_data=False).graph
    dims = lambda v: tuple(d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim)
    return {v.name: dims(v) for v in g.input}, {v.name: dims(v) for v in g.output}


def class_dims(kind, outs):
    """Candidate class counts per output head."""
    res = {}
    for name, shape in outs.items():
        if kind == "onnx" and len(shape) == 3 and isinstance(shape[1], int):
            res[name] = shape[1] - 4  # YOLOv8-style (1, 4 + nc, anchors)
        elif shape:
            res[name] = shape[-1]
    return res


def label_count(text):
    return len([ln for ln in text.splitlines() if ln.strip()])


def main():
    files = sorted(f for f in os.listdir(WEIGHTS) if re.fullmatch(r"rec_\d+\.bin", f))
    info = []
    for f in files:
        path = os.path.join(WEIGHTS, f)
        data = open(path, "rb").read()
        kind = classify(data)
        entry = {"file": f, "kind": kind, "size": len(data)}
        if kind in ("mnn", "onnx"):
            try:
                entry["ins"], entry["outs"] = (mnn_io if kind == "mnn" else onnx_io)(path)
            except Exception as e:  # keep going; report the failure
                entry["error"] = str(e).splitlines()[0]
        elif kind == "floatblob":
            entry["header"] = [int(x) for x in struct.unpack("<6f", data[:24])]
        elif kind == "text":
            text = decode_text(data)
            entry["lines"] = label_count(text)
            entry["preview"] = " | ".join(text.splitlines()[:3])[:60]
        info.append(entry)

    print(f"{'file':<11}{'kind':<10}{'size':>10}  details")
    print("-" * 100)
    for e in info:
        if "error" in e:
            d = f"LOAD ERROR: {e['error']}"
        elif "ins" in e:
            d = f"in={list(e['ins'].values())} out={list(e['outs'].values())}"
        elif "header" in e:
            h = e["header"]
            d = f"header={h}  -> alphabet N={h[5]}"
        elif "lines" in e:
            d = f"{e['lines']} lines: {e['preview']}"
        else:
            d = ""
        print(f"{e['file']:<11}{e['kind']:<10}{e['size']:>10}  {d}")

    print("\nPAIRINGS (model -> following files until next model)")
    print("-" * 100)
    for i, e in enumerate(info):
        if "outs" not in e:
            continue
        followers = []
        for f in info[i + 1:]:
            if f["kind"] in ("mnn", "onnx", "floatblob"):
                break
            followers.append(f)
        heads = class_dims(e["kind"], e["outs"])
        in_shape = next(iter(e["ins"].values()))
        hw = [d for d in in_shape if isinstance(d, int)][-2:]
        in_bytes = hw[0] * hw[1] * 3 if len(hw) == 2 else None
        print(f"{e['file']}  input={in_shape}")
        for head, n in heads.items():
            hits = [f["file"] for f in followers if f.get("lines") == n]
            status = f"MATCH {hits}" if hits else "no label file with that count"
            print(f"    head {head[:55]:<55} classes={n:<6} {status}")
        for f in followers:
            if f["kind"] == "raw":
                ok = "MATCH input" if f["size"] == in_bytes else f"size != {in_bytes}"
                print(f"    sample {f['file']} ({f['size']} B) {ok}")
            elif f["kind"] == "text" and not any(f["lines"] == n for n in heads.values()):
                print(f"    text   {f['file']} ({f['lines']} lines) unmatched")


if __name__ == "__main__":
    sys.exit(main())
