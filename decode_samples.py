"""Run the bundled sample plate crops through the region classifiers and CRNN OCR models.

Preprocessing (found empirically): 96x48 BGR uint8 -> float NCHW; CRNNs take /255,
the MobileNetV3 region classifiers take raw 0-255 (Keras built-in rescaling).
CRNN output (1, 18, C): greedy CTC, blank = last index.
37-class models decode with 0-9A-Z; other alphabets are unknown, so raw indices are shown.
"""
import os

import numpy as np
import MNN

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
ALNUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# classifier -> (id list, sample crop), from inspect_weights.py pairings
CLASSIFIERS = {
    "23": ("24", "25"), "26": ("27", "28"), "29": ("30", "31"), "32": ("33", "34"),
    "35": ("36", "37"), "38": ("39", "40"), "41": ("42", "43"), "44": ("45", "46"),
    "47": ("48", "49"), "82": ("83", "84"),
}
CRNNS = [f"{i}" for i in range(50, 70)]
TRUTH = {"25": "FHW7186", "28": "0907CDS", "31": "SH7194K", "34": "RTL015", "37": "343466",
         "40": "990SKR09", "43": "(Thai)8572", "46": "ASI927", "49": "C9998", "84": "3581KEA"}


def path(n):
    return os.path.join(WEIGHTS, f"rec_{n}.bin")


def load_crop(n, scale):
    a = np.frombuffer(open(path(n), "rb").read(), np.uint8).reshape(48, 96, 3)
    return np.ascontiguousarray((a.astype(np.float32) * scale).transpose(2, 0, 1)[None])


def softmax_if_logits(p):
    if np.allclose(p.sum(-1), 1, atol=1e-3):
        return p
    e = np.exp(p - p.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


class Net:
    def __init__(self, n):
        self.net = MNN.Interpreter(path(n))
        self.sess = self.net.createSession()
        self.inp = self.net.getSessionInput(self.sess)
        self.out = next(iter(self.net.getSessionOutputAll(self.sess).values()))

    def __call__(self, x):
        self.inp.copyFrom(MNN.Tensor(x.shape, MNN.Halide_Type_Float, x, MNN.Tensor_DimensionType_Caffe))
        self.net.runSession(self.sess)
        shape = self.out.getShape()
        host = MNN.Tensor(shape, MNN.Halide_Type_Float, np.zeros(shape, np.float32),
                          MNN.Tensor_DimensionType_Caffe)
        self.out.copyToHostTensor(host)
        return np.array(host.getData(), np.float32).reshape(shape)[0]


def ctc_greedy(probs):
    blank = probs.shape[-1] - 1
    seq, prev = [], None
    for k in probs.argmax(-1):
        if k != prev and k != blank:
            seq.append(int(k))
        prev = k
    conf = float(probs.max(-1).mean())
    return seq, conf


def main():
    raw = {s: load_crop(s, 1.0) for s in TRUTH}  # MobileNetV3 classifiers rescale internally
    crops = {s: load_crop(s, 1 / 255) for s in TRUTH}  # CRNNs expect [0, 1]

    print("REGION CLASSIFIERS (own sample crop, raw 0-255 input)")
    for m, (ids, sample) in CLASSIFIERS.items():
        lines = [ln for ln in open(path(ids), encoding="utf-8").read().splitlines() if ln.strip()]
        p = softmax_if_logits(Net(m)(raw[sample]))
        top = p.argsort()[::-1][:3]
        pretty = ", ".join(f"#{k} '{lines[k]}' {p[k]:.2f}" for k in top)
        print(f"  rec_{m} ({len(p)} cls) on rec_{sample} {TRUTH[sample]:<11} -> {pretty}")

    print("\nCRNN OCR (every model x every crop; best-confidence model per crop marked *)")
    results = {}
    for m in CRNNS:
        net = Net(m)
        for s, x in crops.items():
            probs = softmax_if_logits(net(x))
            results[m, s] = (*ctc_greedy(probs), probs.shape[-1])
    for s in TRUTH:
        best = max(CRNNS, key=lambda m: results[m, s][1])
        print(f"  rec_{s} truth={TRUTH[s]}")
        for m in CRNNS:
            seq, conf, n_cls = results[m, s]
            text = "".join(ALNUM[k] for k in seq) if n_cls == len(ALNUM) + 1 else "(alphabet unknown)"
            print(f"   {'*' if m == best else ' '}rec_{m} C={n_cls:<3} conf={conf:.2f} "
                  f"text={text:<18} idx={seq}")


if __name__ == "__main__":
    main()
