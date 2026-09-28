"""Read a licence plate from an image using the extracted SDK models.

Pipeline:
  1. rec_72.bin  YOLO (ONNX, 320x320) -> plate box
  2. crop + resize to 96x48 BGR
  3. rec_50..69  CRNN OCR (MNN) -> greedy CTC over 0-9A-Z. The region -> OCR model
     mapping is unknown, so by default the most confident 37-class model wins;
     pin one with --ocr-model when you know it (rec_57 read the Malaysian sample).

Usage: python read_plate.py IMAGE [--ocr-model 57] [--deshear 0.4] [--runs N] [--gpu]
  --gpu needs the CUDA 12 onnxruntime-gpu in .venv-gpu (driver 573 supports CUDA <= 12.8)
"""
import argparse
import os
import time

import numpy as np
import MNN
import onnxruntime as ort
from PIL import Image

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
ALNUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DET_SIZE = 320
PLATE_W, PLATE_H = 96, 48
CROP_PAD = 0.02  # fractional padding around the detected box; OCR is sensitive to this


def path(n):
    return os.path.join(WEIGHTS, f"rec_{n}.bin")


class Detector:
    def __init__(self, gpu=False):
        self.gpu = gpu
        if not gpu:
            self.sess = ort.InferenceSession(path("72"), providers=["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name
            return
        # needs onnxruntime-gpu built for the driver's CUDA version (see .venv-gpu)
        ort.preload_dlls()
        opts = {"cudnn_conv_algo_search": "EXHAUSTIVE", "enable_cuda_graph": "1"}
        self.sess = ort.InferenceSession(path("72"), providers=[("CUDAExecutionProvider", opts)])
        # CUDA graphs replay fixed device buffers, so bind input/output once and update in place
        self.x_dev = ort.OrtValue.ortvalue_from_numpy(
            np.zeros((1, 3, DET_SIZE, DET_SIZE), np.float32), "cuda", 0)
        self.binding = self.sess.io_binding()
        self.binding.bind_ortvalue_input(self.sess.get_inputs()[0].name, self.x_dev)
        for o in self.sess.get_outputs():
            self.binding.bind_output(o.name, "cuda", 0)
        self.sess.run_with_iobinding(self.binding)  # capture the graph; fails loudly if CUDA is unusable

    def infer(self, x):
        if not self.gpu:
            return self.sess.run(None, {self.inp: x})[0]
        self.x_dev.update_inplace(x)
        self.sess.run_with_iobinding(self.binding)
        return self.binding.copy_outputs_to_cpu()[0]

    def __call__(self, img):
        """img: PIL RGB. Returns (x1, y1, x2, y2, score) of the best plate, or None."""
        w, h = img.size
        r = DET_SIZE / max(w, h)
        nw, nh = round(w * r), round(h * r)
        # reducing_gap does most of the downscale with a fast integer box pass (~2x faster)
        small = np.asarray(img.resize((nw, nh), Image.BILINEAR, reducing_gap=2.0))
        canvas = np.full((DET_SIZE, DET_SIZE, 3), 114, np.uint8)
        canvas[:nh, :nw] = small
        x = np.empty((1, 3, DET_SIZE, DET_SIZE), np.float32)
        np.multiply(canvas.transpose(2, 0, 1), 1 / 255, out=x[0], casting="unsafe")
        t = time.perf_counter()
        out = self.infer(x)[0].T  # (anchors, 4 + nc)
        self.last_infer_ms = (time.perf_counter() - t) * 1e3
        best = out[:, 4].argmax()  # class 0 = plate
        cx, cy, bw, bh, score = out[best, :5]
        if score < 0.25:
            return None
        return ((cx - bw / 2) / r, (cy - bh / 2) / r, (cx + bw / 2) / r, (cy + bh / 2) / r, float(score))


class Crnn:
    def __init__(self, n):
        self.name = f"rec_{n}"
        self.net = MNN.Interpreter(path(n))
        self.sess = self.net.createSession()
        self.inp = self.net.getSessionInput(self.sess)
        self.out = next(iter(self.net.getSessionOutputAll(self.sess).values()))
        self.classes = self.out.getShape()[-1]

    def __call__(self, x):
        self.inp.copyFrom(MNN.Tensor(x.shape, MNN.Halide_Type_Float, x, MNN.Tensor_DimensionType_Caffe))
        self.net.runSession(self.sess)
        shape = self.out.getShape()
        host = MNN.Tensor(shape, MNN.Halide_Type_Float, np.zeros(shape, np.float32),
                          MNN.Tensor_DimensionType_Caffe)
        self.out.copyToHostTensor(host)
        p = np.array(host.getData(), np.float32).reshape(shape)[0]
        if not np.allclose(p.sum(-1), 1, atol=1e-3):  # some models emit logits
            p = np.exp(p - p.max(-1, keepdims=True))
            p /= p.sum(-1, keepdims=True)
        return decode(p)


def decode(p):
    blank = p.shape[-1] - 1
    chars, prev = [], None
    for k in p.argmax(-1):
        if k != prev and k != blank:
            chars.append(ALNUM[k])
        prev = k
    return "".join(chars), float(p.max(-1).mean())


def load_models(ocr_model=None, gpu=False):
    """ocr_model: e.g. "57" to use one CRNN; None tries every 0-9A-Z model and keeps the most confident.
    gpu: run the detector on CUDA (OCR stays on CPU: the pip MNN build has no CUDA backend)."""
    if ocr_model:
        ocr = [Crnn(ocr_model)]
    else:
        ocr = [m for m in (Crnn(n) for n in range(50, 70)) if m.classes == len(ALNUM) + 1]
    return Detector(gpu), ocr


def crop_plate(img, box, deshear=0.0):
    """deshear > 0 straightens right-leaning (italic) plate fonts, e.g. ~0.4 for Malaysian plates."""
    x1, y1, x2, y2 = box[:4]
    px, py = (x2 - x1) * CROP_PAD, (y2 - y1) * CROP_PAD
    if deshear:
        # shear a larger region so no edge pixels are pulled in from outside, then trim
        mx, my = (x2 - x1) * 0.1, (y2 - y1) * 0.3
        big = img.crop((x1 - mx, y1 - my, x2 + mx, y2 + my))
        big = big.transform(big.size, Image.AFFINE, (1, -deshear, deshear * big.height / 2, 0, 1, 0),
                            Image.BILINEAR)
        img, x1, y1, x2, y2 = big, mx, my, big.width - mx, big.height - my
    crop = img.crop((max(0, x1 - px), max(0, y1 - py), min(img.width, x2 + px), min(img.height, y2 + py)))
    a = np.asarray(crop.resize((PLATE_W, PLATE_H), Image.BILINEAR), np.float32)[..., ::-1]  # RGB -> BGR
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None] / 255), crop


def read_plate(img, detector, ocr_models, deshear=0.0):
    """Returns (text, confidence, info dict with timings in ms)."""
    t0 = time.perf_counter()
    box = detector(img)
    t1 = time.perf_counter()
    if box is None:
        return None, 0.0, {"detect_ms": (t1 - t0) * 1e3}
    x, crop = crop_plate(img, box, deshear)
    t2 = time.perf_counter()
    reads = [(*m(x), m.name) for m in ocr_models]
    t3 = time.perf_counter()
    text, conf, model = max(reads, key=lambda r: r[1])
    return text, conf, {
        "box": [round(v) for v in box[:4]], "det_score": box[4], "ocr_model": model, "crop": crop,
        "all_reads": sorted(reads, key=lambda r: -r[1]),
        "detect_ms": (t1 - t0) * 1e3, "det_infer_ms": detector.last_infer_ms,
        "det_prep_ms": (t1 - t0) * 1e3 - detector.last_infer_ms,
        "crop_ms": (t2 - t1) * 1e3, "ocr_ms": (t3 - t2) * 1e3,
        "total_ms": (t3 - t0) * 1e3,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--runs", type=int, default=20, help="timed runs after one warm-up")
    ap.add_argument("--save-crop", help="write the plate crop here")
    ap.add_argument("--ocr-model", help='CRNN to use, e.g. "57" (default: most confident 0-9A-Z model)')
    ap.add_argument("--deshear", type=float, default=0.0, help="italic-font correction, ~0.4 for Malaysia")
    ap.add_argument("--gpu", action="store_true", help="run the detector on CUDA (use .venv-gpu)")
    args = ap.parse_args()

    t = time.perf_counter()
    detector, ocr = load_models(args.ocr_model, args.gpu)
    load_ms = (time.perf_counter() - t) * 1e3

    t = time.perf_counter()
    img = Image.open(args.image).convert("RGB")
    decode_ms = (time.perf_counter() - t) * 1e3

    text, conf, info = read_plate(img, detector, ocr, args.deshear)  # warm-up
    if text is None:
        print("no plate found")
        return
    if args.save_crop:
        info["crop"].save(args.save_crop)
    runs = [read_plate(img, detector, ocr, args.deshear)[2] for _ in range(args.runs)]

    print(f"plate: {text}  (conf {conf:.3f}, model {info['ocr_model']}, "
          f"box {info['box']}, det {info['det_score']:.2f})")
    if len(ocr) > 1:
        print("other reads:", ", ".join(f"{m}={s}({c:.2f})" for s, c, m in info["all_reads"][1:5]))
    print(f"\nmodel load {load_ms:.0f} ms (one-off), image decode {decode_ms:.1f} ms")
    print(f"  detector on {detector.sess.get_providers()[0]}")
    for k in ("detect_ms", "det_prep_ms", "det_infer_ms", "crop_ms", "ocr_ms", "total_ms"):
        v = np.array([r[k] for r in runs])
        print(f"  {k:<10} median {np.median(v):7.1f} ms   min {v.min():7.1f}   max {v.max():7.1f}")
    print(f"  ({len(ocr)} OCR models per plate; ~{np.median([r['ocr_ms'] for r in runs]) / len(ocr):.1f} ms each)")


if __name__ == "__main__":
    main()
