"""Fine-tune one of the SDK's CRNN OCR models on new plates and write it back as a drop-in rec_NN.bin.

    python finetune/train.py --real sample_data --out weights_ft

What it trains on
  * real plates: every image in --real folders named after its plate (UITM1776.jpeg, WXY123_2.jpg), cropped
    by the pipeline itself with jittered geometry. --holdout keeps matching files out of training and reports
    them separately, to check that a fix generalises rather than memorises.
  * synthetic plates (dataset.Synth), which is where the letters and series the stock model never emits come
    from when there are only a handful of real photos.

How it avoids forgetting: the stock model is kept as a frozen teacher. Wherever the teacher already reads a
training crop correctly the student is also pulled towards the teacher's per-step output, so behaviour that
was right stays the same and only the wrong reads are re-learned through the CTC loss.

Output: OUT/rec_NN.bin plus links to every other weight file, so OUT works as a --weights directory:
    lpr-api/target/release/lpr-api read --weights weights_ft sample_data/*.jpeg
"""
import argparse
import copy
import fnmatch
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import ALNUM, ROOT, Rectifier, Synth, augment_real, load_real, to_tensor  # noqa: E402
from mnn_crnn import MnnCrnn, decode, mnn_run  # noqa: E402

# the SDK's own 96x48 BGR sample crops with 0-9A-Z text: a check that other countries' plates still read
SDK_SAMPLES = {"25": "FHW7186", "31": "SH7194K", "34": "RTL015", "46": "ASI927"}


def read_batch(model, xs):
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.from_numpy(np.stack(xs)).to(device)).cpu()
    return [decode(p) for p in logits]


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def score(model, items, batch=256):
    """items: [(label, tensor)]. Returns (exact-match %, character accuracy %, reads)."""
    reads = []
    for i in range(0, len(items), batch):
        reads += read_batch(model, [x for _, x in items[i:i + batch]])
    exact = sum(r == t for r, (t, _) in zip(reads, items))
    edits = sum(min(edit_distance(r, t), len(t)) for r, (t, _) in zip(reads, items))
    chars = sum(len(t) for t, _ in items)
    return 100 * exact / max(len(items), 1), 100 * (1 - edits / max(chars, 1)), reads


def synth_set(n, seed, exclude=()):
    s = Synth(seed)
    return [(t, to_tensor(img)) for t, img in (s.sample(exclude) for _ in range(n))]


def report(model, sets, log):
    out = {}
    for name, items in sets.items():
        if items:
            exact, char, reads = score(model, items)
            out[name] = exact
            log(f"    {name:<28} n={len(items):<5} exact {exact:5.1f}%   chars {char:5.1f}%")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(ROOT, "weights", "rec_57.bin"), help="CRNN to fine-tune")
    ap.add_argument("--real", action="append", default=[], help="folder of images named after their plate (repeatable)")
    ap.add_argument("--holdout", action="append", default=[], help="plate pattern of real images to test on only, e.g. 'UITM*'")
    ap.add_argument("--exclude-prefix", action="append", default=[], help="plate prefix the synthetic data must not contain")
    ap.add_argument("--out", default=os.path.join(ROOT, "weights_ft"), help="weights directory to write")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--backbone-lr-scale", type=float, default=0.3, help="backbone learning rate relative to the head (0 freezes it)")
    ap.add_argument("--real-frac", type=float, default=0.25, help="share of each batch drawn from the real plates")
    ap.add_argument("--kd", type=float, default=1.0, help="weight of the stay-close-to-the-teacher term")
    ap.add_argument("--val", type=int, default=3000, help="size of the synthetic validation set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
                    help="where the network runs (the CTC loss itself always runs on the CPU)")
    ap.add_argument("--eval-only", action="store_true", help="only score --model on the validation sets")
    args = ap.parse_args()
    log = lambda *a: print(*a, flush=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    student = MnnCrnn(args.model)
    if student.classes != len(ALNUM) + 1:
        sys.exit(f"{args.model} has {student.classes} classes; this script trains the 0-9A-Z (37-class) models")
    student.to(args.device)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)

    # ---- data
    rect = Rectifier()
    real = [item for folder in args.real for item in load_real(folder, rect, log)]
    held = lambda label: any(fnmatch.fnmatch(label, p.upper()) for p in args.holdout)
    held_real = [it for it in real if held(it[0])]
    train_real = [it for it in real if not held(it[0])]
    log(f"real plates: {len(train_real)} to train on {[t for t, _, _ in train_real]}, "
        f"{len(held_real)} held out {[t for t, _, _ in held_real]}")

    val = synth_set(args.val, 10_000 + args.seed, args.exclude_prefix)
    has_io = lambda t: "I" in t or "O" in t
    sets = {
        "synthetic, no I or O": [v for v in val if not has_io(v[0])],
        "synthetic, with I or O": [v for v in val if has_io(v[0])],
        "synthetic, with 11": [v for v in val if "11" in v[0]],
        "real, trained on (clean crop)": [(t, to_tensor(rect.crop(img, box))) for t, img, box in train_real],
        "real, held out (clean crop)": [(t, to_tensor(rect.crop(img, box))) for t, img, box in held_real],
        "real, held out (jittered x50)": [(t, to_tensor(rect.crop(img, box, np.random.default_rng(i))))
                                          for t, img, box in held_real for i in range(50)],
        "SDK samples, other countries": [
            (t, np.ascontiguousarray(np.frombuffer(open(os.path.join(ROOT, "weights", f"rec_{n}.bin"), "rb").read(), np.uint8)
                                     .reshape(48, 96, 3).astype(np.float32).transpose(2, 0, 1) / 255))
            for n, t in SDK_SAMPLES.items()],
    }
    log(f"{os.path.basename(args.model)} before training:")
    before = report(teacher, sets, log)
    for name in ("real, trained on (clean crop)", "real, held out (clean crop)"):
        if sets[name]:
            log(f"    {name}: " + ", ".join(f"{t}->{r}" for (t, _), r in zip(sets[name], score(teacher, sets[name])[2])))
    if args.eval_only:
        return

    # ---- training
    synth = Synth(args.seed)
    head, backbone = student.head_parameters(), student.backbone_parameters()
    groups = [{"params": head, "lr": args.lr}]
    if args.backbone_lr_scale > 0:
        groups.append({"params": backbone, "lr": args.lr * args.backbone_lr_scale})
    else:
        for p in backbone:
            p.requires_grad_(False)
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=args.steps, pct_start=0.1)
    blank = student.classes - 1
    n_real = round(args.batch * args.real_frac) if train_real else 0
    t0 = time.time()
    student.train()
    for step in range(1, args.steps + 1):
        labels, xs = [], []
        for _ in range(n_real):
            t, img, box = train_real[rng.integers(len(train_real))]
            labels.append(t)
            xs.append(to_tensor(augment_real(rect.crop(img, box, rng), rng)))
        while len(xs) < args.batch:
            t, img = synth.sample(args.exclude_prefix)
            labels.append(t)
            xs.append(to_tensor(img))
        x = torch.from_numpy(np.stack(xs)).to(args.device)
        logits = student(x)  # (N, 18, C)
        logp = F.log_softmax(logits, -1)
        targets = torch.tensor([ALNUM.index(c) for t in labels for c in t])
        lengths = torch.tensor([len(t) for t in labels])
        ctc = F.ctc_loss(logp.transpose(0, 1).cpu(), targets, torch.full((len(labels),), logp.shape[1]), lengths,
                         blank=blank, zero_infinity=True)
        with torch.no_grad():
            t_logp = F.log_softmax(teacher(x), -1)
            keep = torch.tensor([decode(p) == t for p, t in zip(t_logp.cpu(), labels)], device=args.device)
        kd = (F.kl_div(logp[keep], t_logp[keep], log_target=True, reduction="sum") / max(int(keep.sum()), 1) / logp.shape[1]
              if keep.any() else logits.sum() * 0)
        loss = ctc.to(args.device) + args.kd * kd
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == args.steps:
            log(f"step {step:5d}/{args.steps}  ctc {ctc.item():.4f}  kd {kd.item():.4f}  "
                f"teacher right on {100 * keep.float().mean():.0f}% of batch  {time.time() - t0:.0f}s")
        if step % 1000 == 0 and step != args.steps:
            student.eval()
            report(student, sets, log)
            student.train()

    # ---- export, then score the file that was written (fp16 rounding included) rather than the float weights
    student.eval()
    os.makedirs(args.out, exist_ok=True)
    name = os.path.basename(args.model)
    src_dir = os.path.dirname(os.path.abspath(args.model))
    for f in sorted(os.listdir(src_dir)):
        dst = os.path.join(args.out, f)
        if f != name and not os.path.lexists(dst):
            os.symlink(os.path.join(src_dir, f), dst)
    out_path = os.path.join(args.out, name)
    if os.path.islink(out_path):
        os.remove(out_path)
    student.export(out_path)
    exported = MnnCrnn(out_path).eval().to(args.device)
    log(f"\nwrote {out_path}\nfine-tuned model (as exported):")
    after = report(exported, sets, log)
    for name_ in ("real, trained on (clean crop)", "real, held out (clean crop)"):
        if sets[name_]:
            log(f"    {name_}: " + ", ".join(f"{t}->{r}" for (t, _), r in zip(sets[name_], score(exported, sets[name_])[2])))
    # the MNN runtime must agree with the PyTorch copy of the exported file
    probe = [x for items in sets.values() for _, x in items[:20]]
    agree = sum(decode(torch.from_numpy(mnn_run(out_path, x[None]))) == r for x, r in zip(probe, read_batch(exported.cpu(), probe)))
    log(f"MNN runtime agrees with PyTorch on {agree}/{len(probe)} crops")
    log("change in exact match: " + ", ".join(f"{k}: {before[k]:.1f} -> {after[k]:.1f}" for k in after))


if __name__ == "__main__":
    main()
