# finetune — teach the SDK's OCR model new plates

Transfer-learns one of the CRNN OCR models (`weights/rec_50..69.bin`) on new plates and writes it back as a
drop-in `rec_NN.bin`. Nothing in the Android app or `lpr-api` changes: point them at the new weights directory.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r finetune/requirements.txt

# put photos of the plates that read wrong in a folder, each named after its plate (UITM1776.jpeg, WXY123_2.jpg)
.venv/bin/python finetune/train.py --real sample_data --out weights_ft

lpr-api/target/release/lpr-api read --weights weights_ft sample_data/*.jpeg
```

About 17 minutes for the default 3000 steps on an Apple-silicon laptop (MPS); CUDA is used when present.

## Why UITM1776 read as UTM1776

The detector, the corner rectifier and the crop are fine: the 96×48 crop given to the OCR shows `UITM 1776`
clearly, and no padding, margin or shear setting changes the read. The CRNN itself gives the letter `I` a
probability of 0.0000 at all 18 steps.

rec_57 was evidently trained on the ordinary Malaysian/Singaporean series only, which never uses `I` or `O`.
On rendered test plates (`XXX 1234` with one letter forced, 150 per letter) the stock model reads every letter
at 68–89 % except:

| letter | read correctly | what happens instead |
|---|---|---|
| `I` | 33 % | dropped, or read as `1` |
| `O` | 26 % | read as `Q` or `0` |

So any special series with those letters (`UITM`, `IIUM`, `PROTON`, `PERODUA`, `UNITEN`, `SUKOM`, …) is at risk,
not just this one plate. It is a gap in what the weights were taught, which is why it takes training to fix.

Two things about the architecture are worth knowing, though neither is the cause here:

- The "18 time steps" are a 3×6 grid of backbone features (16 px per column) that a BiLSTM turns into a
  sequence. That is what lets it read two-row plates, and it caps a plate at 18 characters.
- The BiLSTM carries a strong prior over plate formats, so an unusual format can fail even when every glyph
  is one the model knows.

## What the training does

- `mnn_crnn.py` reads the MNN flatbuffer directly and rebuilds the network in PyTorch (MobileNetV3 backbone →
  3×6×128 features → BiLSTM(64) → Dense). It matches the MNN runtime to about 4e-5. `export` overwrites the
  weight bytes in a copy of the original file, so the graph, tensor names and file size stay identical.
  14 of the 20 CRNNs load this way; rec_51, 52, 54, 61, 62 and 65 have a different head and are refused.
- `dataset.py` makes the crops. Real photos go through the pipeline's own detector → corners → warp with the
  geometry jittered. Synthetic plates are rendered from system fonts in the Malaysian formats — ordinary
  series, special series, and letters with `I`/`O` as likely as the rest — then blurred, down-sampled, noised
  and JPEG-compressed like a CCTV frame. `python finetune/dataset.py sheet.png` writes a contact sheet.
- `train.py` fine-tunes with the CTC loss. The stock model stays as a frozen teacher: wherever it already
  reads a training crop correctly the student is also held to the teacher's output, so what was right stays
  right and only the wrong reads are re-learned. It scores the stock model first, then the *exported file*
  (fp16 rounding included), and checks that the MNN runtime agrees with PyTorch on it.

## Results on rec_57

Held-out test — `--holdout 'UITM*' --exclude-prefix UITM`, so neither the photo nor any synthetic `UITM` plate
was trained on:

| set | stock | fine-tuned |
|---|---|---|
| `UITM1776.jpeg`, pipeline crop | `UTM1776` | **`UITM1776`** |
| `UITM1776.jpeg`, 50 jittered crops | 0 % | 72 % |
| synthetic plates with `I` or `O` (814) | 12.8 % | 49.3 % |
| synthetic plates without (2186) | 72.2 % | 82.4 % |
| the other three sample photos | 3 / 3 | 3 / 3 |
| SDK sample crops from other countries | 4 / 4 | 4 / 4 |

The first model — `--real sample_data` when it held four photos — before the 1/I work below:

| set | stock | fine-tuned |
|---|---|---|
| four sample photos through `lpr-api read` | 3 / 4 | 4 / 4 (all trained on) |
| letter `I` on rendered `XXX 1234` plates | 33 % | 85 % |
| letter `O` | 26 % | 87 % |
| the other 24 letters, mean (worst) | 81 % (68 %) | 89 % (79 %) |
| synthetic plates with `I` or `O` (823) | 12.5 % | 49.3 % |
| synthetic plates without (2177) | 72.4 % | 82.4 % |
| SDK sample crops from other countries | 4 / 4 | 4 / 4 |

The synthetic sets are deliberately harsh (heavy blur, plates down to 60 px wide), so read the percentages as
before/after, not as field accuracy.

## Telling 1 from I (and 0 from O)

On Malaysian plates a `1` is a bare stroke, the same shape as `I`; only its position says which it is. The first
fine-tuned model had learned `I` from synthetic plates where it could appear anywhere, and read `VGG811` as
`VGG8I`. Two things now decide it:

- **Training data.** Synthetic plates are letters first and the number last, a `1` is usually drawn as the bare
  stroke, and numbers full of 1s (`811`, `2111`, `110`) are over-sampled. With every `UITM` and `VGG` plate held
  out, the retrained model keeps `UITM1776` and reads the stroke in `VGG811` as a `1` again.
- **A format rule in `lpr-api`.** `--plate-format my` (also in a channel's settings in the web UI) reads the
  plate as letters + 1–4 digits + at most one suffix letter and changes the fewest `I/1` and `O/0` characters
  that make it fit: `VGG8I` → `VGG81`, `U1TM1776` → `UITM1776`. Reads that already fit, or cannot fit, are
  left alone, as are the digit-bearing series `G1M`, `1M4U`, `T1M`, `A1M`, `K1M`. Off by default. The Android
  app does not have it yet.

A separate problem on the same plate is **not** solved in general: the stock model reads `VGG811` as `VGG81`.
That plate is long with small lettering, so in the 96×48 crop the two strokes of `11` are about 3 px apart and
merge into one character. Cropping 10–30 % tighter horizontally makes both models read it, but cuts plates
whose lettering fills the width (`PUTRAJAYA541`). The model in `weights_ft/` reads the three `VGG811` photos
because it was trained on them; held out, it still reads `VGG81`, and extra synthetic small-lettering plates
did not change that. More real photos of tightly spaced repeats are what would teach it — or a crop that
follows the lettering instead of the plate edge.

## Current `weights_ft/`

Trained on the nine photos now in `sample_data` with the format-aware synthetic data: all nine read correctly
through `lpr-api read`, with or without `--plate-format my` (stock: 5 of 9), and the four SDK crops from other
countries still read. Held-out run of the same recipe (`UITM*` and `VGG*` out): `UITM1776` correct, `VGG811` →
`VGG81`, synthetic plates with `I`/`O` 11 % → 49 %, with `11` 52 % → 67 %, without `I`/`O` 70 % → 80 %.

## Limits

- Four real photos is a smoke test, not a benchmark. The synthetic numbers show the direction; how much the
  model improves on a camera's real traffic is unmeasured until there is a labelled set from that camera.
  `lpr-api bench` can compare `weights` and `weights_ft` over such a set.
- Synthetic plates only approximate real ones. The best use of this is to collect the plates that read wrong
  in production, label them by file name, and retrain: real crops are mixed into every batch (`--real-frac`).
- Hold some real plates out with `--holdout` whenever there are enough of them; a model that has seen a photo
  will read it, which proves nothing.
- Only the 37-class 0-9A-Z models are trainable here; other alphabets need their label tables first.
- The fine-tuned file is a derivative of the SDK's weights and carries the same licence limits as the originals.
