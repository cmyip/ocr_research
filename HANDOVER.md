# Handover: ANPR SDK weights — reverse-engineering & plate-reader pipeline

_Last updated: 2026-09-28 · Source project: `D:\temp\dnn_research`_

## TL;DR

- `weights/rec_00.bin … rec_86.bin` (87 files) are the unpacked assets of a commercial **ANPR / vehicle-analytics SDK**: plate + vehicle detectors, plate corner/bbox refiners, per-country region classifiers, per-alphabet CRNN OCR, vehicle make/model/colour/type classifiers, label tables and sample inputs.
- Every model ↔ label-file ↔ sample-input pairing was **verified** by matching output dims to label line counts and input shapes to raw-image byte sizes (29 matches, 0 contradictions).
- A working reader, [`read_plate.py`](read_plate.py), returns **`BRL4104`** for `sample_data/BRL4104.jpeg` in **15.8 ms median (GPU detector) / 27.1 ms (CPU)**, but only with a pinned OCR model + de-italic shear tuned on that one image.
- **Biggest open problem:** the SDK's country/region → OCR-model mapping (and non-Latin alphabets) is not in the weight files; it's presumably in the SDK's native code.

## Project files

| File | Purpose |
|---|---|
| `weights/rec_NN.bin` | The 87 SDK assets (untouched) |
| `sample_data/BRL4104.jpeg` | 1920×1080 grayscale CCTV frame, Malaysian plate `BRL 4104` (italic white-on-black) |
| `inspect_weights.py` | Classifies every file, prints model I/O shapes, verifies model→label/sample pairings |
| `weights_report.txt` | Output of `inspect_weights.py` |
| `decode_samples.py` | Runs bundled 96×48 plate crops through all region classifiers and all 20 CRNNs |
| `samples_report.txt` | Output of `decode_samples.py` |
| `read_plate.py` | End-to-end reader: image → detector → crop → OCR, with timing breakdown |
| `.venv-gpu/` | 3.8 GB venv with CUDA-12 `onnxruntime-gpu` for `--gpu` (see Environment) |

## File map (all verified unless marked _inferred_)

### Detection / geometry

| File | Format | What | I/O |
|---|---|---|---|
| rec_72 | ONNX (Ultralytics YOLOv8-style, PyTorch 2.7.0 export) | plate+vehicle detector, small | in `images (1,3,320,320)`, out `(1,6,2100)` = cx,cy,w,h + 2 class scores |
| rec_75 | ONNX, same family | plate+vehicle detector, large | in `(1,3,640,640)`, out `(1,6,8400)` |
| rec_73 / rec_76 | text | detector classes: `plate`, `vehicle` (class 0 = plate) | |
| rec_74 / rec_77 | raw uint8 HWC | sample detector inputs 320² / 640² | |
| rec_71 | MNN, `model/corners/Sigmoid` | plate **4-corner keypoints** (8 values) — _not used yet_ | in `(1,3,48,96)`, out `(1,8)` |
| rec_78 | MNN, `bbox/Sigmoid` | single-box regressor (likely plate-in-vehicle-crop) — _not used yet_ | in `(1,3,224,224)`, out `(1,4)` |

### OCR

| File | What | Output |
|---|---|---|
| rec_50–69 (20) | MNN CRNN (MobileNetV3-ish backbone), one per alphabet/country | `(1, 18, C)`, 18 CTC steps, **blank = last index**. Some emit probabilities, some logits (apply softmax if rows don't sum to 1) |
| rec_04–14 (11) | Custom float32 blobs, header `[30|25, 3, 112, 96, 64, N]` then weights | Final layer is 64→N (each extra class = +65 floats), N = alphabet size (36, 35, 27, 37, 31, 46…). Loader unknown — _cannot run yet_ |

CRNN alphabet sizes: C=37 ×9 (rec_50,53,57,60,63,65,66,68,69), 29 (51,54), 31 (52), 38 (55), 13 (56), 43 (58), 39 (59), 49 (61), 23 (62), 28 (64), 40 (67).

Known alphabets:
- **C=37** → `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ` + blank. **Verified** exact reads: `FHW7186`, `0907CDS`, `SH7194K`, `RTL015`, `990SKR09`, `BRL4104`.
- **C=38** (rec_55) → same 36 + **index 36 = group separator / line break** (`3 43466` → `3,36,4,3,4,6,6`).
- **C=49** (rec_61, Thai) → digits at 0–9, Thai consonants elsewhere (ศ→37, ฉ→15?) — _partially inferred_.
- **C=29** (rec_54) → digits 0–9 + reduced Latin set, A=10, E=13, K=18 — _partially inferred_.
- Others unknown.

### Region classifiers (MobileNetV3, input 96×48 plate crop)

Each model is followed by its **class-ID list** and a **sample 96×48 plate crop** (13824 B raw BGR):

| Model | Classes | ID list | Sample | Prediction on own sample |
|---|---|---|---|---|
| rec_23 | 55 | rec_24 | rec_25 (US, `FHW7186` North Carolina) | `1222 2033` (1.00) |
| rec_26 | 53 | rec_27 | rec_28 (`0907CDS`) | `1195` |
| rec_29 | 11 | rec_30 | rec_31 (Singapore `SH7194K`) | `1187` |
| rec_32 | 11 | rec_33 | rec_34 (Victoria AU `RTL015`) | `1012 2207` |
| rec_35 | 12 | rec_36 | rec_37 (Kuwait `3 43466`) | `1110` |
| rec_38 | 8 | rec_39 | rec_40 (Kazakhstan `990SKR09`) | `1106` |
| rec_41 | 78 | rec_42 (4000–4077) | rec_43 (Thailand, Bangkok) | `4001` |
| rec_44 | 4 | rec_45 | rec_46 (`ASI927`) | `1044` |
| rec_47 | 48 | rec_48 (4400–) | rec_49 (UAE-style `C 9998`) | `4416` |
| rec_82 | 2 | rec_83 (4100/4101) | rec_84 (`3581 KEA`, Arabic/Latin) | `4101` |

ID scheme (_inferred_): `1xxx` = country (rec_00 lists 1001… with **193** entries = UN member count), `2xxx` = state/sub-region (tab-separated `country<TAB>state`), `4xxx` = region sets, `9999` = unknown/other. NC → `2033` fits alphabetical US states incl. DC starting at 2000. **ID → name tables are not in the files.**

### Vehicle attributes (MobileNetV3, input 224×224)

| Model | Output | Labels |
|---|---|---|
| rec_15 | 2 heads: `predictions_mmr` (1,4193) + `predictions_pose` (1,2) | rec_16 (4193 make/model lines, cp1252/mixed), rec_18 (`Frontal`,`Rear`); sample rec_17 (224² BGR) |
| rec_19 | (1,14) colour | rec_20 |
| rec_21 | (1,8) vehicle type | rec_22 (`BIGTRUCK`, `BUS`, `CAR`, `MOTORBIKE`, `PICKUP`, `SMALLTRUCK`, …) |

### Other tables / junk

- rec_00, rec_24, 27, 30, 33, 36, 39, 42, 45, 48, 83 — numeric ID lists (above).
- rec_01 — ADR hazardous-goods (Kemler/UN) code config, mostly UTF-8 with a few bad bytes.
- rec_02 German district codes · rec_03 Spanish provinces · rec_85 Jordanian plate categories (65) · rec_86 Kazakh regions (41) — not tied to a model; likely OCR post-processing / region lookup.
- rec_70, 79, 80, 81 — literally `dummy`.

## Preprocessing (empirically established)

| Model family | Input |
|---|---|
| YOLO detector | RGB, letterbox to 320 (top-left, pad 114), `/255`, NCHW float32 |
| CRNN OCR | 96×48 **BGR**, `/255`, NCHW (channel order barely matters) |
| MobileNetV3 classifiers (region, colour, type, MMR) | **raw 0–255** float, NCHW — Keras MobileNetV3 has built-in rescaling. Feeding `/255` gives mostly `9999`/wrong at low confidence |

## Pipeline — `read_plate.py`

1. rec_72 detector → best `plate` box (score ≥ 0.25).
2. Crop with `CROP_PAD = 0.02`; optional `--deshear` horizontal shear to straighten italic fonts.
3. Resize to 96×48, BGR, `/255` → CRNN, greedy CTC decode.
4. OCR model: `--ocr-model NN` pins one; default runs all nine 37-class models and keeps the most confident (**unreliable** — see below).

```bash
# CPU
python read_plate.py sample_data/BRL4104.jpeg --ocr-model 57 --deshear 0.3
# GPU detector
.venv-gpu/Scripts/python read_plate.py sample_data/BRL4104.jpeg --ocr-model 57 --deshear 0.3 --gpu
```

Flags: `--runs N` (timed runs after warm-up), `--save-crop PATH`, `--gpu`.
Python API: `load_models(ocr_model, gpu)` → `read_plate(img, detector, ocr, deshear)` → `(text, conf, info)`.

### Results on `BRL4104.jpeg`

- Detected box `[946, 447, 1233, 503]`, det score 0.75.
- Default (all 37-class models, max confidence) → **`BRL404`** (rec_53, 0.99, confidently wrong).
- `--ocr-model 57 --deshear 0.3` → **`BRL4104`** (0.993). rec_57 is the **only** CRNN that ever reads the `1`.
- Robustness sweep (rec_57): correct for pad ≤ 0.06 and shear 0.3–0.5; flips to `BRL4004`/`BRL404` outside. **Tuned on one image — treat as fragile.**
- De-shear sign: positive `--deshear` straightens right-leaning italics (internally affine `x_in = x - s·y + s·H/2`).

### Timing (median of 200 runs, `--ocr-model 57 --deshear 0.3`)

| Stage | CPU (original) | CPU (now) | GPU detector (now) |
|---|---|---|---|
| detector preprocessing | 16.8 ms | 6.9 ms | 5.5 ms |
| detector inference | 15.5 ms | 14.1 ms | **5.3 ms** |
| crop + OCR | 7.4 ms | 6.1 ms | 4.7 ms |
| **total per image** | **39.9 ms** | **27.1 ms** | **15.8 ms** |

Not included: JPEG decode (28–65 ms for 1920×1080; irrelevant for decoded video frames), model load (~0.15 s CPU / ~1 s GPU, one-off). Running all nine OCR models instead of one adds ~25–35 ms on CPU. Preprocessing speed-up came from `Image.resize(..., reducing_gap=2.0)` + numpy letterbox.

GPU details: CUDA EP with `cudnn_conv_algo_search=EXHAUSTIVE`, `enable_cuda_graph=1`, IOBinding with a persistent device input updated in place. No CPU fallback in `--gpu` mode (fails loudly). Output matches CPU within 2e-4. FP16 conversion (`onnxconverter-common`) fails on the `Resize` node even with `op_block_list=['Resize']`; not pursued (GTX 1650 has no tensor cores, model is launch-bound).

## Environment

- Windows 11, Python 3.13.0 (user site-packages), GPU **NVIDIA GTX 1650 laptop, 4 GB, driver 573.22 → CUDA ≤ 12.8**.
- Global Python: `numpy 2.5.3`, `onnx 1.23.0`, `onnxruntime 1.30.0` (CPU), `MNN 3.6.1`, `protobuf 7.36.2`, `Pillow`.
- `.venv-gpu`: `onnxruntime-gpu==1.24.4` (last CUDA-12 build; 1.25+ need CUDA 13 → error 801 on this driver) with pinned `nvidia-cudnn-cu12 9.10.2.21` (9.26 fails: `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`), `nvidia-cuda-runtime-cu12 12.8.90`, `nvidia-cublas-cu12 12.8.5.5`, cufft 11.3.3.83, curand 10.3.9.90, nvrtc/nvjitlink 12.8.93, plus `numpy`, `pillow`, `MNN`, `onnx`, `onnxconverter-common`. Call `ort.preload_dlls()` before creating sessions.
- MNN pip wheel is **CPU-only**; `MNN.Interpreter` API used (createSession / getSessionInput / copyFrom / runSession / copyToHostTensor). MNN prints `The device supports: …` to stderr on load — filter it.

## ⚠️ Gotcha: MNN CLI tools phone home and mutate your env

`MNN/tools/utils/log.py` (imported by `mnnconvert` and other `MNN.tools` entry points) on import:
1. runs `os.system("pip install -U aliyun-log-python-sdk")` if missing → this **downgraded protobuf to 5.29.6 and broke `onnx`**;
2. fetches STS credentials from `https://1032277949409193.cn-hangzhou.fc.aliyuncs.com/2016-08-15/proxy/mnn-service/workstation-sts/`;
3. on real conversions, `put_log`s tool args, model GUID, sizes and a machine ID to Aliyun Log Service (`mnn-monitor/mnn-tools`).

Only `--help` was run (exits before `put_log`), so no data was uploaded. The auto-installed packages were removed and protobuf restored (`pip check` clean). **Don't run `mnnconvert` / `MNN.tools` again without blocking network or stubbing `log.py`.** `mnnconvert` cannot export MNN → ONNX anyway (output is MNN only). `MNN.Interpreter` does not import the logger.

## Open questions / next steps

1. **OCR model selection.** Find the country → CRNN mapping. Ideas: run each region classifier on plates and correlate with which CRNN reads them; decompile the SDK native lib (if available) for the table; build a small labelled set per country and pick the best CRNN empirically. For Malaysia, rec_57 is the only candidate so far — validate on more Malaysian plates (especially `1`, `I`, italic fonts, two-row plates).
2. **Use rec_71 (corner keypoints)** to perspective-rectify the plate before OCR — may remove the need for hand-tuned `--deshear`/`CROP_PAD`. Input is the 96×48 crop; output 8 sigmoid values (likely normalised x,y ×4, order unverified).
3. **Recover unknown alphabets** (C = 13, 23, 28, 29, 31, 39, 40, 43, 49) via plates with known text, as done for C=37/38.
4. **Custom float blobs rec_04–14**: reverse the layout (header `[T,3,112,96,64,N]`, final 64→N FC) — possibly an older/alternate OCR or char classifier.
5. **Region-ID names**: map `1xxx` country IDs (193 entries, likely UN list order?) and `2xxx`/`4xxx` sub-regions.
6. **Further speed**: TensorRT EP (multi-GB install, expect detector ~2–3 ms); batch frames across cameras; do letterbox resize on GPU; OCR on GPU would need MNN built from source with CUDA (OCR is only ~3 ms, low priority).
7. Try the 640² detector (rec_75) for small/distant plates.
