# Axios LPR — Android kitchen sink for the ANPR SDK weights

On-device licence-plate recognition built directly on `../weights/rec_NN.bin`.
It is a port of `read_plate.py` plus every other model in the SDK, with a toggle for each one.

> The weights are the unpacked assets of a commercial ANPR SDK. This project is set up for local
> research builds only: weights are copied into the APK at build time and never committed here.

## Build and install

```bash
cd android
JAVA_HOME=$(/usr/libexec/java_home -v 17) ./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

- Needs Android SDK 36, NDK 27.0.12077973, CMake 3.22.1 and JDK 17. `local.properties` points at the SDK.
- The first build downloads and compiles MNN 3.6.1 from `github.com/alibaba/MNN`, which takes about two minutes.
- Only `arm64-v8a` is built. That covers modern phones and Apple-silicon emulators.
- The APK is about 185 MB because it carries all 124 MB of weights uncompressed. Models load straight from the APK, so nothing is copied to storage.

## What's in the app

| Tab | What it does |
|---|---|
| **Scan** | Live viewfinder. Each plate gets a box labelled with its reading and confidence, stabilised by a vote across frames. Vehicle boxes show MMC results. Shutter snaps a full-resolution still. Video mode records an MP4 while plates are logged against the video timeline. Auto-record saves every stable plate, with a repeat window. Also: torch, pinch zoom, tap to focus, ROI guide box, performance HUD, and a quick sheet for MMC and OCR toggles. |
| **Simulated camera** | In the quick sheet. Replays photos through the exact live path, so the viewfinder can be tested without a car or on an emulator. |
| **Import** | Photo Picker (multi-select), Files through the Storage Access Framework, and share-to-app from Gallery or Files. Several images become one batch session. |
| **History** | Every stored plate, with search, review filters, source and session filters. Export CSV, JSON, or a training zip. |
| **Capture detail** | Frame with boxes, detector crop, rectified crop, the exact 96×48 CRNN input, every model's read, region classifiers, lookup hints and vehicle attributes. The **corrected reading** field saves the true text for fine-tuning; the prediction is never overwritten. |
| **Lab** | All 87 records with shapes and notes. On/off switch per model, per-model benchmark, backend status, and a self-test over the SDK's bundled sample inputs (26 checks). |
| **Settings** | Presets, detector 320/640, thresholds, OCR strategy, per-CRNN and per-classifier toggles, geometry, MMC toggles, MNN backend and precision, ONNX Runtime provider, thread counts, live and recording behaviour. |

## Pipeline

1. **Detector** rec_72 (320) or rec_75 (640) runs on ONNX Runtime. It finds plates and vehicles, with per-class NMS.
2. **Corner rectification** with rec_71. The plate box is expanded by a context margin (0.2 × width, 0.5 × height) and fed as 96×48 BGR in [-1, 1]. Four corners come back in the order top-left, bottom-left, bottom-right, top-right. The plate is then quad-warped to 192×96 and downscaled to 96×48.
3. **Region classifiers**: ten MobileNetV3 models take 96×48 BGR at raw 0–255. Only classifiers that have a 9999 reject class may route a plate.
4. **OCR** strategies: pinned CRNN, region-routed, ensemble max-confidence, or ensemble vote. Snap and import also run every enabled CRNN for comparison.
5. **MMC**: rec_15 (make/model + pose), rec_19 (colour) and rec_21 (type). Input is a 224×224 RGB vehicle crop in [-1, 1]. rec_78 is an experimental vehicle box refiner.
6. **Lookups**: German district, Spanish province, Jordanian category and Kazakh region hints. A routed plate only gets hints for its own country.

The default preset, "Malaysia (verified)", reads all three samples correctly: BRL4104, EV232, PUTRAJAYA541.
The original PoC settings (`--ocr-model 57 --deshear 0.3`) are kept as the "PoC shear 0.3" preset. That preset misreads PUTRAJAYA541 as PUTRAJAYA1541, exactly as `read_plate.py` does.

### Findings that differ from HANDOVER.md

- **MMC models want normalised input**, not raw 0–255. Raw input labels every car "Large Truck". With [-1, 1] the BRL4104 car reads as a frontal Toyota Corolla Cross SUV, and the bundled rec_17 sample reads as a rear Mitsubishi Outlander SUV.
- **rec_17 is stored RGB**, not BGR: its tail lights turn blue when decoded as BGR.
- **rec_71 corners work**, but only with a context margin and [-1, 1] input. Rectification replaces the hand-tuned shear.
- **Region routing works for Malaysia.** The classifier for group 1187 labels BRL4104 as region 1125, and that group's CRNN is rec_57.
- **rec_78 is a vehicle box refiner** (x1, y1, x2, y2 within a vehicle crop), not a plate locator.

## Storage

SQLite via Room (`axios_lpr.db`) has five tables: `session`, `capture`, `plate`, `plate_read` and `vehicle`.
Images live under `filesDir/captures/<uuid>/`:

- `frame.jpg`: the full frame.
- `original.*`: an untouched copy of an imported file.
- `plate_N_raw.jpg`: the padded detector crop.
- `plate_N_input96x48.png`: the exact CRNN input.
- `plate_N_rectified.png`: the rectified plate.
- `vehicle_N.jpg`: the vehicle crop.

`plate.corrected_text` holds the user's reading. The training zip bundles the 96×48 inputs, raw crops and a `labels.csv` with the preprocessing parameters.

## GPU backends

MNN OpenCL and Vulkan are compiled in. Every GPU session is checked against a CPU session on load, and the app falls back to CPU if they disagree. On the Apple-silicon emulator, Vulkan passes for the region classifiers but rec_57 fails (difference 0.987), so it falls back automatically.

## Tests

```bash
./gradlew :app:testDebugUnitTest            # 24 JVM tests: Pillow parity, CTC, NMS, tracker, mapper, lookups, config
./gradlew :app:connectedDebugAndroidTest    # 9 device tests: real models vs Python goldens, routing, MMC, GPU fallback, storage
```

`tools/make_goldens.py` regenerates `golden_pipeline.json` from the Python PoC.

## Known limits

- Only the 37- and 38-class CRNN alphabets decode to text. The others show class indices.
- The SDK ships no region-ID → name table. Names shown come from the samples and are inferred.
- rec_04–14 are custom float blobs whose loader is unknown, so they are not runnable.
- Accuracy is verified on three Malaysian images only.
