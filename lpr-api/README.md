# lpr-api — the extracted OCR model behind a camera-compatible API

A Rust service and benchmark harness over `../weights/rec_NN.bin`.

- **`serve`** reads a plate from an image and answers in the shape of the camera provider's event
  (the JSON a Vaxtor camera posts to ParkingDashboard's `POST /api/alpr/events`). `/v1/trigger`
  also posts that event to a ParkingDashboard API, so this model can stand in for the camera.
- The same server has a **web UI** for RTSP channels: each channel decodes a camera stream, reads
  plates with its own choice of models and skew correction, and raises events.
- **`bench`** runs one image set through this model and any other plate-reading API and reports
  accuracy and latency side by side.
- **`read`** prints the read for image files, for quick checks.

The pipeline is a port of the Android app's default preset: rec_72 detector → rec_71 corner
rectification → 96×48 crop → rec_57 CRNN. The image preprocessing reproduces Pillow's, as the
Kotlin port does, because the OCR is sensitive to it.

> The weights are the unpacked assets of a commercial ANPR SDK. Like the Android app, this is for
> local research and benchmarking; the weights are read from disk at run time and not bundled.

## Build

```bash
cd lpr-api
cargo build --release
```

Needs Rust 1.88+, CMake and a C++ compiler. The first build downloads and compiles MNN 3.6.1 from
`github.com/alibaba/MNN` (about two minutes; set `MNN_SOURCE_DIR` to use a local checkout) and
downloads a prebuilt ONNX Runtime. Both are linked statically, so the binary stands alone.

```bash
cargo test --release     # 31 tests; one reads the three samples end to end
```

### OpenVINO build (Ubuntu 26.04, x86_64)

```bash
scripts/build-openvino-ubuntu.sh
```

The `ort` crate has no prebuilt ONNX Runtime with the OpenVINO provider, so the script downloads
Intel's OpenVINO 2026.3.1 runtime, builds ONNX Runtime 1.28.3 from source against it (20-60
minutes, about 6 GB under `.openvino-build/`), builds this crate with `--features openvino`, and
bundles the result in `dist/lpr-api-openvino/`. It finishes by timing the samples on both
backends. Settings are environment variables documented at the top of the script.

```bash
dist/lpr-api-openvino/lpr-api serve --weights ../weights --detector-backend openvino
```

`--detector-backend openvino` (or `LPR_DETECTOR_BACKEND=openvino`) works on `serve`, `read` and
`bench`, and applies to every channel. Only the detector moves to OpenVINO; the OCR and corner
models stay on MNN. If the provider cannot load, the command fails rather than falling back to
the default CPU kernels, so a benchmark never measures the wrong backend. An ordinary build
refuses the flag.

The script has not been run on Ubuntu: it was written and checked from macOS (syntax, the
download and its checksum, the archive layout, and the ONNX Runtime build flags). The x86 build of
this crate, including MNN, is likewise untested.

## Serve

```bash
./target/release/lpr-api serve --weights ../weights
```

Listens on `127.0.0.1:8088`. An image is accepted three ways:

```bash
# raw body
curl -X POST -H 'content-type: image/jpeg' --data-binary @../sample_data/BRL4104.jpeg \
     'http://127.0.0.1:8088/v1/read?cameraid=CAM-7'

# multipart: file field image, upload or file; optional cameraid and date fields
curl -F upload=@../sample_data/EV232.jpeg -F cameraid=CAM-7 http://127.0.0.1:8088/v1/read

# JSON with a base64 "image" - the provider's own payload, so a captured camera post replays as is
curl -X POST -H 'content-type: application/json' --data-binary @captured-event.json http://127.0.0.1:8088/v1/read
```

The reply has the provider's fields at the top level, every value quoted as the camera's template
does, plus extras:

```json
{
  "plate": "BRL4104", "date": "2026-10-05T15:20:21.061+08:00", "confidence": "99.21", "cameraid": "CAM-7",
  "top": "447", "left": "946", "bottom": "503", "right": "1234", "height": "1080",
  "absolutetop": "0.41", "absoluteleft": "0.49", "sizeinbytes": "574489", "image": null, "cropplate": null,
  "plates": [{ "plate": "BRL4104", "confidence": 0.992, "det_score": 0.745, "box": {}, "corners": [], "ocr_model": "rec_57" }],
  "frame": { "width": 1920, "height": 1080 },
  "timing_ms": { "decode": 13.0, "det_prep": 4.5, "det_infer": 14.8, "rectify": 1.2, "region": 0.0, "ocr": 1.5,
                 "total": 34.9, "queued": 0.4, "request": 35.5 }
}
```

- `plate` is the most confident detection; `plates` lists every plate in the frame. `plate` is
  `null` when nothing was read.
- `confidence` is 0–100 like the camera's; it is the mean CTC step probability, not a calibrated score.
- `timing_ms.total` is decode plus pipeline. `request` is the whole time in the handler, including
  receiving the body and `queued`, the wait for a free worker.
- `?images=1` embeds the frame and the 96×48 plate crop as base64 in `image` and `cropplate`.

### Triggering ParkingDashboard

Set the URL under **Settings** in the web UI (or start with `--target http://localhost:5200`), then:

```bash
curl -X POST -H 'content-type: image/jpeg' --data-binary @frame.jpg \
     'http://127.0.0.1:8088/v1/trigger?cameraid=SIM-ENTRY'
```

`/v1/trigger` reads the image, then posts the event (with `image` and `cropplate`; `?images=0`
leaves them out) to `{target}/api/alpr/events`. The reply adds `delivery` with the API's status,
its response (`gateDecision`, `gateOpened`, …) and the time the post took. Nothing is sent when no
plate is read, as with the camera.

The URL is a ParkingDashboard base URL (`http://localhost:5200` posts to its `/api/alpr/events`)
or, with a path, any endpoint that takes the camera payload; that one is used exactly as written.
It is saved in the channels file and takes effect on the next read without a restart. `--target`
only supplies the initial value: once a URL has been saved from the UI, the saved one is used.

A read sent to a real site is real: it can raise a ticket and open a barrier. A URL that is not
this machine is refused until "This URL is on another machine" is ticked in Settings (or
`--allow-remote` accompanies `--target`). The camera id must match a lane's camera id for the API
to decide anything.

## RTSP channels (web UI)

Open `http://127.0.0.1:8088/` while `serve` is running. Decoding needs `ffmpeg` on the PATH (or
`--ffmpeg /path/to/ffmpeg`); nothing else is installed.

The main page is the **live view**: a tile per channel with its picture, the plate outlined and
labelled, the last read, frame rate and time per frame, and below them the recent reads across
all channels. A channel that is down shows "No signal" with ffmpeg's message. Clicking a tile
opens that channel's settings; the **Add channel** box at the end of the tiles creates one, and
**Settings** holds the HTTP URL reads are posted to.

**Live picture.** The selector above the tiles chooses how pictures arrive, per browser:

- *Snapshots, one per second* (the default) polls `/api/channels/{id}/snapshot.jpg`.
- *MJPEG video* plays `/api/channels/{id}/stream.mjpg`, every frame the channel reads, so motion
  is as smooth as the channel's "frames read per second". It costs one JPEG encode per frame
  (shared between viewers of the same channel and size): about a tenth of a core for 15 frames a
  second at tile size, against 2-3% for snapshots, plus the browser's decoding. Reads are
  fetched three times a second in this mode so the outlines keep up.
- A browser opens at most six connections to one server and each video holds one, so video plays
  on the first four running channels and the rest stay on snapshots.
- Both URLs take `?w=` (width, 160-1920); the stream also takes `?fps=` to cap its rate, and
  works in anything that plays motion JPEG.

**Events.** Clicking a row of the recent reads opens the event in a pop-up: the frame the plate
was read in (kept at up to 1280 pixels wide) with the plate and its vehicle outlined, the crop
the OCR read, time, channel, camera ID, confidence, model, the vehicle details, and what the HTTP
URL answered. Esc, *Close* or a click outside closes it.

Each channel has six groups of settings:

| Group | Settings |
|---|---|
| **Stream** | name, `rtsp://` or `rtsps://` URL, TCP or UDP transport, frames read per second (default 5), listen on/off |
| **Models** | detector (320 or 640); OCR strategy: one pinned CRNN, region-routed, or an ensemble; which CRNNs are in the ensemble (also the fallback when routing finds no group); which region classifiers run |
| **Vehicle details** | make and model, pose (seen from the front or the rear), colour, type; each on or off |
| **Skew correction** | corner rectification with rec_71 on/off and its context margins; shear (−1 to 1, positive straightens right-leaning italics); padding around the plate. Presets: *Corner rectification* (the app's default), *Shear only* (the PoC's 0.3 shear, 2% padding), *Off* |
| **Detection area** | a region drawn on the channel's picture, and whether to skip frames where nothing has changed (and how much of the area must change, default 1%) |
| **Events** | minimum confidence, how many frames must agree, how long the same plate is ignored, the camera ID, and whether to post events to the HTTP URL from Settings |

These are the Android app's geometry settings (`rectify`, `cornerMarginX/Y`, `cropPad`, `deshear`)
and behave the same: with rectification the plate is warped to 192×96, sheared, then scaled to
96×48; without it the padded detector box is sheared and scaled. The same settings are flags on
`read`, `bench` and `serve` (`--no-rectify`, `--deshear`, `--corner-margin-x`, …).

A channel's page shows its frame larger, with the 96×48 image the OCR actually read, so a skew
change can be judged on the next plate after saving.

**Vehicle details (MMC).** The detector also finds vehicles; the smallest vehicle box around a
plate is cropped to 224×224 and given to rec_15 (make/model and pose in one pass), rec_19
(colour) and rec_21 (type), whichever are ticked. On a channel this runs once per event, on the
frame that raises it, not on every frame: about 10 ms for all four. The result is logged with the
event, shown in the reads table and the event pop-up, and added to the posted payload as
`vehiclemake`, `vehiclemodel`, `vehiclecolor`, `vehicletype` and `vehiclepose` (`Front` or
`Rear`); these keys are absent when nothing was read, so the payload is unchanged with the
feature off. An event has no vehicle details when the detector finds no vehicle around the plate
(score below 0.40), which happens when the vehicle is cut off by the picture or by a tight
detection area. On the samples: BRL4104 is a front-facing Toyota Corolla Cross SUV, EV232 a BYD
Atto 3, as the Android app found. Colour is only as good as the picture: the samples are
monochrome night-mode frames, so their colours (White/Grey/Black) are unconfirmed.

The same switches are flags for `read`, `bench` and `serve` (`--mmc-make-model`, `--mmc-pose`,
`--mmc-colour`, `--mmc-type`); there every plate's vehicle is described, in `plates[].vehicle`.

**Detection region.** On a channel's page, *Draw region* lets you click points on the picture to
outline the lane (3 to 32 points; *Undo point*, *Done*, then *Save*). The detector then sees only
the region's bounding box, so a plate fills more of its 320-pixel input, and only plates centred
inside the outline are read. The region is stored as fractions of the picture, so it survives a
change of stream resolution. It is shown dashed on the channel page and on the live view.

**Skipping unchanged frames.** With this on (the default), each frame is first compared with the
last frame that was read, on a 64×36 grid of brightness samples inside the detection area, which
costs almost nothing. A frame is read when at least the set share of the grid has changed, for 2
seconds after the last change (a car that has just stopped still needs several agreeing reads),
and once every 5 seconds regardless. Otherwise the detector does not run. Two consequences:

- A parked car is reported once, when it arrives, rather than again after every repeat window.
- If the threshold is set higher than the change a vehicle causes, its plate is not read. The
  default of 1% is well below a vehicle entering the area; raise it only for a noisy picture.

Measured with four channels at 15 fps on a test video of a gate (8 s empty, 8 s with a car):
3.6 cores without skipping, 1.1 with it, the same plates read. That video is perfectly still
between cars, so a real camera with sensor noise, moving shadows or rain will save less. The
region made each read about 10% cheaper on these frames; its main purpose is accuracy on small
plates and ignoring other traffic, which the samples cannot show.

How it runs:

- One ffmpeg process and one set of models per channel, so channels do not wait on each other.
  Saving a channel restarts it with the new settings.
- Only the newest frame is kept: if reading is slower than the stream, frames are skipped (the
  count is shown) rather than queued.
- A plate becomes an event when it is read in enough frames within 3 s, and not again until the
  repeat window passes. A one-frame misread never becomes an event.
- A stream that ends, cannot be opened, or delivers nothing for 15 s is reconnected after 2 s,
  backing off to 15 s. The channel shows ffmpeg's last message meanwhile.
- A channel that posts events sends the camera payload (frame and crop included) to the HTTP URL
  from Settings, the same one `/v1/trigger` uses.
- Channels are saved to `lpr-channels.json` (`--channels-file`) and start again with the server.
  Stream URLs often contain camera passwords: the file is written owner-only, and the UI and API
  show passwords as `***`.

The UI and its API (`/api/channels`, `/api/settings`, `/api/events`, `/api/models`) have no
login, and they decide which streams are read and where frames are posted. Keep `--listen` on
localhost, or put it behind something that authenticates, before exposing it to a network.

`--allow-file-sources` lets a channel read a looping video file on the server instead of a stream,
for trying settings without a camera.

Measured with two channels on 1920×1080 H.264 over RTSP (VLC serving a test video of the three
samples): 5 fps and 4 fps held with no skipped frames, 25–31 ms per frame, about 70% of one core
and 320 MB for the whole process.

### Options

| Flag | Default | |
|---|---|---|
| `--weights` / `LPR_WEIGHTS` | `weights` | directory of `rec_NN.bin` |
| `--ocr` | `pinned` | `pinned` (one CRNN), `routed` (region classifiers pick the CRNN; +10 ms, falls back to the ensemble when no group claims the plate), `ensemble` (all nine 0-9A-Z CRNNs, most confident wins; +9 ms, and it misreads BRL4104 as BRL404) |
| `--ocr-model` | `57` | the CRNN for `pinned`; rec_57 is the one verified on Malaysian plates |
| `--detector` | `320` | `640` uses rec_75 for small or distant plates (detector 37 ms instead of 12) |
| `--ensemble-models` | all nine | the CRNNs `ensemble` runs, and `routed` falls back to |
| `--region-classifiers` | all ten | the region classifiers `routed` (or `--region`) runs |
| `--no-rectify` | off | skip rec_71 and read the padded detector box (misreads two of the three samples) |
| `--corner-margin-x`, `--corner-margin-y` | `0.2`, `0.5` | context around the detector box given to rec_71 |
| `--deshear` | `0` | horizontal shear; `--no-rectify --deshear 0.3 --crop-pad 0.02` is the PoC's setting |
| `--plate-format` | `none` | `my` settles look-alike `I/1` and `O/0` by position: letters, then 1-4 digits, then at most one letter |
| `--crop-pad` | `0.03` | padding around the plate |
| `--mmc-make-model`, `--mmc-pose`, `--mmc-colour`, `--mmc-type` | off | vehicle details for each plate's vehicle (about 10 ms for all four) |
| `--threads` | `4` | per model, ONNX Runtime and MNN |
| `--workers` | `2` | `serve` only: model sets loaded, one request each at a time |

## Bench

```bash
# this model alone; the true plate is the file name (BRL4104.jpeg, BRL4104_2.jpg)
./target/release/lpr-api bench --weights ../weights --images ../sample_data

# against other APIs
cp providers.example.toml providers.toml      # then edit
PLATE_RECOGNIZER_TOKEN=... ./target/release/lpr-api bench --weights ../weights \
    --images /path/to/images --labels labels.csv --providers providers.toml --csv rows.csv --json summary.json
```

```
provider      images  read  errors       exact  char acc  agree w/ local  p50 ms  p90 ms  p99 ms  mean ms
------------  ------  ----  ------  ----------  --------  --------------  ------  ------  ------  -------
local              3     3       0  3/3 100.0%    100.0%          100.0%    25.9    26.6    27.5     25.6
lpr-api-http       3     3       0  3/3 100.0%    100.0%          100.0%    26.7    27.2    27.5     26.4

local stage medians (ms): decode 8.2, det_prep 3.2, det_infer 12.4, rectify 1.0, region 0.0, ocr 1.1
```

(Apple-silicon laptop, 1920×1080 JPEGs, 20 calls per image.)

**Providers**

- `local` is the pipeline in this process, timed from the encoded image bytes to the read, so it
  includes JPEG decode like any HTTP API does. `--no-local` leaves it out.
- HTTP APIs come from `--providers`, a TOML file: URL, how the image is sent (raw, multipart or
  base64 JSON), headers, and JSON pointers to the plate, confidence and the API's own reported
  time. `providers.example.toml` has this crate's server and a Plate Recognizer template. Tokens
  are read from the environment with `${NAME}`. Their latency is measured at the client, so it
  includes the network; when the API reports its own processing time, that is shown separately.
- Recorded reads: any column after `file,plate` in the labels CSV is treated as another API's
  read for that image. This is how to score an API that cannot be called on demand, such as the
  plates the cameras themselves reported for the same frames. It gets accuracy columns only.

**Truth and scoring**

- `--labels labels.csv` (`file,plate[,name…]`) gives the true plates; only labelled images run.
  Without it the file name up to the first `_` or `.` is the plate. `--no-truth` says neither
  applies: then only agreement between providers is reported.
- Plates compare upper-cased with spaces and punctuation removed. `exact` is whole-plate matches;
  `char acc` is 1 − edit distance ÷ true characters, summed over the set; a missing read counts as
  every character wrong.
- `agree w/ <first>` is the share of images where the provider gives the same read as the first
  provider, useful without labels.
- Each image is called `--runs` times (default 3) after `--warmup` untimed calls (default 2).
  The read is taken from the first call. Failed calls are counted under `errors` and not timed.
- Images where a provider is wrong, or providers disagree, are listed after the table.

Every image is uploaded to each HTTP provider in the file, so only list APIs the images may go to.

## Limits

- Accuracy is verified on the three Malaysian samples only; that is not a benchmark. A labelled
  set of real frames is needed before the numbers mean anything.
- Calls are sequential: latency is per request with no contention, not throughput under load.
- For `/v1/read` and `/v1/trigger` one plate becomes the event (the highest detector score).
- Vehicle details were checked against the Android app's results on two pictures; make/model
  accuracy on local traffic is unmeasured.
- Only the 37- and 38-class CRNN alphabets decode to text; other models cannot be pinned.
- JPEG decoding differs slightly from Pillow's, so values drift from the Python goldens in the
  third significant figure (the reads are the same).
- EXIF orientation is ignored.
- RTSP channels were tested against a local VLC stream, not a physical camera. Streams are
  decoded on the CPU by ffmpeg at full resolution.
- Stream events, with their frames, are kept in memory (the last 200, roughly 30 MB) and are
  lost on restart.
