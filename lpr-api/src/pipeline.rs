//! Image → plates. A port of the Android app's default pipeline ("Malaysia (verified)" preset):
//! rec_72/75 detector → rec_71 corner rectification → 96×48 crop → CRNN, with optional region
//! routing through the ten group classifiers. See HANDOVER.md for how each step was established.
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};

use crate::image::{round_half_even, RgbImage};
use crate::mnn::Net;

const ALNUM: &str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const PLATE_W: usize = 96;
const PLATE_H: usize = 48;
const DET_320: u32 = 72;
const DET_640: u32 = 75;
const CORNERS: u32 = 71;
/// CRNNs with the 0-9A-Z alphabet: the only ones an ensemble can compare.
pub const LATIN_CRNNS: [u32; 9] = [50, 53, 57, 60, 63, 65, 66, 68, 69];
/// Every CRNN whose alphabet is known, so its output decodes to text (rec_55 adds a separator).
pub const DECODABLE_CRNNS: [u32; 10] = [50, 53, 55, 57, 60, 63, 65, 66, 68, 69];

/// OCR "country groups" recovered from the SDK catalog (model_resolver.py GROUPS):
/// (group name, region classifier, its id-list record, the group's CRNN if it has one).
pub const REGION_GROUPS: [(&str, u32, u32, Option<u32>); 10] = [
    ("2033", 23, 24, Some(50)),
    ("1195", 26, 27, None),
    ("1187", 29, 30, Some(57)),
    ("2207", 32, 33, Some(53)),
    ("1110", 35, 36, Some(55)),
    ("1106", 38, 39, Some(60)),
    ("4001", 41, 42, None),
    ("1044", 44, 45, None),
    ("4416", 47, 48, None),
    ("4101", 82, 83, None),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OcrMode {
    /// Always use one CRNN (--ocr-model).
    Pinned,
    /// The region classifiers pick the group's CRNN; falls back to the most confident 0-9A-Z model.
    Routed,
    /// Run every 0-9A-Z CRNN and keep the most confident read.
    Ensemble,
}

/// What runs the plate detector. The OCR and corner models always run on MNN's CPU backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum DetectorBackend {
    /// ONNX Runtime's own CPU kernels (SSE/AVX2/AVX-512 or NEON, chosen at run time).
    #[default]
    Cpu,
    /// Intel OpenVINO on the CPU; only in a binary built with the `openvino` feature.
    Openvino,
}

/// Every knob of the pipeline. The same struct is the CLI's flags and, as JSON, a channel's
/// model and skew-correction settings; `weights` and `threads` belong to the process, not a channel.
#[derive(Clone, Debug, clap::Args, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PipelineConfig {
    /// Directory holding the SDK's rec_NN.bin files.
    #[arg(long, env = "LPR_WEIGHTS", default_value = "weights")]
    #[serde(skip)]
    pub weights: PathBuf,
    /// Detector input size: 320 (rec_72) or 640 (rec_75, for small or distant plates).
    #[arg(long, default_value_t = 320)]
    pub detector: u32,
    #[arg(long, value_enum, default_value_t = OcrMode::Pinned)]
    pub ocr: OcrMode,
    /// CRNN used when --ocr pinned; rec_57 is the one verified on Malaysian plates.
    #[arg(long, default_value_t = 57)]
    pub ocr_model: u32,
    /// The 0-9A-Z CRNNs an ensemble runs (also the fallback when routing finds no group).
    #[arg(long, value_delimiter = ',', default_values_t = LATIN_CRNNS)]
    pub ensemble_models: Vec<u32>,
    /// Also run the region classifiers when the OCR mode does not need them.
    #[arg(long)]
    pub region: bool,
    /// The region classifiers to run, by record number.
    #[arg(long, value_delimiter = ',', default_values_t = REGION_GROUPS.map(|g| g.1))]
    pub region_classifiers: Vec<u32>,
    /// Skew correction: rec_71 finds the plate's four corners and the plate is warped flat.
    /// --no-rectify reads the padded detector box instead.
    #[arg(long = "no-rectify", action = clap::ArgAction::SetFalse, default_value_t = true)]
    pub rectify: bool,
    /// Context given to the corner model around the detector box, as a fraction of its width.
    #[arg(long, default_value_t = 0.2)]
    #[serde(serialize_with = "short_f32")]
    pub corner_margin_x: f32,
    /// Context given to the corner model around the detector box, as a fraction of its height.
    #[arg(long, default_value_t = 0.5)]
    #[serde(serialize_with = "short_f32")]
    pub corner_margin_y: f32,
    /// Fractional padding around the plate; the OCR is sensitive to this.
    #[arg(long, default_value_t = 0.03)]
    #[serde(serialize_with = "short_f32")]
    pub crop_pad: f32,
    /// Horizontal shear applied to the plate; positive straightens right-leaning italic
    /// characters (the PoC used 0.3 for Malaysian plates without rectification).
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    #[serde(serialize_with = "short_f32")]
    pub deshear: f32,
    /// Detection region: a polygon of (x, y) points as fractions of the frame, drawn in the web UI.
    /// The detector only looks inside its bounding box, and only plates centred inside the polygon
    /// are read. Empty means the whole frame.
    #[arg(skip)]
    #[serde(serialize_with = "short_points")]
    pub roi: Vec<[f32; 2]>,
    #[arg(long, default_value_t = 0.25)]
    #[serde(serialize_with = "short_f32")]
    pub plate_score: f32,
    #[arg(long, default_value_t = 0.45)]
    #[serde(serialize_with = "short_f32")]
    pub nms_iou: f32,
    #[arg(long, default_value_t = 6)]
    pub max_plates: usize,
    /// Threads per model, for both ONNX Runtime and MNN.
    #[arg(long, default_value_t = 4)]
    #[serde(skip)]
    pub threads: usize,
    /// What runs the plate detector; like the weights and threads, this belongs to the process.
    #[arg(long, value_enum, default_value_t = DetectorBackend::Cpu, env = "LPR_DETECTOR_BACKEND")]
    #[serde(skip)]
    pub detector_backend: DetectorBackend,
}

/// Writes a setting as the decimal a person typed (0.2), not its 32-bit neighbour (0.20000000298).
pub fn short_f32<S: serde::Serializer>(v: &f32, s: S) -> std::result::Result<S::Ok, S::Error> {
    s.serialize_f64((*v as f64 * 1e4).round() / 1e4)
}

fn short_points<S: serde::Serializer>(points: &[[f32; 2]], s: S) -> std::result::Result<S::Ok, S::Error> {
    s.collect_seq(points.iter().map(|p| p.map(|v| (v as f64 * 1e4).round() / 1e4)))
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            weights: PathBuf::from("weights"),
            detector: 320,
            ocr: OcrMode::Pinned,
            ocr_model: 57,
            ensemble_models: LATIN_CRNNS.to_vec(),
            region: false,
            region_classifiers: REGION_GROUPS.map(|g| g.1).to_vec(),
            rectify: true,
            corner_margin_x: 0.2,
            corner_margin_y: 0.5,
            crop_pad: 0.03,
            deshear: 0.0,
            roi: Vec::new(),
            plate_score: 0.25,
            nms_iou: 0.45,
            max_plates: 6,
            threads: 4,
            detector_backend: DetectorBackend::Cpu,
        }
    }
}

impl PipelineConfig {
    /// Bounding box of the detection region as fractions of the frame: x0, y0, x1, y1.
    fn roi_extent(&self) -> [f32; 4] {
        let fold = |i: usize, init: f32, f: fn(f32, f32) -> f32| self.roi.iter().map(|p| p[i]).fold(init, f);
        [fold(0, 1.0, f32::min), fold(1, 1.0, f32::min), fold(0, 0.0, f32::max), fold(1, 0.0, f32::max)]
    }

    /// The pixel rectangle (x0, y0, x1, y1) the detector looks at in a frame of this size: the
    /// detection region's bounding box, or the whole frame when no region is drawn.
    pub fn roi_bounds(&self, width: usize, height: usize) -> [usize; 4] {
        if self.roi.is_empty() {
            return [0, 0, width, height];
        }
        let [x0, y0, x1, y1] = self.roi_extent();
        let x0 = ((x0 * width as f32).floor() as usize).min(width.saturating_sub(1));
        let y0 = ((y0 * height as f32).floor() as usize).min(height.saturating_sub(1));
        let x1 = ((x1 * width as f32).ceil() as usize).clamp(x0 + 1, width);
        let y1 = ((y1 * height as f32).ceil() as usize).clamp(y0 + 1, height);
        [x0, y0, x1, y1]
    }

    /// Whether a point in frame pixels lies inside the detection region (always, without one).
    fn roi_contains(&self, x: f32, y: f32, width: usize, height: usize) -> bool {
        if self.roi.is_empty() {
            return true;
        }
        let (px, py) = (x / width as f32, y / height as f32);
        let mut inside = false;
        let mut j = self.roi.len() - 1;
        for i in 0..self.roi.len() {
            let ([xi, yi], [xj, yj]) = (self.roi[i], self.roi[j]);
            if (yi > py) != (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    /// Checks the values a person can set, so a bad channel is refused instead of failing later.
    pub fn validate(&self) -> Result<()> {
        if !matches!(self.detector, 320 | 640) {
            bail!("detector must be 320 or 640");
        }
        if !DECODABLE_CRNNS.contains(&self.ocr_model) {
            bail!("rec_{} cannot be pinned: only {DECODABLE_CRNNS:?} decode to text", self.ocr_model);
        }
        if let Some(bad) = self.ensemble_models.iter().find(|id| !LATIN_CRNNS.contains(id)) {
            bail!("rec_{bad} is not a 0-9A-Z CRNN, so it cannot join the ensemble");
        }
        if self.ocr != OcrMode::Pinned && self.ensemble_models.is_empty() {
            bail!("choose at least one CRNN for the ensemble");
        }
        if let Some(bad) = self.region_classifiers.iter().find(|id| !REGION_GROUPS.iter().any(|g| g.1 == **id)) {
            bail!("rec_{bad} is not a region classifier");
        }
        if self.ocr == OcrMode::Routed && self.region_classifiers.is_empty() {
            bail!("routing needs at least one region classifier");
        }
        let range = |name: &str, v: f32, lo: f32, hi: f32| {
            if v.is_finite() && (lo..=hi).contains(&v) { Ok(()) } else { Err(anyhow!("{name} must be between {lo} and {hi}")) }
        };
        range("corner margin X", self.corner_margin_x, 0.0, 1.0)?;
        range("corner margin Y", self.corner_margin_y, 0.0, 1.5)?;
        range("crop padding", self.crop_pad, 0.0, 0.5)?;
        range("shear", self.deshear, -1.0, 1.0)?;
        range("plate score", self.plate_score, 0.01, 0.99)?;
        range("NMS IoU", self.nms_iou, 0.05, 0.95)?;
        if !self.roi.is_empty() {
            if !(3..=32).contains(&self.roi.len()) {
                bail!("the detection region needs between 3 and 32 points");
            }
            if self.roi.iter().flatten().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v)) {
                bail!("the detection region must stay inside the picture");
            }
            let [x0, y0, x1, y1] = self.roi_extent();
            if x1 - x0 < 0.05 || y1 - y0 < 0.05 {
                bail!("the detection region is too small: make it at least 5% of the picture each way");
            }
        }
        if !(1..=20).contains(&self.max_plates) {
            bail!("max plates must be between 1 and 20");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BBox {
    fn w(&self) -> f32 {
        self.x2 - self.x1
    }
    fn h(&self) -> f32 {
        self.y2 - self.y1
    }
    fn area(&self) -> f32 {
        self.w().max(0.0) * self.h().max(0.0)
    }
    fn iou(&self, o: &BBox) -> f32 {
        let ix = (self.x2.min(o.x2) - self.x1.max(o.x1)).max(0.0);
        let iy = (self.y2.min(o.y2) - self.y1.max(o.y1)).max(0.0);
        let union = self.area() + o.area() - ix * iy;
        if union <= 0.0 { 0.0 } else { ix * iy / union }
    }
    fn expand(&self, fx: f32, fy: f32) -> BBox {
        BBox { x1: self.x1 - self.w() * fx, y1: self.y1 - self.h() * fy, x2: self.x2 + self.w() * fx, y2: self.y2 + self.h() * fy }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct RegionRead {
    pub classifier: u32,
    pub group: &'static str,
    /// The region id the classifier chose, e.g. "1125" (Malaysia) or "9999" (not this group).
    pub label: String,
    pub confidence: f32,
}

#[derive(Clone, Serialize)]
pub struct PlateRead {
    pub plate: String,
    /// Mean over the CTC steps of the winning probability, 0-1.
    pub confidence: f32,
    pub det_score: f32,
    #[serde(rename = "box")]
    pub bbox: BBox,
    /// TL, BL, BR, TR in image coordinates when rec_71 rectified the plate.
    pub corners: Option<[f32; 8]>,
    pub ocr_model: String,
    pub strategy: String,
    /// The group that routed this plate, if any.
    pub region: Option<RegionRead>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub regions: Vec<RegionRead>,
    /// The exact 96×48 image fed to the CRNN.
    #[serde(skip)]
    pub ocr_input: RgbImage,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Timings {
    pub det_prep_ms: f64,
    pub det_infer_ms: f64,
    pub rectify_ms: f64,
    pub region_ms: f64,
    pub ocr_ms: f64,
    /// Decoded image in, plates out.
    pub pipeline_ms: f64,
}

pub struct FrameResult {
    /// Most confident detection first.
    pub plates: Vec<PlateRead>,
    pub timings: Timings,
}

struct Detector {
    session: Session,
    input: String,
    size: usize,
}

struct RegionClassifier {
    net: Net,
    id: u32,
    group: &'static str,
    labels: Vec<String>,
    crnn: Option<u32>,
    /// Only classifiers with a 9999 "other" class can reject a plate, so only they route.
    can_reject: bool,
}

struct Crnn {
    id: u32,
    net: Net,
}

/// One set of loaded models. Not shareable across threads: use one per worker.
pub struct Pipeline {
    cfg: PipelineConfig,
    detector: Detector,
    corners: Option<Net>,
    regions: Vec<RegionClassifier>,
    crnns: Vec<Crnn>,
}

fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn rec(weights: &Path, id: u32) -> PathBuf {
    weights.join(format!("rec_{id:02}.bin"))
}

impl Detector {
    fn load(weights: &Path, size: u32, threads: usize, backend: DetectorBackend) -> Result<Self> {
        let id = match size {
            320 => DET_320,
            640 => DET_640,
            other => bail!("--detector must be 320 or 640, not {other}"),
        };
        let path = rec(weights, id);
        if !path.is_file() {
            bail!("detector {} not found (set --weights or LPR_WEIGHTS to the rec_NN.bin directory)", path.display());
        }
        let mut builder = Session::builder()
            .and_then(|b| b.with_intra_threads(threads).map_err(Into::into))
            .map_err(|e| anyhow!("loading detector {}: {e}", path.display()))?;
        if backend == DetectorBackend::Openvino {
            builder = with_openvino(builder, threads)?;
        }
        let session = builder.commit_from_file(&path).map_err(|e| anyhow!("loading detector {}: {e}", path.display()))?;
        let input = session.inputs().first().map(|i| i.name().to_string()).context("detector has no input")?;
        Ok(Detector { session, input, size: size as usize })
    }

    /// Returns the raw YOLOv8 output (4 + classes rows of `anchors` values) and the anchor count.
    fn infer(&mut self, x: Vec<f32>) -> Result<(Vec<f32>, usize)> {
        let tensor = Tensor::from_array(([1usize, 3, self.size, self.size], x)).map_err(|e| anyhow!("detector input: {e}"))?;
        let outputs = self.session.run(ort::inputs![self.input.as_str() => tensor]).map_err(|e| anyhow!("detector run: {e}"))?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(|e| anyhow!("detector output: {e}"))?;
        let anchors = *shape.last().context("detector output has no shape")? as usize;
        if anchors == 0 || data.len() < 5 * anchors {
            bail!("unexpected detector output shape {shape:?}");
        }
        Ok((data.to_vec(), anchors))
    }
}

/// Registers the OpenVINO provider, failing loudly instead of quietly falling back to the CPU
/// kernels: a benchmark that silently measured the wrong backend would be worse than an error.
#[cfg(feature = "openvino")]
fn with_openvino(builder: ort::session::builder::SessionBuilder, threads: usize) -> Result<ort::session::builder::SessionBuilder> {
    let provider = ort::ep::OpenVINO::default().with_device_type("CPU").with_num_threads(threads).build().error_on_failure();
    builder.with_execution_providers([provider]).map_err(|e| anyhow!("the OpenVINO provider did not load (are the ONNX Runtime and OpenVINO libraries on the library path?): {e}"))
}

#[cfg(not(feature = "openvino"))]
fn with_openvino(_: ort::session::builder::SessionBuilder, _: usize) -> Result<ort::session::builder::SessionBuilder> {
    bail!("this binary was built without OpenVINO; build it with scripts/build-openvino-ubuntu.sh (cargo feature `openvino`)")
}

impl Pipeline {
    pub fn load(cfg: &PipelineConfig) -> Result<Self> {
        let w = &cfg.weights;
        let detector = Detector::load(w, cfg.detector, cfg.threads, cfg.detector_backend)?;
        cfg.validate()?;
        let corners = if cfg.rectify { Some(Net::load(&rec(w, CORNERS), cfg.threads)?) } else { None };

        let mut regions = Vec::new();
        if cfg.region || cfg.ocr == OcrMode::Routed {
            for (group, classifier, id_list, crnn) in REGION_GROUPS.into_iter().filter(|g| cfg.region_classifiers.contains(&g.1)) {
                let text = std::fs::read(rec(w, id_list)).with_context(|| format!("region id list rec_{id_list}"))?;
                let labels: Vec<String> =
                    String::from_utf8_lossy(&text).lines().filter(|l| !l.trim().is_empty()).map(|l| l.trim_end().to_string()).collect();
                let can_reject = labels.iter().any(|l| l.starts_with("9999"));
                regions.push(RegionClassifier { net: Net::load(&rec(w, classifier), cfg.threads)?, id: classifier, group, labels, crnn, can_reject });
            }
        }

        let mut ids: Vec<u32> = match cfg.ocr {
            OcrMode::Pinned => vec![cfg.ocr_model],
            // Routing can land on any group's CRNN, and falls back to the 0-9A-Z ensemble.
            OcrMode::Routed => cfg.ensemble_models.iter().copied().chain(regions.iter().filter_map(|r| r.crnn)).collect(),
            OcrMode::Ensemble => cfg.ensemble_models.clone(),
        };
        ids.sort_unstable();
        ids.dedup();
        let mut crnns = Vec::new();
        for id in ids {
            let net = Net::load(&rec(w, id), cfg.threads)?;
            let classes = net.output_shape.last().copied().unwrap_or(0);
            if alphabet(classes).is_none() {
                bail!("rec_{id} has {classes} classes; only the 37- and 38-class alphabets are known");
            }
            crnns.push(Crnn { id, net });
        }
        Ok(Pipeline { cfg: cfg.clone(), detector, corners, regions, crnns })
    }

    pub fn process(&mut self, img: &RgbImage) -> Result<FrameResult> {
        let start = Instant::now();
        let mut timings = Timings::default();

        // 1. Detect: letterbox as in read_plate.py (longest side to `size`, top-left on grey 114).
        // With a detection region the detector sees only its bounding box, so a plate fills more
        // of the detector's input; everything after detection works on the full frame.
        let t = Instant::now();
        let size = self.detector.size;
        let [rx0, ry0, rx1, ry1] = self.cfg.roi_bounds(img.width, img.height);
        let cropped;
        let view = if (rx1 - rx0, ry1 - ry0) == (img.width, img.height) {
            img
        } else {
            cropped = img.crop(rx0 as f32, ry0 as f32, rx1 as f32, ry1 as f32);
            &cropped
        };
        let ratio = size as f32 / view.width.max(view.height) as f32;
        let new_w = ((view.width as f32 * ratio).round() as usize).clamp(1, size);
        let new_h = ((view.height as f32 * ratio).round() as usize).clamp(1, size);
        let small = view.resize(new_w, new_h, 2.0);
        let mut canvas = RgbImage::filled(size, size, 114);
        for y in 0..new_h {
            canvas.data[y * size * 3..(y * size + new_w) * 3].copy_from_slice(&small.data[y * new_w * 3..(y + 1) * new_w * 3]);
        }
        let x = canvas.to_tensor(1.0 / 255.0, 0.0, false);
        timings.det_prep_ms = ms_since(t);

        let t = Instant::now();
        let (out, anchors) = self.detector.infer(x)?;
        let (ox, oy) = (rx0 as f32, ry0 as f32);
        let detections: Vec<(BBox, f32)> = plate_nms(&out, anchors, ratio, self.cfg.plate_score, self.cfg.nms_iou, self.cfg.max_plates)
            .into_iter()
            .map(|(b, score)| (BBox { x1: b.x1 + ox, y1: b.y1 + oy, x2: b.x2 + ox, y2: b.y2 + oy }, score))
            .filter(|(b, _)| self.cfg.roi_contains((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0, img.width, img.height))
            .collect();
        timings.det_infer_ms = ms_since(t);

        // 2. Per plate: rectify, classify the region if asked, read.
        let mut plates = Vec::with_capacity(detections.len());
        for (bbox, det_score) in detections {
            let t = Instant::now();
            let mut corners = None;
            if let Some(net) = self.corners.as_mut() {
                corners = find_corners(net, img, &bbox, self.cfg.corner_margin_x, self.cfg.corner_margin_y)?;
            }
            let ocr_input = match &corners {
                Some(q) => warp(img, q, self.cfg.crop_pad, self.cfg.deshear),
                None => plain_crop(img, &bbox, self.cfg.crop_pad, self.cfg.deshear),
            };
            timings.rectify_ms += ms_since(t);

            let t = Instant::now();
            let mut regions = Vec::new();
            let mut routed: Option<(RegionRead, Option<u32>)> = None;
            if !self.regions.is_empty() {
                let raw = ocr_input.to_tensor(1.0, 0.0, true);
                for rc in self.regions.iter_mut() {
                    let p = softmax_if_logits(rc.net.run(&raw)?, 1);
                    let best = argmax(&p);
                    let read = RegionRead {
                        classifier: rc.id,
                        group: rc.group,
                        label: rc.labels.get(best).cloned().unwrap_or_else(|| "?".into()),
                        confidence: p[best],
                    };
                    let routes = rc.can_reject && !read.label.starts_with("9999") && read.confidence >= 0.5;
                    if routes && routed.as_ref().is_none_or(|(r, _)| read.confidence > r.confidence) {
                        routed = Some((read.clone(), rc.crnn));
                    }
                    regions.push(read);
                }
            }
            timings.region_ms += ms_since(t);

            let t = Instant::now();
            let tensor = ocr_input.to_tensor(1.0 / 255.0, 0.0, true);
            let (id, text, confidence, strategy) = match self.cfg.ocr {
                OcrMode::Pinned => {
                    let (text, conf) = self.read(self.cfg.ocr_model, &tensor)?;
                    (self.cfg.ocr_model, text, conf, format!("pinned rec_{}", self.cfg.ocr_model))
                }
                OcrMode::Routed => match &routed {
                    Some((r, Some(crnn))) => {
                        let (text, conf) = self.read(*crnn, &tensor)?;
                        (*crnn, text, conf, format!("routed {} -> rec_{crnn}", r.group))
                    }
                    Some((r, None)) => self.ensemble(&tensor, format!("routed {} (no CRNN) -> max", r.group))?,
                    None => self.ensemble(&tensor, "unrouted -> max".into())?,
                },
                OcrMode::Ensemble => self.ensemble(&tensor, "ensemble max".into())?,
            };
            timings.ocr_ms += ms_since(t);

            plates.push(PlateRead {
                plate: text,
                confidence,
                det_score,
                bbox,
                corners,
                ocr_model: format!("rec_{id}"),
                strategy,
                region: routed.map(|(r, _)| r),
                regions,
                ocr_input,
            });
        }
        timings.pipeline_ms = ms_since(start);
        Ok(FrameResult { plates, timings })
    }

    fn read(&mut self, id: u32, tensor: &[f32]) -> Result<(String, f32)> {
        let crnn = self.crnns.iter_mut().find(|c| c.id == id).with_context(|| format!("rec_{id} is not loaded"))?;
        let classes = crnn.net.output_shape.last().copied().unwrap_or(0);
        let p = softmax_if_logits(crnn.net.run(tensor)?, classes);
        Ok(ctc_greedy(&p, classes))
    }

    /// Every loaded 0-9A-Z CRNN, most confident read wins.
    fn ensemble(&mut self, tensor: &[f32], strategy: String) -> Result<(u32, String, f32, String)> {
        let mut best: Option<(u32, String, f32)> = None;
        for id in self.cfg.ensemble_models.clone() {
            let (text, conf) = self.read(id, tensor)?;
            if best.as_ref().is_none_or(|b| conf > b.2) {
                best = Some((id, text, conf));
            }
        }
        let (id, text, conf) = best.context("no 0-9A-Z CRNN is loaded")?;
        Ok((id, text, conf, strategy))
    }
}

/// Decodes Ultralytics YOLOv8 output rows (cx, cy, w, h, then one score per class; class 0 =
/// plate) into image-space boxes, with greedy NMS.
fn plate_nms(out: &[f32], n: usize, ratio: f32, min_score: f32, max_iou: f32, max_plates: usize) -> Vec<(BBox, f32)> {
    let mut candidates: Vec<(BBox, f32)> = (0..n)
        .filter(|&i| out[4 * n + i] >= min_score)
        .map(|i| {
            let (cx, cy, w, h) = (out[i], out[n + i], out[2 * n + i], out[3 * n + i]);
            let b = BBox { x1: (cx - w / 2.0) / ratio, y1: (cy - h / 2.0) / ratio, x2: (cx + w / 2.0) / ratio, y2: (cy + h / 2.0) / ratio };
            (b, out[4 * n + i])
        })
        .collect();
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut keep: Vec<(BBox, f32)> = Vec::new();
    for c in candidates {
        if keep.len() >= max_plates {
            break;
        }
        if keep.iter().all(|k| k.0.iou(&c.0) <= max_iou) {
            keep.push(c);
        }
    }
    keep
}

/// rec_71 corner keypoints. Input: the plate box expanded by a context margin (0.2·w, 0.5·h by
/// default; with the tight detector box the model is much less accurate), 96×48 BGR in [-1, 1].
/// Output: 8 sigmoids = TL, BL, BR, TR as fractions of that context crop.
fn find_corners(net: &mut Net, img: &RgbImage, b: &BBox, margin_x: f32, margin_y: f32) -> Result<Option<[f32; 8]>> {
    let ctx = b.expand(margin_x, margin_y);
    let crop = img.crop(ctx.x1, ctx.y1, ctx.x2, ctx.y2).resize(PLATE_W, PLATE_H, 0.0);
    let p = net.run(&crop.to_tensor(1.0 / 127.5, -1.0, true))?;
    if p.len() < 8 {
        return Ok(None);
    }
    // Pillow rounds the crop box, so map back through the rounded box.
    let cx1 = round_half_even(ctx.x1) as f32;
    let cy1 = round_half_even(ctx.y1) as f32;
    let cw = round_half_even(ctx.x2) as f32 - cx1;
    let ch = round_half_even(ctx.y2) as f32 - cy1;
    let mut q = [0f32; 8];
    for i in 0..8 {
        q[i] = if i % 2 == 0 { cx1 + p[i] * cw } else { cy1 + p[i] * ch };
    }
    Ok(plausible(&q, b).then_some(q))
}

/// Rejects degenerate quads: must be convex and cover a sensible share of the detector box.
fn plausible(q: &[f32; 8], b: &BBox) -> bool {
    let pt = |i: usize| (q[i % 4 * 2], q[i % 4 * 2 + 1]);
    let mut sign = 0;
    for i in 0..4 {
        let (a, bb, c) = (pt(i), pt(i + 1), pt(i + 2));
        let cross = (bb.0 - a.0) * (c.1 - bb.1) - (bb.1 - a.1) * (c.0 - bb.0);
        let s = if cross > 0.0 { 1 } else if cross < 0.0 { -1 } else { return false };
        if sign == 0 {
            sign = s;
        } else if s != sign {
            return false;
        }
    }
    let area = (0..4).map(|i| pt(i).0 * pt(i + 1).1 - pt(i + 1).0 * pt(i).1).sum::<f32>().abs() / 2.0;
    area > 0.25 * b.area() && area < 4.0 * b.area()
}

/// Warps the quad (grown by `pad` around its centre) to 192×96, applies the optional shear, then
/// scales down to the 96×48 CRNN input.
fn warp(img: &RgbImage, quad: &[f32; 8], pad: f32, deshear: f32) -> RgbImage {
    let cx = (quad[0] + quad[2] + quad[4] + quad[6]) / 4.0;
    let cy = (quad[1] + quad[3] + quad[5] + quad[7]) / 4.0;
    let mut q = [0f32; 8];
    for i in 0..8 {
        let c = if i % 2 == 0 { cx } else { cy };
        q[i] = c + (quad[i] - c) * (1.0 + pad);
    }
    img.quad(PLATE_W * 2, PLATE_H * 2, &q).deshear(deshear).resize(PLATE_W, PLATE_H, 0.0)
}

/// Port of read_plate.crop_plate: the padded detector box, optionally sheared, resized to 96×48.
fn plain_crop(img: &RgbImage, b: &BBox, pad: f32, deshear: f32) -> RgbImage {
    let (px, py) = (b.w() * pad, b.h() * pad);
    if deshear == 0.0 {
        return img
            .crop((b.x1 - px).max(0.0), (b.y1 - py).max(0.0), (b.x2 + px).min(img.width as f32), (b.y2 + py).min(img.height as f32))
            .resize(PLATE_W, PLATE_H, 0.0);
    }
    // Shear a larger region so no edge pixels are pulled in from outside, then trim.
    let (mx, my) = (b.w() * 0.1, b.h() * 0.3);
    let big = img.crop(b.x1 - mx, b.y1 - my, b.x2 + mx, b.y2 + my).deshear(deshear);
    let (bw, bh) = (big.width as f32, big.height as f32);
    big.crop((mx - px).max(0.0), (my - py).max(0.0), (bw - mx + px).min(bw), (bh - my + py).min(bh)).resize(PLATE_W, PLATE_H, 0.0)
}

/// Known alphabets by class count (including the CTC blank). C=38 adds a group separator.
fn alphabet(classes: usize) -> Option<&'static str> {
    match classes {
        37 => Some(ALNUM),
        38 => Some("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "),
        _ => None,
    }
}

fn argmax(p: &[f32]) -> usize {
    p.iter().enumerate().fold(0, |best, (i, &v)| if v > p[best] { i } else { best })
}

/// Softmaxes each row of `classes` values unless the rows already sum to 1 (some models emit logits).
/// `classes` = 1 treats the whole vector as one row.
fn softmax_if_logits(mut p: Vec<f32>, classes: usize) -> Vec<f32> {
    let c = if classes <= 1 { p.len().max(1) } else { classes };
    if p.chunks(c).all(|row| (row.iter().sum::<f32>() - 1.0).abs() <= 1e-3) {
        return p;
    }
    for row in p.chunks_mut(c) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f64;
        for v in row.iter_mut() {
            let e = ((*v - max) as f64).exp();
            *v = e as f32;
            sum += e;
        }
        row.iter_mut().for_each(|v| *v = (*v as f64 / sum) as f32);
    }
    p
}

/// Greedy CTC, blank = last index. Confidence = mean over all steps of the winning probability
/// (matches read_plate.decode).
fn ctc_greedy(p: &[f32], classes: usize) -> (String, f32) {
    let alphabet = alphabet(classes).unwrap_or(ALNUM).as_bytes();
    let blank = classes - 1;
    let mut text = String::new();
    let mut prev = usize::MAX;
    let mut sum = 0f32;
    let steps = p.len() / classes;
    for row in p.chunks_exact(classes) {
        let best = argmax(row);
        sum += row[best];
        if best != blank && best != prev {
            text.push(alphabet.get(best).map_or('?', |&b| b as char));
        }
        prev = best;
    }
    (text.trim().to_string(), if steps == 0 { 0.0 } else { sum / steps as f32 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_hot(steps: &[usize], classes: usize) -> Vec<f32> {
        let mut p = vec![0f32; steps.len() * classes];
        steps.iter().enumerate().for_each(|(t, &k)| p[t * classes + k] = 1.0);
        p
    }

    #[test]
    fn ctc_collapses_repeats_and_drops_blanks() {
        // B B _ R _ 4 4 _ 4  ->  BR44
        let p = one_hot(&[11, 11, 36, 27, 36, 4, 4, 36, 4], 37);
        assert_eq!(ctc_greedy(&p, 37), ("BR44".to_string(), 1.0));
    }

    #[test]
    fn logits_are_softmaxed_and_probabilities_left_alone() {
        let probs = softmax_if_logits(vec![0.25, 0.75, 0.5, 0.5], 2);
        assert_eq!(probs, vec![0.25, 0.75, 0.5, 0.5]);
        let p = softmax_if_logits(vec![0.0, 0.0, 2.0, 0.0], 2);
        assert!((p[0] - 0.5).abs() < 1e-6 && (p[2] - 0.880797).abs() < 1e-5);
    }

    #[test]
    fn nms_keeps_the_best_of_overlapping_plates() {
        // Two anchors on the same plate, one elsewhere; rows are cx, cy, w, h, plate, vehicle.
        let n = 3;
        let mut out = vec![0f32; 6 * n];
        let boxes = [(100.0, 50.0, 40.0, 20.0, 0.9), (102.0, 51.0, 40.0, 20.0, 0.8), (250.0, 200.0, 40.0, 20.0, 0.6)];
        for (i, b) in boxes.iter().enumerate() {
            out[i] = b.0;
            out[n + i] = b.1;
            out[2 * n + i] = b.2;
            out[3 * n + i] = b.3;
            out[4 * n + i] = b.4;
        }
        let kept = plate_nms(&out, n, 0.5, 0.25, 0.45, 6);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].1, 0.9);
        assert_eq!(kept[0].0.x1, 160.0); // (100 - 20) / 0.5
    }

    #[test]
    fn a_bow_tie_quad_is_rejected() {
        let b = BBox { x1: 0.0, y1: 0.0, x2: 100.0, y2: 50.0 };
        assert!(plausible(&[0.0, 0.0, 0.0, 50.0, 100.0, 50.0, 100.0, 0.0], &b));
        assert!(!plausible(&[0.0, 0.0, 100.0, 50.0, 0.0, 50.0, 100.0, 0.0], &b));
    }

    /// End to end on the repository's samples, against the reads the Python PoC and the Android
    /// app agree on. Skipped when the weights are not next to the crate.
    #[test]
    fn reads_the_three_malaysian_samples() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        if !root.join("weights/rec_72.bin").is_file() {
            eprintln!("weights not found, skipping");
            return;
        }
        let weights = root.join("weights");
        let cfg = PipelineConfig { weights: weights.clone(), ..PipelineConfig::default() };
        let mut pipeline = Pipeline::load(&cfg).unwrap();
        for plate in ["BRL4104", "EV232", "PUTRAJAYA541"] {
            let bytes = std::fs::read(root.join(format!("sample_data/{plate}.jpeg"))).unwrap();
            let result = pipeline.process(&RgbImage::decode(&bytes).unwrap()).unwrap();
            let best = result.plates.first().expect("a plate is detected");
            assert_eq!(best.plate, plate);
            assert!(best.confidence > 0.9, "{plate}: confidence {}", best.confidence);
            assert!(best.corners.is_some(), "{plate}: rectified");
        }

        // A region around BRL4104's plate still reads it (the detector sees a crop, the boxes come
        // back in frame coordinates); a region elsewhere reads nothing.
        let bytes = std::fs::read(root.join("sample_data/BRL4104.jpeg")).unwrap();
        let frame = RgbImage::decode(&bytes).unwrap();
        let around = PipelineConfig { weights: weights.clone(), roi: vec![[0.3, 0.25], [0.8, 0.25], [0.8, 0.7], [0.3, 0.7]], ..PipelineConfig::default() };
        let read = Pipeline::load(&around).unwrap().process(&frame).unwrap();
        assert_eq!(read.plates[0].plate, "BRL4104");
        assert!((read.plates[0].bbox.x1 - 946.0).abs() < 8.0 && (read.plates[0].bbox.y1 - 447.0).abs() < 8.0, "box in frame coordinates");
        let elsewhere = PipelineConfig { weights: weights.clone(), roi: vec![[0.0, 0.0], [0.3, 0.0], [0.3, 0.3], [0.0, 0.3]], ..PipelineConfig::default() };
        assert!(Pipeline::load(&elsewhere).unwrap().process(&frame).unwrap().plates.is_empty());

        // The PoC's skew correction: no corner model, a 0.3 shear and 2% padding. It reads
        // BRL4104 but turns PUTRAJAYA541 into PUTRAJAYA1541, exactly as read_plate.py does.
        let poc = PipelineConfig { weights, rectify: false, deshear: 0.3, crop_pad: 0.02, ..PipelineConfig::default() };
        let mut pipeline = Pipeline::load(&poc).unwrap();
        for (file, expected) in [("BRL4104", "BRL4104"), ("PUTRAJAYA541", "PUTRAJAYA1541")] {
            let bytes = std::fs::read(root.join(format!("sample_data/{file}.jpeg"))).unwrap();
            let result = pipeline.process(&RgbImage::decode(&bytes).unwrap()).unwrap();
            assert_eq!(result.plates[0].plate, expected);
            assert!(result.plates[0].corners.is_none());
        }
    }

    #[test]
    fn the_detection_region_limits_where_the_detector_looks_and_what_is_read() {
        // A trapezoid lane in the lower middle of a 1920x1080 frame.
        let cfg = PipelineConfig { roi: vec![[0.4, 0.3], [0.7, 0.3], [0.9, 0.8], [0.2, 0.8]], ..PipelineConfig::default() };
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.roi_bounds(1920, 1080), [384, 324, 1728, 864]);
        assert!(cfg.roi_contains(1000.0, 600.0, 1920, 1080));
        assert!(!cfg.roi_contains(450.0, 400.0, 1920, 1080), "inside the bounding box, outside the trapezoid");
        assert!(!cfg.roi_contains(100.0, 100.0, 1920, 1080));
        let whole = PipelineConfig::default();
        assert_eq!(whole.roi_bounds(1920, 1080), [0, 0, 1920, 1080]);
        assert!(whole.roi_contains(5.0, 5.0, 1920, 1080));
        assert!(PipelineConfig { roi: vec![[0.1, 0.1], [0.2, 0.2]], ..whole.clone() }.validate().is_err(), "two points");
        assert!(PipelineConfig { roi: vec![[0.1, 0.1], [1.2, 0.2], [0.5, 0.9]], ..whole.clone() }.validate().is_err(), "outside");
        assert!(PipelineConfig { roi: vec![[0.1, 0.1], [0.12, 0.1], [0.11, 0.9]], ..whole }.validate().is_err(), "a sliver");
    }

    #[test]
    fn settings_a_person_can_break_are_refused() {
        let ok = PipelineConfig::default();
        assert!(ok.validate().is_ok());
        assert!(PipelineConfig { ocr_model: 61, ..ok.clone() }.validate().is_err(), "Thai alphabet is unknown");
        assert!(PipelineConfig { ensemble_models: vec![55], ..ok.clone() }.validate().is_err());
        assert!(PipelineConfig { deshear: 3.0, ..ok.clone() }.validate().is_err());
        assert!(PipelineConfig { detector: 512, ..ok.clone() }.validate().is_err());
        assert!(PipelineConfig { ocr: OcrMode::Routed, region_classifiers: vec![], ..ok }.validate().is_err());
    }
}
