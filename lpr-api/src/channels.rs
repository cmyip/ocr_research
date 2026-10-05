//! RTSP channels: each one decodes a stream with ffmpeg, runs its own pipeline (its own choice of
//! models and skew correction) on the frames, and raises an event when a plate is read steadily.
//!
//! One worker thread per channel owns the models; a reader thread keeps only the newest frame,
//! so a slow pipeline drops frames instead of falling behind the camera.
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context as _, Result};
use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::image::RgbImage;
use crate::payload::{now_iso, ProviderEvent};
use crate::pipeline::{Pipeline, PipelineConfig, PlateRead};

const MASK: &str = "***";
const STALL: Duration = Duration::from_secs(15);
const EVENTS_KEPT: usize = 200;
/// Reads of one plate count towards "steady" only if they are this close together.
const STEADY_GAP_S: f64 = 3.0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Transport {
    #[default]
    Tcp,
    Udp,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChannelConfig {
    pub id: u32,
    pub name: String,
    /// rtsp:// or rtsps:// URL of the stream.
    pub url: String,
    pub enabled: bool,
    pub transport: Transport,
    /// Frames per second taken from the stream for reading.
    #[serde(serialize_with = "crate::pipeline::short_f32")]
    pub fps: f32,
    /// The camera id events carry; it must match a lane's camera id for ParkingDashboard to decide.
    pub camera_id: String,
    /// Post each event to the HTTP URL set in the web UI's settings.
    pub trigger: bool,
    /// Reads below this confidence (0-1) are ignored.
    #[serde(serialize_with = "crate::pipeline::short_f32")]
    pub min_confidence: f32,
    /// A plate becomes an event once it is read in this many frames.
    pub stable_frames: u32,
    /// The same plate does not raise another event for this long.
    pub dedupe_seconds: u32,
    /// Skip frames in which nothing has changed, instead of running the detector on them.
    pub skip_unchanged: bool,
    /// How much of the detection region must change, in percent, for a frame to count as changed.
    #[serde(serialize_with = "crate::pipeline::short_f32")]
    pub change_percent: f32,
    /// Models, detection region and skew correction.
    pub pipeline: PipelineConfig,
}

impl Default for ChannelConfig {
    fn default() -> Self {
        ChannelConfig {
            id: 0,
            name: String::new(),
            url: String::new(),
            enabled: true,
            transport: Transport::Tcp,
            fps: 5.0,
            camera_id: String::new(),
            trigger: false,
            min_confidence: 0.9,
            stable_frames: 2,
            dedupe_seconds: 30,
            skip_unchanged: true,
            change_percent: 1.0,
            pipeline: PipelineConfig::default(),
        }
    }
}

/// The HTTP URL events are posted to, as a person sets it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TargetSettings {
    /// A ParkingDashboard base URL (http://localhost:5200 posts to its /api/alpr/events), or the
    /// full URL of any endpoint that takes the camera payload. Empty turns posting off.
    pub url: String,
    /// Confirms that a URL on another machine is intended: reads sent to a real site are real.
    pub allow_remote: bool,
}

/// What every channel shares: where the models are, how to reach ffmpeg and the event target.
pub struct Context {
    pub weights: PathBuf,
    pub threads: usize,
    pub detector_backend: crate::pipeline::DetectorBackend,
    pub ffmpeg: String,
    /// Accept local video files as sources, for testing without a camera.
    pub allow_files: bool,
    /// Where events are posted: what was entered, and the URL it resolves to when it is usable.
    /// Changed from the web UI while channels run.
    pub target: RwLock<(TargetSettings, Option<String>)>,
    pub http: reqwest::Client,
    pub runtime: tokio::runtime::Handle,
}

#[derive(Clone, Serialize)]
pub struct PlateView {
    plate: String,
    confidence: f32,
    #[serde(rename = "box")]
    bbox: [f32; 4],
    corners: Option<[f32; 8]>,
    ocr_model: String,
    strategy: String,
    region: Option<String>,
}

#[derive(Clone, Default, Serialize)]
pub struct Status {
    /// stopped, starting, connecting, running or error.
    state: &'static str,
    message: Option<String>,
    /// Frames the models read.
    frames: u64,
    /// Frames skipped because nothing had changed.
    unchanged: u64,
    /// Frames skipped because reading was slower than the stream.
    dropped: u64,
    fps: f32,
    process_ms: f32,
    width: usize,
    height: usize,
    /// Plates in the frame the preview shows.
    plates: Vec<PlateView>,
    last_plate: Option<String>,
    last_confidence: Option<f32>,
    last_read_at: Option<String>,
    /// The 96×48 image the OCR last read, base64 JPEG: what the skew correction produced.
    crop: Option<String>,
    events: u64,
}

#[derive(Clone, Serialize)]
pub struct Event {
    pub id: u64,
    pub channel_id: u32,
    pub channel: String,
    pub camera_id: String,
    pub plate: String,
    pub confidence: f32,
    pub at: String,
    pub ocr_model: String,
    pub crop: String,
    /// Outcome of posting to the target, once known.
    pub delivery: Option<Value>,
}

#[derive(Default)]
pub struct EventLog {
    inner: Mutex<(VecDeque<Event>, u64)>,
}

impl EventLog {
    fn push(&self, mut event: Event) -> u64 {
        let mut g = self.inner.lock().unwrap();
        g.1 += 1;
        event.id = g.1;
        g.0.push_back(event);
        while g.0.len() > EVENTS_KEPT {
            g.0.pop_front();
        }
        g.1
    }

    fn set_delivery(&self, id: u64, delivery: Value) {
        if let Some(e) = self.inner.lock().unwrap().0.iter_mut().find(|e| e.id == id) {
            e.delivery = Some(delivery);
        }
    }

    /// Newest first.
    pub fn recent(&self, limit: usize) -> Vec<Event> {
        self.inner.lock().unwrap().0.iter().rev().take(limit).cloned().collect()
    }
}

/// Decides when a read becomes an event: seen in enough frames, and not just reported.
struct Tracker {
    min_confidence: f32,
    stable_frames: u32,
    dedupe_s: f64,
    seen: HashMap<String, (u32, f64)>,
    emitted: HashMap<String, f64>,
}

impl Tracker {
    fn new(cfg: &ChannelConfig) -> Self {
        Tracker {
            min_confidence: cfg.min_confidence,
            stable_frames: cfg.stable_frames.max(1),
            dedupe_s: cfg.dedupe_seconds as f64,
            seen: HashMap::new(),
            emitted: HashMap::new(),
        }
    }

    /// `now` is seconds on any steady clock. Returns true when this read should raise an event.
    fn observe(&mut self, plate: &str, confidence: f32, now: f64) -> bool {
        if plate.len() < 2 || confidence < self.min_confidence {
            return false;
        }
        self.seen.retain(|_, (_, last)| now - *last <= STEADY_GAP_S);
        let dedupe = self.dedupe_s;
        self.emitted.retain(|_, at| now - *at < dedupe);
        let entry = self.seen.entry(plate.to_string()).or_insert((0, now));
        entry.0 += 1;
        entry.1 = now;
        if entry.0 < self.stable_frames || self.emitted.contains_key(plate) {
            return false;
        }
        self.emitted.insert(plate.to_string(), now);
        true
    }
}

/// Decides which frames are worth reading. A frame is read when the picture differs from the
/// last frame that was read, for a short while after that (a car that has just stopped still needs
/// several agreeing reads), and once in a while regardless, so the view never goes stale.
struct MotionGate {
    enabled: bool,
    /// Share of the grid's cells that must differ.
    min_fraction: f32,
    reference: Option<Vec<u8>>,
    last_change: f64,
    last_read: f64,
}

impl MotionGate {
    const GRID_W: usize = 64;
    const GRID_H: usize = 36;
    /// Brightness difference (of 255) at which a cell counts as different: above sensor noise
    /// and compression shimmer, far below a vehicle moving in.
    const CELL_DELTA: i16 = 12;
    /// Keep reading this long after the last change.
    const HOLD_S: f64 = 2.0;
    /// Read one frame at least this often.
    const HEARTBEAT_S: f64 = 5.0;

    fn new(cfg: &ChannelConfig) -> Self {
        MotionGate { enabled: cfg.skip_unchanged, min_fraction: cfg.change_percent / 100.0, reference: None, last_change: f64::NEG_INFINITY, last_read: f64::NEG_INFINITY }
    }

    /// Mean brightness of each grid cell over the rectangle, from a sparse sample of its pixels:
    /// a few tens of thousands of reads instead of the two million in a 1080p frame.
    fn signature(img: &RgbImage, [x0, y0, x1, y1]: [usize; 4]) -> Vec<u8> {
        const SAMPLES: usize = 4;
        let (w, h) = ((x1 - x0) as f32, (y1 - y0) as f32);
        let mut sig = Vec::with_capacity(Self::GRID_W * Self::GRID_H);
        for cy in 0..Self::GRID_H {
            for cx in 0..Self::GRID_W {
                let mut sum = 0u32;
                for sy in 0..SAMPLES {
                    let y = (y0 + ((cy as f32 + (sy as f32 + 0.5) / SAMPLES as f32) * h / Self::GRID_H as f32) as usize).min(img.height - 1);
                    for sx in 0..SAMPLES {
                        let x = (x0 + ((cx as f32 + (sx as f32 + 0.5) / SAMPLES as f32) * w / Self::GRID_W as f32) as usize).min(img.width - 1);
                        sum += img.data[(y * img.width + x) * 3 + 1] as u32; // green stands in for brightness
                    }
                }
                sig.push((sum / (SAMPLES * SAMPLES) as u32) as u8);
            }
        }
        sig
    }

    /// `now` is seconds on any steady clock. Returns whether this frame should be read.
    fn should_read(&mut self, img: &RgbImage, bounds: [usize; 4], now: f64) -> bool {
        if !self.enabled {
            return true;
        }
        let sig = Self::signature(img, bounds);
        let changed = match &self.reference {
            Some(reference) => {
                let differing = sig.iter().zip(reference).filter(|(a, b)| (**a as i16 - **b as i16).abs() > Self::CELL_DELTA).count();
                differing >= ((self.min_fraction * sig.len() as f32).ceil() as usize).max(1)
            }
            None => true,
        };
        if changed {
            self.last_change = now;
        }
        let read = changed || now - self.last_change < Self::HOLD_S || now - self.last_read >= Self::HEARTBEAT_S;
        if read {
            // Compared with the last frame that was read, so a slow change adds up until it counts.
            self.reference = Some(sig);
            self.last_read = now;
        }
        read
    }
}

struct Shared {
    status: Mutex<Status>,
    preview: Mutex<Option<Arc<RgbImage>>>,
}

impl Shared {
    fn set_state(&self, state: &'static str, message: Option<String>) {
        let mut s = self.status.lock().unwrap();
        s.state = state;
        s.message = message;
        if state != "running" {
            s.fps = 0.0;
            s.plates.clear();
        }
    }
}

struct Worker {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    shared: Arc<Shared>,
}

impl Worker {
    fn start(cfg: ChannelConfig, ctx: Arc<Context>, events: Arc<EventLog>) -> Worker {
        let stop = Arc::new(AtomicBool::new(false));
        let shared = Arc::new(Shared { status: Mutex::new(Status { state: "starting", ..Status::default() }), preview: Mutex::new(None) });
        let (s, sh) = (stop.clone(), shared.clone());
        let handle = std::thread::Builder::new().name(format!("channel-{}", cfg.id)).spawn(move || run(cfg, ctx, events, sh, s)).ok();
        Worker { stop, handle, shared }
    }

    fn stop(mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Hides the password of a URL, for anything shown to a person or written to a log.
pub fn mask_url(url: &str) -> String {
    match reqwest::Url::parse(url) {
        Ok(mut u) if u.password().is_some() => {
            let _ = u.set_password(Some(MASK));
            u.to_string()
        }
        _ => url.to_string(),
    }
}

/// A URL that came back from the UI still masked gets the stored password put back.
fn unmask_url(submitted: &str, stored: &str) -> String {
    let (Ok(mut new), Ok(old)) = (reqwest::Url::parse(submitted), reqwest::Url::parse(stored)) else {
        return submitted.to_string();
    };
    if new.password() == Some(MASK) && old.password().is_some() {
        let _ = new.set_password(old.password());
        return new.to_string();
    }
    submitted.to_string()
}

fn is_rtsp(url: &str) -> bool {
    reqwest::Url::parse(url).is_ok_and(|u| matches!(u.scheme(), "rtsp" | "rtsps") && u.host_str().is_some_and(|h| !h.is_empty()))
}

fn validate(cfg: &ChannelConfig, allow_files: bool) -> Result<()> {
    if cfg.name.trim().is_empty() {
        bail!("give the channel a name");
    }
    if cfg.name.chars().count() > 60 {
        bail!("the name is too long");
    }
    let url = cfg.url.trim();
    if !is_rtsp(url) {
        let is_file = allow_files && !url.contains("://") && Path::new(url).is_file();
        if !is_file {
            bail!("the stream URL must look like rtsp://host:554/path{}", if allow_files { ", or be the path of a video file on this machine" } else { "" });
        }
    }
    if !(cfg.fps.is_finite() && (0.2..=30.0).contains(&cfg.fps)) {
        bail!("frames per second must be between 0.2 and 30");
    }
    if !(0.0..=1.0).contains(&cfg.min_confidence) {
        bail!("minimum confidence must be between 0 and 1");
    }
    if !(cfg.change_percent.is_finite() && (0.1..=50.0).contains(&cfg.change_percent)) {
        bail!("the share of the picture that must change must be between 0.1% and 50%");
    }
    if !(1..=20).contains(&cfg.stable_frames) {
        bail!("stable frames must be between 1 and 20");
    }
    if cfg.dedupe_seconds > 86_400 {
        bail!("the repeat window cannot exceed a day");
    }
    cfg.pipeline.validate()
}

fn ffmpeg_args(cfg: &ChannelConfig) -> Vec<String> {
    let mut args: Vec<String> = ["-hide_banner", "-loglevel", "error", "-nostdin"].map(String::from).to_vec();
    if is_rtsp(&cfg.url) {
        let transport = if cfg.transport == Transport::Udp { "udp" } else { "tcp" };
        args.extend(["-rtsp_transport", transport, "-fflags", "nobuffer", "-flags", "low_delay"].map(String::from));
    } else {
        // A file plays at its own frame rate and loops, like a camera that never ends.
        args.extend(["-re", "-stream_loop", "-1"].map(String::from));
    }
    args.extend(["-i".to_string(), cfg.url.trim().to_string(), "-an".to_string(), "-vf".to_string(), format!("fps={}", cfg.fps)]);
    // PPM frames carry their own size and need no decoding on this side.
    args.extend(["-f", "image2pipe", "-c:v", "ppm", "pipe:1"].map(String::from));
    args
}

/// Reads one binary PPM (P6, maxval 255) frame; `Ok(None)` at a clean end of stream.
fn read_ppm(r: &mut impl BufRead) -> Result<Option<RgbImage>> {
    fn token(r: &mut impl BufRead) -> Result<Option<String>> {
        let mut out = String::new();
        loop {
            let mut b = [0u8; 1];
            if r.read(&mut b)? == 0 {
                return if out.is_empty() { Ok(None) } else { Err(anyhow!("stream ended inside a frame header")) };
            }
            if b[0].is_ascii_whitespace() {
                if !out.is_empty() {
                    return Ok(Some(out));
                }
            } else {
                out.push(b[0] as char);
                if out.len() > 16 {
                    bail!("not a PPM frame");
                }
            }
        }
    }
    let Some(magic) = token(r)? else { return Ok(None) };
    if magic != "P6" {
        bail!("unexpected frame format {magic:?}");
    }
    let mut dims = [0usize; 3];
    for d in dims.iter_mut() {
        *d = token(r)?.context("stream ended inside a frame header")?.parse().context("bad PPM header")?;
    }
    let [width, height, maxval] = dims;
    if width == 0 || height == 0 || width > 16_384 || height > 16_384 || maxval != 255 {
        bail!("unsupported frame {width}x{height} (maxval {maxval})");
    }
    let mut data = vec![0u8; width * height * 3];
    r.read_exact(&mut data).context("stream ended inside a frame")?;
    Ok(Some(RgbImage { width, height, data }))
}

#[derive(Default)]
struct Slot {
    frame: Option<RgbImage>,
    dropped: u64,
    ended: bool,
}

fn run(cfg: ChannelConfig, ctx: Arc<Context>, events: Arc<EventLog>, shared: Arc<Shared>, stop: Arc<AtomicBool>) {
    let mut pcfg = cfg.pipeline.clone();
    pcfg.weights = ctx.weights.clone();
    pcfg.threads = ctx.threads;
    pcfg.detector_backend = ctx.detector_backend;
    let mut pipeline = match Pipeline::load(&pcfg) {
        Ok(p) => p,
        Err(e) => return shared.set_state("error", Some(format!("models did not load: {e:#}"))),
    };
    let mut tracker = Tracker::new(&cfg);
    let mut gate = MotionGate::new(&cfg);
    let clock = Instant::now();
    let mut backoff = 2u64;
    while !stop.load(Ordering::SeqCst) {
        shared.set_state("connecting", None);
        let frames_before = shared.status.lock().unwrap().frames;
        let reason = stream_once(&cfg, &ctx, &events, &shared, &stop, &mut pipeline, &mut tracker, &mut gate, clock);
        if stop.load(Ordering::SeqCst) {
            break;
        }
        if shared.status.lock().unwrap().frames > frames_before {
            backoff = 2; // it was working: reconnect promptly
        }
        shared.set_state("error", Some(format!("{reason}; reconnecting in {backoff} s")));
        let until = Instant::now() + Duration::from_secs(backoff);
        while Instant::now() < until && !stop.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(100));
        }
        backoff = (backoff * 2).min(15);
    }
    shared.set_state("stopped", None);
}

/// Runs ffmpeg until the stream ends, stalls or the channel is stopped; returns why it ended.
#[allow(clippy::too_many_arguments)]
fn stream_once(
    cfg: &ChannelConfig,
    ctx: &Context,
    events: &Arc<EventLog>,
    shared: &Shared,
    stop: &AtomicBool,
    pipeline: &mut Pipeline,
    tracker: &mut Tracker,
    gate: &mut MotionGate,
    clock: Instant,
) -> String {
    let mut child = match Command::new(&ctx.ffmpeg).args(ffmpeg_args(cfg)).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
        Ok(c) => c,
        Err(e) => return format!("could not start {}: {e}", ctx.ffmpeg),
    };
    let slot = Arc::new((Mutex::new(Slot::default()), Condvar::new()));
    let reader = {
        let slot = slot.clone();
        let stdout = child.stdout.take().expect("piped");
        std::thread::spawn(move || {
            let mut r = BufReader::with_capacity(1 << 20, stdout);
            while let Ok(Some(img)) = read_ppm(&mut r) {
                let mut s = slot.0.lock().unwrap();
                if s.frame.replace(img).is_some() {
                    s.dropped += 1;
                }
                slot.1.notify_one();
            }
            slot.0.lock().unwrap().ended = true;
            slot.1.notify_one();
        })
    };
    let last_error = Arc::new(Mutex::new(String::new()));
    let errors = {
        let (last_error, stderr, url, masked) = (last_error.clone(), child.stderr.take().expect("piped"), cfg.url.trim().to_string(), mask_url(cfg.url.trim()));
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok).filter(|l| !l.trim().is_empty()) {
                *last_error.lock().unwrap() = line.replace(&url, &masked).chars().take(300).collect();
            }
        })
    };

    let mut last_frame = Instant::now();
    let (mut window_start, mut window_frames) = (Instant::now(), 0u32);
    let reason = loop {
        if stop.load(Ordering::SeqCst) {
            break String::new();
        }
        let (frame, ended, dropped) = {
            let guard = slot.0.lock().unwrap();
            let (mut guard, _) = slot.1.wait_timeout_while(guard, Duration::from_millis(250), |s| s.frame.is_none() && !s.ended).unwrap();
            (guard.frame.take(), guard.ended, guard.dropped)
        };
        let Some(img) = frame else {
            if ended {
                let detail = last_error.lock().unwrap().clone();
                // ffmpeg's last line is the reason when it could not connect, and only a hint otherwise.
                let had_frames = shared.status.lock().unwrap().state == "running";
                break match (detail.is_empty(), had_frames) {
                    (true, _) => "the stream ended".to_string(),
                    (false, true) => format!("the stream ended (ffmpeg: {detail})"),
                    (false, false) => detail,
                };
            }
            if last_frame.elapsed() > STALL {
                break format!("no frames for {} s", STALL.as_secs());
            }
            continue;
        };
        last_frame = Instant::now();
        window_frames += 1;

        if !gate.should_read(&img, cfg.pipeline.roi_bounds(img.width, img.height), clock.elapsed().as_secs_f64()) {
            // Nothing moved: the last read still describes the scene, so only the picture is refreshed.
            let mut s = shared.status.lock().unwrap();
            s.state = "running";
            s.unchanged += 1;
            s.dropped = dropped;
            if window_start.elapsed() >= Duration::from_secs(2) {
                s.fps = window_frames as f32 / window_start.elapsed().as_secs_f32();
                (window_start, window_frames) = (Instant::now(), 0);
            }
            drop(s);
            *shared.preview.lock().unwrap() = Some(Arc::new(img));
            continue;
        }

        let t = Instant::now();
        let result = pipeline.process(&img);
        let ms = t.elapsed().as_secs_f32() * 1e3;
        let mut raised: Vec<PlateRead> = Vec::new();
        {
            let mut s = shared.status.lock().unwrap();
            s.state = "running";
            s.frames += 1;
            s.dropped = dropped;
            s.process_ms = if s.process_ms == 0.0 { ms } else { s.process_ms * 0.8 + ms * 0.2 };
            (s.width, s.height) = (img.width, img.height);
            if window_start.elapsed() >= Duration::from_secs(2) {
                s.fps = window_frames as f32 / window_start.elapsed().as_secs_f32();
                (window_start, window_frames) = (Instant::now(), 0);
            }
            match result {
                Ok(r) => {
                    s.message = None;
                    s.plates = r.plates.iter().map(view).collect();
                    if let Some(best) = r.plates.first() {
                        s.last_plate = Some(best.plate.clone());
                        s.last_confidence = Some(best.confidence);
                        s.last_read_at = Some(now_iso());
                        s.crop = best.ocr_input.encode_jpeg(95).ok().map(|j| base64::engine::general_purpose::STANDARD.encode(j));
                    }
                    let now = clock.elapsed().as_secs_f64();
                    raised = r.plates.into_iter().filter(|p| tracker.observe(&p.plate, p.confidence, now)).collect();
                    s.events += raised.len() as u64;
                }
                Err(e) => s.message = Some(format!("reading a frame failed: {e:#}")),
            }
        }
        for plate in &raised {
            raise(cfg, ctx, events, plate, &img);
        }
        *shared.preview.lock().unwrap() = Some(Arc::new(img));
    };
    let _ = child.kill();
    let _ = child.wait();
    let _ = reader.join();
    let _ = errors.join();
    reason
}

fn view(p: &PlateRead) -> PlateView {
    PlateView {
        plate: p.plate.clone(),
        confidence: p.confidence,
        bbox: [p.bbox.x1, p.bbox.y1, p.bbox.x2, p.bbox.y2],
        corners: p.corners,
        ocr_model: p.ocr_model.clone(),
        strategy: p.strategy.clone(),
        region: p.region.as_ref().map(|r| r.label.clone()),
    }
}

/// Records the event and, when the channel triggers, posts it to the target like a camera would.
fn raise(cfg: &ChannelConfig, ctx: &Context, events: &Arc<EventLog>, plate: &PlateRead, frame: &RgbImage) {
    let b64 = |bytes: Vec<u8>| base64::engine::general_purpose::STANDARD.encode(bytes);
    let camera_id = if cfg.camera_id.trim().is_empty() { cfg.name.clone() } else { cfg.camera_id.trim().to_string() };
    let id = events.push(Event {
        id: 0,
        channel_id: cfg.id,
        channel: cfg.name.clone(),
        camera_id: camera_id.clone(),
        plate: plate.plate.clone(),
        confidence: plate.confidence,
        at: now_iso(),
        ocr_model: plate.ocr_model.clone(),
        crop: plate.ocr_input.encode_jpeg(95).map(b64).unwrap_or_default(),
        delivery: None,
    });
    let (true, Some(url)) = (cfg.trigger, ctx.target.read().unwrap().1.clone()) else { return };
    let jpeg = frame.encode_jpeg(85).unwrap_or_default();
    let payload = ProviderEvent::build(Some(plate), frame, &jpeg, &camera_id, None, true);
    let (http, events) = (ctx.http.clone(), events.clone());
    ctx.runtime.spawn(async move {
        let t = Instant::now();
        let mut outcome = match http.post(&url).json(&payload).send().await {
            Ok(resp) => {
                let status = resp.status().as_u16();
                let body = resp.text().await.unwrap_or_default();
                let decision = serde_json::from_str::<Value>(&body).ok().and_then(|v| v.get("gateDecision").cloned());
                json!({ "sent": true, "status": status, "decision": decision })
            }
            Err(e) => json!({ "sent": false, "error": e.to_string() }),
        };
        outcome["ms"] = json!((t.elapsed().as_secs_f64() * 1e3).round());
        events.set_delivery(id, outcome);
    });
}

struct Entry {
    cfg: ChannelConfig,
    worker: Option<Worker>,
}

#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
struct Saved {
    channels: Vec<ChannelConfig>,
    /// Absent in a file written before the URL could be set from the UI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    target: Option<TargetSettings>,
}

pub struct Manager {
    ctx: Arc<Context>,
    path: PathBuf,
    pub events: Arc<EventLog>,
    entries: Mutex<Vec<Entry>>,
}

impl Manager {
    /// Loads the channels file (if there is one) and starts every enabled channel.
    pub fn open(ctx: Context, path: PathBuf) -> Result<Arc<Manager>> {
        let saved: Saved = match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Saved::default(),
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        // A URL saved from the UI wins over the one the server was started with.
        if let Some(saved_target) = saved.target {
            let resolved = match saved_target.url.trim() {
                "" => None,
                url => crate::server::events_url(url, saved_target.allow_remote)
                    .map_err(|e| eprintln!("warning: the saved event URL is not usable, so events are not posted: {e:#}"))
                    .ok(),
            };
            *ctx.target.write().unwrap() = (saved_target, resolved);
        }
        let manager = Arc::new(Manager { ctx: Arc::new(ctx), path, events: Arc::new(EventLog::default()), entries: Mutex::new(Vec::new()) });
        let mut entries = manager.entries.lock().unwrap();
        for cfg in saved.channels {
            let worker = cfg.enabled.then(|| manager.spawn(&cfg));
            entries.push(Entry { cfg, worker });
        }
        drop(entries);
        Ok(manager)
    }

    fn spawn(&self, cfg: &ChannelConfig) -> Worker {
        Worker::start(cfg.clone(), self.ctx.clone(), self.events.clone())
    }

    fn describe(entry: &Entry) -> Value {
        let status = match &entry.worker {
            Some(w) => w.shared.status.lock().unwrap().clone(),
            None => Status { state: "stopped", ..Status::default() },
        };
        let mut config = entry.cfg.clone();
        config.url = mask_url(&config.url);
        json!({ "config": config, "status": status })
    }

    pub fn list(&self) -> Vec<Value> {
        self.entries.lock().unwrap().iter().map(Self::describe).collect()
    }

    fn save(&self, entries: &[Entry]) -> Result<()> {
        let saved = Saved { channels: entries.iter().map(|e| e.cfg.clone()).collect(), target: Some(self.ctx.target.read().unwrap().0.clone()) };
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(&saved)?).with_context(|| format!("writing {}", tmp.display()))?;
        // Stream URLs can carry camera passwords.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600));
        }
        std::fs::rename(&tmp, &self.path).with_context(|| format!("writing {}", self.path.display()))
    }

    /// Creates a channel (`id` None) or replaces one, restarting its worker with the new settings.
    pub fn upsert(&self, id: Option<u32>, mut cfg: ChannelConfig) -> Result<Value> {
        cfg.name = cfg.name.trim().to_string();
        cfg.url = cfg.url.trim().to_string();
        let mut entries = self.entries.lock().unwrap();
        let index = match id {
            Some(id) => Some(entries.iter().position(|e| e.cfg.id == id).with_context(|| format!("channel {id} does not exist"))?),
            None => None,
        };
        if let Some(i) = index {
            cfg.url = unmask_url(&cfg.url, &entries[i].cfg.url);
        }
        validate(&cfg, self.ctx.allow_files)?;
        if entries.iter().enumerate().any(|(i, e)| Some(i) != index && e.cfg.name.eq_ignore_ascii_case(&cfg.name)) {
            bail!("another channel is already called {}", cfg.name);
        }
        let i = match index {
            Some(i) => {
                cfg.id = entries[i].cfg.id;
                if let Some(w) = entries[i].worker.take() {
                    w.stop();
                }
                entries[i].cfg = cfg;
                i
            }
            None => {
                cfg.id = entries.iter().map(|e| e.cfg.id).max().unwrap_or(0) + 1;
                entries.push(Entry { cfg, worker: None });
                entries.len() - 1
            }
        };
        if entries[i].cfg.enabled {
            entries[i].worker = Some(self.spawn(&entries[i].cfg));
        }
        self.save(&entries)?;
        Ok(Self::describe(&entries[i]))
    }

    pub fn set_enabled(&self, id: u32, enabled: bool) -> Result<Value> {
        let mut entries = self.entries.lock().unwrap();
        let i = entries.iter().position(|e| e.cfg.id == id).with_context(|| format!("channel {id} does not exist"))?;
        entries[i].cfg.enabled = enabled;
        if let Some(w) = entries[i].worker.take() {
            w.stop();
        }
        if enabled {
            entries[i].worker = Some(self.spawn(&entries[i].cfg));
        }
        self.save(&entries)?;
        Ok(Self::describe(&entries[i]))
    }

    pub fn delete(&self, id: u32) -> Result<()> {
        let mut entries = self.entries.lock().unwrap();
        let i = entries.iter().position(|e| e.cfg.id == id).with_context(|| format!("channel {id} does not exist"))?;
        if let Some(w) = entries.remove(i).worker {
            w.stop();
        }
        self.save(&entries)
    }

    /// The frame the channel last read, for the preview.
    pub fn preview(&self, id: u32) -> Option<Arc<RgbImage>> {
        let entries = self.entries.lock().unwrap();
        let worker = entries.iter().find(|e| e.cfg.id == id)?.worker.as_ref()?;
        let frame = worker.shared.preview.lock().unwrap().clone();
        frame
    }

    /// The URL events are posted to right now, if one is set and usable.
    pub fn events_url(&self) -> Option<String> {
        self.ctx.target.read().unwrap().1.clone()
    }

    pub fn target(&self) -> Value {
        let t = self.ctx.target.read().unwrap();
        json!({ "url": t.0.url, "allow_remote": t.0.allow_remote, "events_url": t.1 })
    }

    /// Sets the URL events are posted to. It takes effect on the next event; channels keep running.
    pub fn set_target(&self, mut settings: TargetSettings) -> Result<Value> {
        settings.url = settings.url.trim().to_string();
        let resolved = if settings.url.is_empty() { None } else { Some(crate::server::events_url(&settings.url, settings.allow_remote)?) };
        // Hold the channel list while saving, so two saves cannot interleave.
        let entries = self.entries.lock().unwrap();
        *self.ctx.target.write().unwrap() = (settings, resolved);
        self.save(&entries)?;
        Ok(self.target())
    }

    pub fn shutdown(&self) {
        for entry in self.entries.lock().unwrap().iter_mut() {
            if let Some(w) = entry.worker.take() {
                w.stop();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plate_is_raised_once_it_is_steady_and_not_again_inside_the_window() {
        let cfg = ChannelConfig { stable_frames: 2, dedupe_seconds: 30, min_confidence: 0.9, ..ChannelConfig::default() };
        let mut t = Tracker::new(&cfg);
        assert!(!t.observe("BRL4104", 0.99, 0.0), "first sighting");
        assert!(!t.observe("BRL404", 0.99, 0.2), "a one-frame misread never becomes an event");
        assert!(t.observe("BRL4104", 0.99, 0.4), "second sighting");
        assert!(!t.observe("BRL4104", 0.99, 0.6), "already raised");
        assert!(!t.observe("BRL4104", 0.99, 20.0), "still inside the window");
        assert!(!t.observe("BRL4104", 0.99, 31.0), "window over, but it must be steady again");
        assert!(t.observe("BRL4104", 0.99, 31.2));
        assert!(!t.observe("EV232", 0.5, 40.0) && !t.observe("EV232", 0.5, 40.2), "below the confidence floor");
        assert!(!t.observe("EV232", 0.95, 50.0) && !t.observe("EV232", 0.95, 60.0), "sightings too far apart");
    }

    /// A flat grey 320x180 frame with an optional bright block.
    fn frame(block: Option<(usize, usize, usize, usize)>) -> RgbImage {
        let mut img = RgbImage::filled(320, 180, 90);
        if let Some((x0, y0, w, h)) = block {
            for y in y0..y0 + h {
                img.data[(y * 320 + x0) * 3..(y * 320 + x0 + w) * 3].fill(220);
            }
        }
        img
    }

    #[test]
    fn unchanged_frames_are_skipped_after_a_hold_and_read_again_on_change() {
        let cfg = ChannelConfig { skip_unchanged: true, change_percent: 1.0, ..ChannelConfig::default() };
        let mut gate = MotionGate::new(&cfg);
        let whole = [0, 0, 320, 180];
        let empty = frame(None);
        assert!(gate.should_read(&empty, whole, 0.0), "the first frame is always read");
        assert!(gate.should_read(&empty, whole, 1.0), "still inside the hold after a change");
        assert!(!gate.should_read(&empty, whole, 2.5), "nothing changed and the hold is over");
        assert!(!gate.should_read(&empty, whole, 4.0));
        assert!(gate.should_read(&empty, whole, 6.2), "heartbeat: one frame every few seconds regardless");
        assert!(!gate.should_read(&empty, whole, 6.4));

        let car = frame(Some((100, 60, 120, 80)));
        assert!(gate.should_read(&car, whole, 7.0), "a vehicle moved in");
        assert!(gate.should_read(&car, whole, 8.5), "it stopped, but reads continue for the hold");
        assert!(!gate.should_read(&car, whole, 9.5), "parked: skipped");
        assert!(gate.should_read(&empty, whole, 9.7), "it left");

        // A change smaller than the threshold (a timestamp ticking over) does not count...
        let mut gate = MotionGate::new(&cfg);
        gate.should_read(&empty, whole, 0.0);
        let tick = frame(Some((4, 4, 12, 6)));
        assert!(!gate.should_read(&tick, whole, 3.0));
        // ...and neither does a change outside the detection region.
        let mut gate = MotionGate::new(&cfg);
        let right_half = [160, 0, 320, 180];
        gate.should_read(&empty, right_half, 0.0);
        assert!(!gate.should_read(&frame(Some((10, 40, 100, 100))), right_half, 3.0));

        // Switched off, every frame is read.
        let mut off = MotionGate::new(&ChannelConfig { skip_unchanged: false, ..ChannelConfig::default() });
        assert!(off.should_read(&empty, whole, 0.0) && off.should_read(&empty, whole, 10.0) && off.should_read(&empty, whole, 10.1));
    }

    #[test]
    fn ppm_frames_parse_back_to_back() {
        let mut bytes = b"P6\n2 1\n255\n".to_vec();
        bytes.extend([1, 2, 3, 4, 5, 6]);
        bytes.extend(b"P6 1 1 255\n");
        bytes.extend([9, 9, 9]);
        let mut r = std::io::Cursor::new(bytes);
        let a = read_ppm(&mut r).unwrap().unwrap();
        assert_eq!((a.width, a.height, a.data.as_slice()), (2, 1, &[1u8, 2, 3, 4, 5, 6][..]));
        assert_eq!(read_ppm(&mut r).unwrap().unwrap().data, vec![9, 9, 9]);
        assert!(read_ppm(&mut r).unwrap().is_none());
        assert!(read_ppm(&mut std::io::Cursor::new(b"P6\n2 2\n255\nxx".to_vec())).is_err(), "truncated frame");
        assert!(read_ppm(&mut std::io::Cursor::new(b"JUNK".to_vec())).is_err());
    }

    #[test]
    fn passwords_are_masked_and_restored() {
        let stored = "rtsp://admin:s3cret@10.0.0.9:554/Streaming/Channels/101";
        let masked = mask_url(stored);
        assert_eq!(masked, "rtsp://admin:***@10.0.0.9:554/Streaming/Channels/101");
        assert_eq!(unmask_url(&masked, stored), stored);
        assert_eq!(unmask_url("rtsp://admin:new@10.0.0.9/x", stored), "rtsp://admin:new@10.0.0.9/x");
        assert_eq!(mask_url("rtsp://10.0.0.9/x"), "rtsp://10.0.0.9/x");
    }

    #[test]
    fn only_rtsp_sources_are_accepted_unless_files_are_allowed() {
        let ok = ChannelConfig { name: "Entry".into(), url: "rtsp://10.0.0.9:554/live".into(), ..ChannelConfig::default() };
        assert!(validate(&ok, false).is_ok());
        for url in ["http://10.0.0.9/x.mjpg", "/etc/passwd", "concat:a|b", "rtsp://", ""] {
            assert!(validate(&ChannelConfig { url: url.into(), ..ok.clone() }, false).is_err(), "{url}");
        }
        let file = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
        assert!(validate(&ChannelConfig { url: file.into(), ..ok.clone() }, true).is_ok());
        assert!(validate(&ChannelConfig { url: file.into(), ..ok.clone() }, false).is_err());
        assert!(validate(&ChannelConfig { fps: 0.0, ..ok.clone() }, false).is_err());
        assert!(validate(&ChannelConfig { name: " ".into(), ..ok }, false).is_err());
    }

    #[test]
    fn ffmpeg_is_asked_for_ppm_frames_at_the_channel_rate() {
        let cfg = ChannelConfig { url: "rtsp://10.0.0.9/live".into(), fps: 4.0, transport: Transport::Udp, ..ChannelConfig::default() };
        let args = ffmpeg_args(&cfg).join(" ");
        assert!(args.contains("-rtsp_transport udp") && args.contains("-i rtsp://10.0.0.9/live") && args.contains("-vf fps=4"));
        assert!(args.ends_with("-f image2pipe -c:v ppm pipe:1"));
    }
}
