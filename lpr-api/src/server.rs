//! HTTP layer over the pipeline.
//!
//!   POST /v1/read      image in, the read out, shaped like the camera provider's event
//!   POST /v1/trigger   the same, then the event is posted to `{target}/api/alpr/events`
//!   GET  /healthz
//!   GET  /             web UI for the RTSP channels (see channels.rs); its API lives under /api
//!
//! An image arrives as a raw body (image/jpeg, image/png), a multipart file field (`image`,
//! `upload` or `file`), or JSON with a base64 `image` - which is the provider's own payload, so a
//! captured camera post can be replayed here as it is.
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use axum::extract::{DefaultBodyLimit, FromRequest, Multipart, Path as UrlPath, Query, Request, State};
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use base64::Engine;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::Semaphore;

use crate::channels::{ChannelConfig, Context as ChannelContext, Manager, TargetSettings};
use crate::image::RgbImage;
use crate::payload::ProviderEvent;
use crate::pipeline::{FrameResult, Pipeline, PipelineConfig, DECODABLE_CRNNS, LATIN_CRNNS, REGION_GROUPS};

const MAX_BODY: usize = 32 * 1024 * 1024;

#[derive(Clone, Debug, clap::Args)]
pub struct ServeArgs {
    #[command(flatten)]
    pub pipeline: PipelineConfig,
    #[arg(long, default_value = "127.0.0.1:8088")]
    pub listen: SocketAddr,
    /// Model sets kept loaded; each serves one request at a time.
    #[arg(long, default_value_t = 2)]
    pub workers: usize,
    /// Initial URL that /v1/trigger and triggering channels post reads to, e.g. http://localhost:5200.
    /// It can be set and changed in the web UI; once saved there, the saved one is used.
    #[arg(long, env = "LPR_TARGET")]
    pub target: Option<String>,
    /// With --target: confirms a host that is not this machine, where a real barrier may open.
    #[arg(long)]
    pub allow_remote: bool,
    /// Camera id used when a request does not name one; it must match a lane's camera id to be decided.
    #[arg(long, default_value = "LPR-API")]
    pub camera_id: String,
    /// Where the RTSP channels set up in the web UI are kept. Stream URLs in it may hold passwords.
    #[arg(long, default_value = "lpr-channels.json")]
    pub channels_file: PathBuf,
    /// The ffmpeg binary that decodes the streams.
    #[arg(long, default_value = "ffmpeg")]
    pub ffmpeg: String,
    /// Let a channel read a local video file instead of an RTSP stream, for testing without a camera.
    #[arg(long)]
    pub allow_file_sources: bool,
}

struct AppState {
    pool: Mutex<Vec<Pipeline>>,
    permits: Semaphore,
    http: reqwest::Client,
    camera_id: String,
    manager: Arc<Manager>,
    /// What the web UI can offer: which records exist, whether ffmpeg and a target are there.
    models: Value,
}

#[derive(Debug, Default, Deserialize)]
struct Params {
    cameraid: Option<String>,
    date: Option<String>,
    /// Embed the frame and plate crop as base64. Defaults to off for /v1/read, on for /v1/trigger.
    images: Option<String>,
}

struct Input {
    image: Vec<u8>,
    cameraid: Option<String>,
    date: Option<String>,
}

struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}

fn bad_request(msg: impl Into<String>) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, msg.into())
}

/// Resolves what a person entered to the URL events are posted to, refusing a remote host unless
/// it was confirmed. A bare base URL means a ParkingDashboard API, so its events path is added;
/// a URL with a path is used as it is.
pub fn events_url(target: &str, allow_remote: bool) -> Result<String> {
    let url = reqwest::Url::parse(target.trim()).with_context(|| format!("{target} is not a URL (expected something like http://localhost:5200)"))?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none_or(str::is_empty) {
        bail!("the URL must start with http:// or https:// and name a host");
    }
    let host = url.host_str().unwrap_or_default().trim_matches(['[', ']']).to_string();
    let local = host == "localhost" || host.ends_with(".localhost") || host.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback());
    if !local && !allow_remote {
        bail!("{host} is not this machine. Reads sent there are real to that site and may open a barrier: confirm that a remote URL is intended.");
    }
    if matches!(url.path(), "" | "/") && url.query().is_none() {
        return Ok(url.join("/api/alpr/events")?.to_string());
    }
    Ok(url.to_string())
}

pub async fn serve(args: ServeArgs) -> Result<()> {
    let seed = TargetSettings { url: args.target.clone().unwrap_or_default().trim().to_string(), allow_remote: args.allow_remote };
    let seed_url = if seed.url.is_empty() { None } else { Some(events_url(&seed.url, seed.allow_remote).context("--target")?) };
    let workers = args.workers.max(1);
    let t = Instant::now();
    let pool = (0..workers).map(|_| Pipeline::load(&args.pipeline)).collect::<Result<Vec<_>>>()?;
    eprintln!(
        "loaded {workers} model set(s) from {} in {:.0} ms (detector on {:?})",
        args.pipeline.weights.display(),
        t.elapsed().as_secs_f64() * 1e3,
        args.pipeline.detector_backend
    );

    let http = reqwest::Client::builder().timeout(std::time::Duration::from_secs(20)).build()?;
    let has_ffmpeg = std::process::Command::new(&args.ffmpeg)
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    if !has_ffmpeg {
        eprintln!("warning: {} did not run, so RTSP channels cannot decode; install ffmpeg or pass --ffmpeg", args.ffmpeg);
    }
    let models = catalog(&args, has_ffmpeg);
    let manager = Manager::open(
        ChannelContext {
            weights: args.pipeline.weights.clone(),
            threads: args.pipeline.threads,
            detector_backend: args.pipeline.detector_backend,
            ffmpeg: args.ffmpeg.clone(),
            allow_files: args.allow_file_sources,
            target: std::sync::RwLock::new((seed, seed_url)),
            http: http.clone(),
            runtime: tokio::runtime::Handle::current(),
        },
        args.channels_file.clone(),
    )?;
    eprintln!("{} channel(s) from {}", manager.list().len(), args.channels_file.display());

    let state = Arc::new(AppState { pool: Mutex::new(pool), permits: Semaphore::new(workers), http, camera_id: args.camera_id, manager: manager.clone(), models });
    match manager.events_url() {
        Some(url) => eprintln!("/v1/trigger and triggering channels post reads to {url}"),
        None => eprintln!("no event URL is set: set one in the web UI's settings to post reads"),
    }
    let app = Router::new()
        .route("/healthz", get(|| async { Json(json!({ "status": "ok" })) }))
        .route("/v1/read", post(read))
        .route("/v1/trigger", post(trigger))
        .route("/", get(|| async { Html(include_str!("ui.html")) }))
        .route("/api/models", get(|State(s): State<Arc<AppState>>| async move { Json(s.models.clone()) }))
        .route("/api/channels", get(list_channels).post(create_channel))
        .route("/api/channels/{id}", put(update_channel).delete(delete_channel))
        .route("/api/channels/{id}/start", post(start_channel))
        .route("/api/channels/{id}/stop", post(stop_channel))
        .route("/api/channels/{id}/snapshot.jpg", get(snapshot))
        .route("/api/settings", get(|State(s): State<Arc<AppState>>| async move { Json(s.manager.target()) }).put(update_settings))
        .route("/api/events", get(|State(s): State<Arc<AppState>>| async move { Json(s.manager.events.recent(50)) }))
        .layer(DefaultBodyLimit::max(MAX_BODY))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(args.listen).await.with_context(|| format!("binding {}", args.listen))?;
    eprintln!("listening on http://{} (web UI at /)", args.listen);
    axum::serve(listener, app).with_graceful_shutdown(async { tokio::signal::ctrl_c().await.ok(); }).await?;
    tokio::task::spawn_blocking(move || manager.shutdown()).await.ok();
    Ok(())
}

/// The models a channel can choose from, with what is known about each (see HANDOVER.md; the
/// region names are inferred from the SDK's samples, not shipped with it).
fn catalog(args: &ServeArgs, has_ffmpeg: bool) -> Value {
    let exists = |id: u32| args.pipeline.weights.join(format!("rec_{id:02}.bin")).is_file();
    let crnn_note = |id: u32| match id {
        50 => "USA group",
        53 => "Australia group",
        55 => "Kuwait group, 38 classes (digit groups)",
        57 => "Singapore/Malaysia group - verified on Malaysian plates",
        60 => "Kazakhstan group",
        _ => "0-9A-Z, region unknown",
    };
    let region_note = |group: &str| match group {
        "2033" => "USA states",
        "1195" => "Spain and neighbours",
        "1187" => "Singapore, Malaysia",
        "2207" => "Australia",
        "1110" => "Kuwait",
        "1106" => "Kazakhstan",
        "4001" => "Thailand provinces (cannot route)",
        "1044" => "group 1044",
        "4416" => "region set 44xx (cannot route)",
        "4101" => "region set 41xx (cannot route)",
        _ => "",
    };
    json!({
        "detectors": [
            { "size": 320, "record": 72, "note": "fast, the default", "available": exists(72) },
            { "size": 640, "record": 75, "note": "small or distant plates, about 3x slower", "available": exists(75) },
        ],
        "crnns": DECODABLE_CRNNS.iter().map(|&id| json!({ "id": id, "note": crnn_note(id), "ensemble": LATIN_CRNNS.contains(&id), "available": exists(id) })).collect::<Vec<_>>(),
        "regions": REGION_GROUPS.iter().map(|g| json!({ "id": g.1, "group": g.0, "crnn": g.3, "note": region_note(g.0), "available": exists(g.1) })).collect::<Vec<_>>(),
        "corners_available": exists(71),
        "ffmpeg": has_ffmpeg,
        "allow_files": args.allow_file_sources,
        "defaults": ChannelConfig::default(),
    })
}

async fn list_channels(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!(state.manager.list()))
}

/// Channel changes stop and start worker threads, so they run off the async executor.
async fn change<T: Send + 'static>(state: &Arc<AppState>, f: impl FnOnce(&Manager) -> Result<T> + Send + 'static) -> Result<T, ApiError> {
    let manager = state.manager.clone();
    tokio::task::spawn_blocking(move || f(&manager))
        .await
        .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| {
            let msg = format!("{e:#}");
            ApiError(if msg.contains("does not exist") { StatusCode::NOT_FOUND } else { StatusCode::BAD_REQUEST }, msg)
        })
}

fn parse_channel(body: Value) -> Result<ChannelConfig, ApiError> {
    serde_json::from_value(body).map_err(|e| bad_request(format!("invalid channel: {e}")))
}

async fn create_channel(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Result<Json<Value>, ApiError> {
    let cfg = parse_channel(body)?;
    Ok(Json(change(&state, move |m| m.upsert(None, cfg)).await?))
}

async fn update_channel(State(state): State<Arc<AppState>>, UrlPath(id): UrlPath<u32>, Json(body): Json<Value>) -> Result<Json<Value>, ApiError> {
    let cfg = parse_channel(body)?;
    Ok(Json(change(&state, move |m| m.upsert(Some(id), cfg)).await?))
}

async fn delete_channel(State(state): State<Arc<AppState>>, UrlPath(id): UrlPath<u32>) -> Result<Json<Value>, ApiError> {
    change(&state, move |m| m.delete(id)).await?;
    Ok(Json(json!({ "deleted": id })))
}

async fn start_channel(State(state): State<Arc<AppState>>, UrlPath(id): UrlPath<u32>) -> Result<Json<Value>, ApiError> {
    Ok(Json(change(&state, move |m| m.set_enabled(id, true)).await?))
}

async fn stop_channel(State(state): State<Arc<AppState>>, UrlPath(id): UrlPath<u32>) -> Result<Json<Value>, ApiError> {
    Ok(Json(change(&state, move |m| m.set_enabled(id, false)).await?))
}

async fn update_settings(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Result<Json<Value>, ApiError> {
    let settings: TargetSettings = serde_json::from_value(body).map_err(|e| bad_request(format!("invalid settings: {e}")))?;
    Ok(Json(change(&state, move |m| m.set_target(settings)).await?))
}

#[derive(Debug, Default, Deserialize)]
struct SnapshotParams {
    /// Width in pixels; the live wall asks for small ones.
    w: Option<usize>,
}

/// The frame the channel last read, scaled down for the preview.
async fn snapshot(State(state): State<Arc<AppState>>, UrlPath(id): UrlPath<u32>, Query(params): Query<SnapshotParams>) -> Result<Response, ApiError> {
    let frame = state.manager.preview(id).ok_or(ApiError(StatusCode::NOT_FOUND, "no frame yet".into()))?;
    let jpeg = tokio::task::spawn_blocking(move || {
        let width = frame.width.min(params.w.unwrap_or(960).clamp(160, 1920));
        let height = (frame.height * width / frame.width).max(1);
        frame.resize(width, height, 2.0).encode_jpeg(80)
    })
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;
    Ok(([(header::CONTENT_TYPE, "image/jpeg"), (header::CACHE_CONTROL, "no-store")], jpeg).into_response())
}

async fn read(State(state): State<Arc<AppState>>, Query(params): Query<Params>, req: Request) -> Result<Json<Value>, ApiError> {
    let started = Instant::now();
    let with_images = flag(params.images.as_deref(), false);
    let (event, mut body) = recognise(&state, params, req, with_images, started).await?;
    merge(&mut body, &event);
    Ok(Json(body))
}

async fn trigger(State(state): State<Arc<AppState>>, Query(params): Query<Params>, req: Request) -> Result<Json<Value>, ApiError> {
    let started = Instant::now();
    let Some(url) = state.manager.events_url() else {
        return Err(ApiError(StatusCode::SERVICE_UNAVAILABLE, "no event URL is set: set one in the web UI's settings".into()));
    };
    let with_images = flag(params.images.as_deref(), true);
    let (event, mut body) = recognise(&state, params, req, with_images, started).await?;

    // Like the camera, nothing is sent when no plate was read.
    let delivery = if event.plate.as_deref().is_some_and(|p| !p.is_empty()) {
        let t = Instant::now();
        let outcome = match state.http.post(&url).json(&event).send().await {
            Ok(resp) => {
                let status = resp.status().as_u16();
                let text = resp.text().await.unwrap_or_default();
                let parsed = serde_json::from_str::<Value>(&text).unwrap_or(Value::String(text.chars().take(2000).collect()));
                json!({ "sent": true, "url": url, "status": status, "response": parsed })
            }
            Err(e) => json!({ "sent": false, "url": url, "error": e.to_string() }),
        };
        let mut outcome = outcome;
        outcome["ms"] = json!(round2(t.elapsed().as_secs_f64() * 1e3));
        outcome
    } else {
        json!({ "sent": false, "url": url, "reason": "no plate read" })
    };

    // The reply keeps the event's fields but not megabytes of base64 the caller just sent.
    let mut echoed = event;
    echoed.image = None;
    echoed.cropplate = None;
    merge(&mut body, &echoed);
    body["delivery"] = delivery;
    body["timing_ms"]["request"] = json!(round2(started.elapsed().as_secs_f64() * 1e3));
    Ok(Json(body))
}

/// Reads the request's image and returns the provider event plus the reply's extra fields.
async fn recognise(state: &Arc<AppState>, params: Params, req: Request, with_images: bool, started: Instant) -> Result<(ProviderEvent, Value), ApiError> {
    let input = read_input(req).await?;
    let cameraid = params.cameraid.or(input.cameraid).filter(|c| !c.trim().is_empty()).unwrap_or_else(|| state.camera_id.clone());
    let date = params.date.or(input.date);

    let _permit = state.permits.acquire().await.map_err(|_| ApiError(StatusCode::SERVICE_UNAVAILABLE, "shutting down".into()))?;
    let queued_ms = started.elapsed().as_secs_f64() * 1e3;
    let st = state.clone();
    let (image, frame, decode_ms, result) = tokio::task::spawn_blocking(move || {
        let mut pipeline = st.pool.lock().unwrap().pop().expect("a permit guarantees a free pipeline");
        let t = Instant::now();
        let outcome = RgbImage::decode(&input.image).map_err(|e| (StatusCode::BAD_REQUEST, format!("{e:#}"))).and_then(|frame| {
            let decode_ms = t.elapsed().as_secs_f64() * 1e3;
            let result = pipeline.process(&frame).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;
            Ok((input.image, frame, decode_ms, result))
        });
        st.pool.lock().unwrap().push(pipeline);
        outcome
    })
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|(code, msg)| ApiError(code, msg))?;

    let FrameResult { plates, timings } = result;
    let event = ProviderEvent::build(plates.first(), &frame, &image, &cameraid, date, with_images);
    let body = json!({
        "plates": plates,
        "frame": { "width": frame.width, "height": frame.height },
        "timing_ms": {
            "decode": round2(decode_ms),
            "det_prep": round2(timings.det_prep_ms),
            "det_infer": round2(timings.det_infer_ms),
            "rectify": round2(timings.rectify_ms),
            "region": round2(timings.region_ms),
            "ocr": round2(timings.ocr_ms),
            // decode + pipeline: what recognising this image cost
            "total": round2(decode_ms + timings.pipeline_ms),
            // receiving the body and waiting for a free worker, before recognition started
            "queued": round2(queued_ms),
            "request": round2(started.elapsed().as_secs_f64() * 1e3),
        },
    });
    Ok((event, body))
}

async fn read_input(req: Request) -> Result<Input, ApiError> {
    let content_type = req.headers().get(header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or_default().to_ascii_lowercase();
    if content_type.starts_with("multipart/form-data") {
        let mut form = Multipart::from_request(req, &()).await.map_err(|e| bad_request(e.to_string()))?;
        let mut input = Input { image: Vec::new(), cameraid: None, date: None };
        while let Some(field) = form.next_field().await.map_err(|e| bad_request(e.to_string()))? {
            match field.name().unwrap_or_default().to_ascii_lowercase().as_str() {
                "image" | "upload" | "file" => input.image = field.bytes().await.map_err(|e| bad_request(e.to_string()))?.to_vec(),
                "cameraid" | "camera_id" => input.cameraid = field.text().await.ok(),
                "date" => input.date = field.text().await.ok(),
                _ => {}
            }
        }
        if input.image.is_empty() {
            return Err(bad_request("multipart body has no file field named image, upload or file"));
        }
        return Ok(input);
    }

    let body = axum::body::to_bytes(req.into_body(), MAX_BODY).await.map_err(|e| ApiError(StatusCode::PAYLOAD_TOO_LARGE, e.to_string()))?;
    if body.is_empty() {
        return Err(bad_request("empty body: send an image"));
    }
    if content_type.starts_with("application/json") {
        let v: Value = serde_json::from_slice(&body).map_err(|e| bad_request(format!("invalid JSON: {e}")))?;
        let text = |key: &str| v.get(key).and_then(Value::as_str).map(str::to_string);
        let b64 = text("image").ok_or_else(|| bad_request("JSON body needs a base64 \"image\""))?;
        // Tolerate a data: URL prefix and line-wrapped base64.
        let b64: String = b64.rsplit(',').next().unwrap_or_default().chars().filter(|c| !c.is_whitespace()).collect();
        let image = base64::engine::general_purpose::STANDARD.decode(b64).map_err(|e| bad_request(format!("\"image\" is not base64: {e}")))?;
        return Ok(Input { image, cameraid: text("cameraid").or_else(|| text("cameraId")), date: text("date") });
    }
    Ok(Input { image: body.to_vec(), cameraid: None, date: None })
}

fn flag(value: Option<&str>, default: bool) -> bool {
    match value.map(|v| v.to_ascii_lowercase()) {
        Some(v) if matches!(v.as_str(), "1" | "true" | "yes" | "both" | "on") => true,
        Some(v) if matches!(v.as_str(), "0" | "false" | "no" | "none" | "off") => false,
        _ => default,
    }
}

/// Puts the event's fields at the top level of the reply, next to the extras.
fn merge(body: &mut Value, event: &ProviderEvent) {
    if let (Value::Object(body), Ok(Value::Object(event))) = (body, serde_json::to_value(event)) {
        body.extend(event);
    }
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_remote_target_is_refused_until_allowed() {
        assert_eq!(events_url("http://localhost:5200", false).unwrap(), "http://localhost:5200/api/alpr/events");
        assert_eq!(events_url(" http://127.0.0.1:5200/ ", false).unwrap(), "http://127.0.0.1:5200/api/alpr/events");
        assert_eq!(events_url("http://[::1]:5200", false).unwrap(), "http://[::1]:5200/api/alpr/events");
        assert!(events_url("http://10.0.0.5:5200", false).is_err());
        assert!(events_url("https://parking.example.com", false).is_err());
        assert!(events_url("http://localhost.evil.com", false).is_err());
        assert_eq!(events_url("https://parking.example.com", true).unwrap(), "https://parking.example.com/api/alpr/events");
    }

    #[test]
    fn a_url_with_a_path_is_used_as_it_is() {
        assert_eq!(events_url("http://localhost:9000/hooks/plates", false).unwrap(), "http://localhost:9000/hooks/plates");
        assert_eq!(events_url("http://localhost:9000/?key=1", false).unwrap(), "http://localhost:9000/?key=1");
        for bad in ["localhost:5200", "ftp://localhost/x", "file:///etc/passwd", "not a url", "http://"] {
            assert!(events_url(bad, true).is_err(), "{bad}");
        }
    }
}
