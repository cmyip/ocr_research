//! The camera provider's event payload, as ParkingDashboard's `POST /api/alpr/events` receives it
//! from a Vaxtor camera: its payload template quotes every value and renders unset ones as null.
//! Field meanings follow ParkingDashboard's `AlprEventRequest` and `tools/TriggerBench`.
use base64::Engine;
use serde::Serialize;

use crate::image::RgbImage;
use crate::pipeline::PlateRead;

#[derive(Clone, Debug, Serialize)]
pub struct ProviderEvent {
    pub plate: Option<String>,
    /// Capture time, ISO 8601 with offset.
    pub date: String,
    /// 0-100, as the camera reports it.
    pub confidence: Option<String>,
    pub cameraid: String,
    /// Full frame, base64 (the bytes that were submitted).
    pub image: Option<String>,
    pub sizeinbytes: String,
    /// The plate crop the OCR read, base64 JPEG.
    pub cropplate: Option<String>,
    // Plate box in frame pixels.
    pub top: Option<String>,
    pub left: Option<String>,
    pub bottom: Option<String>,
    pub right: Option<String>,
    /// Frame height in pixels.
    pub height: String,
    // Box origin as a fraction of the frame.
    pub absolutetop: Option<String>,
    pub absoluteleft: Option<String>,
}

pub fn now_iso() -> String {
    chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%.3f%:z").to_string()
}

impl ProviderEvent {
    /// `image_bytes` is the submitted frame; pass `with_images` to embed it and the plate crop.
    pub fn build(
        best: Option<&PlateRead>,
        frame: &RgbImage,
        image_bytes: &[u8],
        cameraid: &str,
        date: Option<String>,
        with_images: bool,
    ) -> Self {
        let b64 = |bytes: &[u8]| base64::engine::general_purpose::STANDARD.encode(bytes);
        let px = |v: f32, max: usize| format!("{}", (v.round() as i64).clamp(0, max as i64));
        let bbox = best.map(|p| p.bbox);
        ProviderEvent {
            plate: best.map(|p| p.plate.clone()),
            date: date.filter(|d| !d.trim().is_empty()).unwrap_or_else(now_iso),
            confidence: best.map(|p| format!("{:.2}", p.confidence * 100.0)),
            cameraid: cameraid.to_string(),
            image: with_images.then(|| b64(image_bytes)),
            sizeinbytes: image_bytes.len().to_string(),
            cropplate: best.filter(|_| with_images).and_then(|p| p.ocr_input.encode_jpeg(90).ok()).map(|j| b64(&j)),
            top: bbox.map(|b| px(b.y1, frame.height)),
            left: bbox.map(|b| px(b.x1, frame.width)),
            bottom: bbox.map(|b| px(b.y2, frame.height)),
            right: bbox.map(|b| px(b.x2, frame.width)),
            height: frame.height.to_string(),
            absolutetop: bbox.map(|b| format!("{:.2}", (b.y1 / frame.height as f32).clamp(0.0, 1.0))),
            absoluteleft: bbox.map(|b| format!("{:.2}", (b.x1 / frame.width as f32).clamp(0.0, 1.0))),
        }
    }
}
