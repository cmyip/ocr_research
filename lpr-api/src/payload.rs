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
    // The vehicle carrying the plate. Not part of the camera's template: present only when
    // vehicle attributes are switched on and were read.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vehiclemake: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vehiclemodel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vehiclecolor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vehicletype: Option<String>,
    /// "Front" or "Rear": which end of the vehicle faced the camera.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vehiclepose: Option<String>,
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
        let vehicle = best.and_then(|p| p.vehicle.as_ref());
        ProviderEvent {
            vehiclemake: vehicle.and_then(|v| v.make_model.as_ref()).map(|m| m.make.clone()),
            vehiclemodel: vehicle.and_then(|v| v.make_model.as_ref()).map(|m| m.model.clone()),
            vehiclecolor: vehicle.and_then(|v| v.colour.as_ref()).map(|l| l.label.clone()),
            vehicletype: vehicle.and_then(|v| v.kind.as_ref()).map(|l| l.label.clone()),
            vehiclepose: vehicle.and_then(|v| v.pose.as_ref()).map(|l| l.label.clone()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{BBox, Label, MakeModel, Vehicle};

    fn read(vehicle: Option<Vehicle>) -> PlateRead {
        PlateRead {
            plate: "BRL4104".into(),
            confidence: 0.99,
            det_score: 0.8,
            bbox: BBox { x1: 900.0, y1: 400.0, x2: 1200.0, y2: 500.0 },
            corners: None,
            ocr_model: "rec_57".into(),
            strategy: "pinned rec_57".into(),
            region: None,
            regions: Vec::new(),
            vehicle,
            ocr_input: RgbImage::new(96, 48),
        }
    }

    #[test]
    fn vehicle_fields_appear_only_when_they_were_read() {
        let frame = RgbImage::new(1920, 1080);
        let plain = serde_json::to_value(ProviderEvent::build(Some(&read(None)), &frame, b"jpeg", "CAM", None, false)).unwrap();
        assert_eq!(plain["plate"], "BRL4104");
        assert!(plain.as_object().unwrap().keys().all(|k| !k.starts_with("vehicle")), "the camera's payload is unchanged");

        let vehicle = Vehicle {
            bbox: BBox { x1: 500.0, y1: 100.0, x2: 1500.0, y2: 900.0 },
            det_score: 0.9,
            make_model: Some(MakeModel { make: "Toyota".into(), model: "Corolla Cross".into(), confidence: 0.99 }),
            pose: Some(Label { label: "Front".into(), confidence: 0.99 }),
            colour: None,
            kind: Some(Label { label: "SUV".into(), confidence: 0.76 }),
        };
        let with = serde_json::to_value(ProviderEvent::build(Some(&read(Some(vehicle))), &frame, b"jpeg", "CAM", None, false)).unwrap();
        assert_eq!((with["vehiclemake"].as_str(), with["vehiclemodel"].as_str()), (Some("Toyota"), Some("Corolla Cross")));
        assert_eq!((with["vehiclepose"].as_str(), with["vehicletype"].as_str()), (Some("Front"), Some("SUV")));
        assert!(with.get("vehiclecolor").is_none(), "colour was not asked for");
    }
}
