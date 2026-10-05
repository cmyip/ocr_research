//! Interleaved 8-bit RGB image with Pillow's semantics.
//!
//! A port of the Android app's `RgbImage.kt`, which mirrors the Pillow calls in `read_plate.py`
//! (crop with rounding and zero fill, antialiased bilinear resize, `reduce()`, QUAD transform),
//! so the models are fed the same tensors as the Python PoC. The OCR is sensitive to these details.
use anyhow::{Context, Result};

#[derive(Clone)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

struct Kernel {
    /// (first input index, tap count) per output pixel.
    bounds: Vec<(usize, usize)>,
    weights: Vec<Vec<f32>>,
}

impl RgbImage {
    pub fn new(width: usize, height: usize) -> Self {
        Self::filled(width, height, 0)
    }

    pub fn filled(width: usize, height: usize, value: u8) -> Self {
        assert!(width > 0 && height > 0, "empty image {width}x{height}");
        RgbImage { width, height, data: vec![value; width * height * 3] }
    }

    /// Decodes a JPEG or PNG into RGB.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let img = image::load_from_memory(bytes).context("not a readable JPEG or PNG image")?.into_rgb8();
        let (width, height) = (img.width() as usize, img.height() as usize);
        Ok(RgbImage { width, height, data: img.into_raw() })
    }

    pub fn encode_jpeg(&self, quality: u8) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality)
            .encode(&self.data, self.width as u32, self.height as u32, image::ExtendedColorType::Rgb8)
            .context("JPEG encode")?;
        Ok(out)
    }

    /// Pillow `Image.crop` with a float box: coordinates are rounded, outside pixels are 0.
    pub fn crop(&self, x0: f32, y0: f32, x1: f32, y1: f32) -> RgbImage {
        let x0 = round_half_even(x0);
        let y0 = round_half_even(y0);
        let x1 = round_half_even(x1).max(x0 + 1);
        let y1 = round_half_even(y1).max(y0 + 1);
        let (w, h) = ((x1 - x0) as usize, (y1 - y0) as usize);
        let mut out = RgbImage::new(w, h);
        let sx0 = x0.max(0);
        let sx1 = x1.min(self.width as i64);
        if sx1 <= sx0 {
            return out;
        }
        let n = (sx1 - sx0) as usize * 3;
        for y in y0.max(0)..y1.min(self.height as i64) {
            let src = (y as usize * self.width + sx0 as usize) * 3;
            let dst = ((y - y0) as usize * w + (sx0 - x0) as usize) * 3;
            out.data[dst..dst + n].copy_from_slice(&self.data[src..src + n]);
        }
        out
    }

    /// Pillow `Image.reduce`: box average over fx×fy blocks (partial edge blocks averaged).
    fn reduce(&self, fx: usize, fy: usize) -> RgbImage {
        let w = self.width.div_ceil(fx);
        let h = self.height.div_ceil(fy);
        let mut out = RgbImage::new(w, h);
        for oy in 0..h {
            let y0 = oy * fy;
            let y1 = (y0 + fy).min(self.height);
            for ox in 0..w {
                let x0 = ox * fx;
                let x1 = (x0 + fx).min(self.width);
                let mut sum = [0u32; 3];
                for y in y0..y1 {
                    let row = &self.data[(y * self.width + x0) * 3..(y * self.width + x1) * 3];
                    for px in row.chunks_exact(3) {
                        sum[0] += px[0] as u32;
                        sum[1] += px[1] as u32;
                        sum[2] += px[2] as u32;
                    }
                }
                let n = ((y1 - y0) * (x1 - x0)) as u32;
                let o = (oy * w + ox) * 3;
                for c in 0..3 {
                    out.data[o + c] = ((sum[c] + n / 2) / n) as u8;
                }
            }
        }
        out
    }

    /// Pillow `Image.resize(size, BILINEAR, reducing_gap)`: optional integer box reduce first,
    /// then a separable triangle filter whose support grows with the downscale factor.
    pub fn resize(&self, dst_w: usize, dst_h: usize, reducing_gap: f32) -> RgbImage {
        if dst_w == self.width && dst_h == self.height {
            return self.clone();
        }
        let mut reduced = None;
        // After reduce(), Pillow resamples the fractional box (0, 0, W/fx, H/fy) of the reduced image.
        let mut box_w = self.width as f64;
        let mut box_h = self.height as f64;
        if reducing_gap > 0.0 {
            let fx = ((self.width as f64 / dst_w as f64 / reducing_gap as f64).floor() as usize).max(1);
            let fy = ((self.height as f64 / dst_h as f64 / reducing_gap as f64).floor() as usize).max(1);
            if fx > 1 || fy > 1 {
                reduced = Some(self.reduce(fx, fy));
                box_w = self.width as f64 / fx as f64;
                box_h = self.height as f64 / fy as f64;
            }
        }
        let src = reduced.as_ref().unwrap_or(self);
        let horiz = if dst_w != src.width || box_w != src.width as f64 { src.resample_h(dst_w, box_w) } else { src.clone() };
        if dst_h != horiz.height || box_h != horiz.height as f64 {
            horiz.resample_v(dst_h, box_h)
        } else {
            horiz
        }
    }

    fn kernel(in_size: usize, out_size: usize, box_size: f64) -> Kernel {
        let scale = box_size / out_size as f64;
        let filter_scale = scale.max(1.0);
        let support = filter_scale; // bilinear (triangle) support = 1
        let mut bounds = Vec::with_capacity(out_size);
        let mut weights = Vec::with_capacity(out_size);
        for i in 0..out_size {
            let center = (i as f64 + 0.5) * scale;
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = (((center + support + 0.5) as i64).max(0) as usize).min(in_size);
            let raw: Vec<f64> = (xmin..xmax)
                .map(|k| {
                    let t = (k as f64 - center + 0.5) / filter_scale;
                    (if t < 0.0 { 1.0 + t } else { 1.0 - t }).max(0.0)
                })
                .collect();
            let sum: f64 = raw.iter().sum();
            let norm = if sum > 0.0 { sum } else { 1.0 };
            bounds.push((xmin, xmax.saturating_sub(xmin)));
            weights.push(raw.iter().map(|&v| (v as f32 as f64 / norm) as f32).collect());
        }
        Kernel { bounds, weights }
    }

    fn resample_h(&self, dst_w: usize, box_w: f64) -> RgbImage {
        let k = Self::kernel(self.width, dst_w, box_w);
        let mut out = RgbImage::new(dst_w, self.height);
        for y in 0..self.height {
            let row = &self.data[y * self.width * 3..(y + 1) * self.width * 3];
            for x in 0..dst_w {
                let (xmin, n) = k.bounds[x];
                let mut acc = [0f32; 3];
                for (px, &w) in row[xmin * 3..(xmin + n) * 3].chunks_exact(3).zip(&k.weights[x]) {
                    acc[0] += px[0] as f32 * w;
                    acc[1] += px[1] as f32 * w;
                    acc[2] += px[2] as f32 * w;
                }
                let o = (y * dst_w + x) * 3;
                for c in 0..3 {
                    out.data[o + c] = clamp_byte(acc[c]);
                }
            }
        }
        out
    }

    fn resample_v(&self, dst_h: usize, box_h: f64) -> RgbImage {
        let k = Self::kernel(self.height, dst_h, box_h);
        let mut out = RgbImage::new(self.width, dst_h);
        let stride = self.width * 3;
        let mut acc = vec![0f32; stride];
        for y in 0..dst_h {
            let (ymin, n) = k.bounds[y];
            acc.iter_mut().for_each(|v| *v = 0.0);
            for (j, &w) in k.weights[y].iter().enumerate().take(n) {
                let row = &self.data[(ymin + j) * stride..(ymin + j + 1) * stride];
                for (a, &p) in acc.iter_mut().zip(row) {
                    *a += p as f32 * w;
                }
            }
            for (o, &a) in out.data[y * stride..(y + 1) * stride].iter_mut().zip(&acc) {
                *o = clamp_byte(a);
            }
        }
        out
    }

    /// Bilinear sample with Pillow's transform semantics: outside the image is 0.
    fn sample_into(&self, out: &mut [u8], x_raw: f64, y_raw: f64) {
        if x_raw < 0.0 || x_raw >= self.width as f64 || y_raw < 0.0 || y_raw >= self.height as f64 {
            out.fill(0);
            return;
        }
        let (xin, yin) = (x_raw - 0.5, y_raw - 0.5);
        let (x, y) = (xin.floor(), yin.floor());
        let (dx, dy) = ((xin - x) as f32, (yin - y) as f32);
        let clamp = |v: f64, max: usize| (v as i64).clamp(0, max as i64 - 1) as usize;
        let (x0, x1) = (clamp(x, self.width), clamp(x + 1.0, self.width));
        let (y0, y1) = (clamp(y, self.height), clamp(y + 1.0, self.height));
        let at = |xx: usize, yy: usize, c: usize| self.data[(yy * self.width + xx) * 3 + c] as f32;
        for c in 0..3 {
            let top = at(x0, y0, c) * (1.0 - dx) + at(x1, y0, c) * dx;
            let bottom = at(x0, y1, c) * (1.0 - dx) + at(x1, y1, c) * dx;
            out[c] = clamp_byte(top * (1.0 - dy) + bottom * dy);
        }
    }

    /// Pillow `Image.transform(size, QUAD, (nw, sw, se, ne), BILINEAR)`: maps the source
    /// quadrilateral (top-left, bottom-left, bottom-right, top-right) onto the output rectangle.
    pub fn quad(&self, out_w: usize, out_h: usize, q: &[f32; 8]) -> RgbImage {
        let [x0, y0, swx, swy, sex, sey, nex, ney] = q.map(|v| v as f64);
        let (a_s, a_t) = (1.0 / out_w as f64, 1.0 / out_h as f64);
        let (a0, a1, a2, a3) = (x0, (nex - x0) * a_s, (swx - x0) * a_t, (sex - swx - nex + x0) * a_s * a_t);
        let (b0, b1, b2, b3) = (y0, (ney - y0) * a_s, (swy - y0) * a_t, (sey - swy - ney + y0) * a_s * a_t);
        let mut out = RgbImage::new(out_w, out_h);
        for y in 0..out_h {
            let yy = y as f64 + 0.5;
            for x in 0..out_w {
                let xx = x as f64 + 0.5;
                let o = (y * out_w + x) * 3;
                self.sample_into(&mut out.data[o..o + 3], a0 + a1 * xx + a2 * yy + a3 * xx * yy, b0 + b1 * xx + b2 * yy + b3 * xx * yy);
            }
        }
        out
    }

    /// Pillow `Image.transform(size, AFFINE, (a,b,c,d,e,f), BILINEAR)` at the same size:
    /// output (x, y) samples input (ax+by+c, dx+ey+f).
    fn affine(&self, m: [f64; 6]) -> RgbImage {
        let mut out = RgbImage::new(self.width, self.height);
        for y in 0..self.height {
            let yy = y as f64 + 0.5;
            for x in 0..self.width {
                let xx = x as f64 + 0.5;
                let o = (y * self.width + x) * 3;
                self.sample_into(&mut out.data[o..o + 3], m[0] * xx + m[1] * yy + m[2], m[3] * xx + m[4] * yy + m[5]);
            }
        }
        out
    }

    /// Horizontal shear used by read_plate.py: x_in = x - s·y + s·H/2. Positive `s` straightens
    /// right-leaning (italic) characters.
    pub fn deshear(&self, s: f32) -> RgbImage {
        if s == 0.0 {
            return self.clone();
        }
        let s = s as f64;
        self.affine([1.0, -s, s * self.height as f64 / 2.0, 0.0, 1.0, 0.0])
    }

    /// NCHW float tensor (batch 1): value = pixel * scale + offset. `bgr` swaps the channel order.
    pub fn to_tensor(&self, scale: f32, offset: f32, bgr: bool) -> Vec<f32> {
        let plane = self.width * self.height;
        let mut t = vec![0f32; plane * 3];
        let (c0, c2) = if bgr { (2, 0) } else { (0, 2) };
        for (p, px) in self.data.chunks_exact(3).enumerate() {
            t[p] = px[c0] as f32 * scale + offset;
            t[plane + p] = px[1] as f32 * scale + offset;
            t[2 * plane + p] = px[c2] as f32 * scale + offset;
        }
        t
    }
}

fn clamp_byte(v: f32) -> u8 {
    ((v + 0.5) as i32).clamp(0, 255) as u8
}

/// Python's `round()`: banker's rounding, as used by Pillow's crop.
pub fn round_half_even(v: f32) -> i64 {
    (v as f64).round_ties_even() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_rounds_half_to_even_and_zero_fills_outside() {
        let img = RgbImage::filled(4, 4, 200);
        let c = img.crop(-1.5, 0.5, 2.5, 2.0);
        assert_eq!((c.width, c.height), (4, 2)); // x: -2..2, y: 0..2
        assert_eq!(&c.data[0..3], &[0, 0, 0]);
        assert_eq!(&c.data[6..9], &[200, 200, 200]);
    }

    #[test]
    fn resize_of_a_flat_image_stays_flat() {
        let img = RgbImage::filled(1920, 1080, 77);
        let small = img.resize(320, 180, 2.0);
        assert_eq!((small.width, small.height), (320, 180));
        assert!(small.data.iter().all(|&v| v == 77));
    }

    #[test]
    fn deshear_keeps_the_middle_row_and_shifts_the_others() {
        // One white column at x = 4 in an 8-wide, 4-high image.
        let mut img = RgbImage::new(8, 4);
        for y in 0..4 {
            img.data[(y * 8 + 4) * 3..(y * 8 + 5) * 3].fill(255);
        }
        let out = img.deshear(1.0);
        let lit = |y: usize| (0..8).filter(|&x| out.data[(y * 8 + x) * 3] > 0).collect::<Vec<_>>();
        // x_in = x - (y + 0.5) + 2: the column moves right by one pixel per row going down.
        assert_eq!(lit(0), vec![2, 3]);
        assert_eq!(lit(1), vec![3, 4]);
        assert_eq!(lit(3), vec![5, 6]);
        assert_eq!(img.deshear(0.0).data, img.data);
    }

    #[test]
    fn identity_quad_reproduces_the_image() {
        let mut img = RgbImage::new(8, 4);
        img.data.iter_mut().enumerate().for_each(|(i, v)| *v = (i * 7 % 251) as u8);
        let out = img.quad(8, 4, &[0.0, 0.0, 0.0, 4.0, 8.0, 4.0, 8.0, 0.0]);
        assert_eq!(out.data, img.data);
    }
}
