package com.axios.lpr.engine

import kotlin.math.abs

/** Plate crops prepared for OCR. */
data class PlateCrops(
    val raw: RgbImage,            // padded detector crop at native resolution (for display/export)
    val plain96: RgbImage,        // read_plate.crop_plate output (pad + optional deshear), 96×48
    val rectified: RgbImage?,     // 192×96 perspective-rectified plate, if the corner model ran and succeeded
    val ocrInput: RgbImage,       // exact 96×48 image fed to the CRNNs
    val corners: FloatArray?,     // TL, BL, BR, TR in image coordinates (x,y ×4)
    val cornerMs: Float,
)

object PlateCropper {
    const val W = 96
    const val H = 48

    /** Port of read_plate.crop_plate. */
    fun plain(img: RgbImage, b: Box, pad: Float, deshear: Float): Pair<RgbImage, RgbImage> {
        var x1 = b.x1; var y1 = b.y1; var x2 = b.x2; var y2 = b.y2
        val px = (x2 - x1) * pad
        val py = (y2 - y1) * pad
        var src = img
        if (deshear != 0f) {
            val mx = (x2 - x1) * 0.1f
            val my = (y2 - y1) * 0.3f
            val big = img.crop(x1 - mx, y1 - my, x2 + mx, y2 + my).deshear(deshear)
            src = big; x1 = mx; y1 = my; x2 = big.width - mx; y2 = big.height - my
        }
        val crop = src.crop(maxOf(0f, x1 - px), maxOf(0f, y1 - py), minOf(src.width.toFloat(), x2 + px), minOf(src.height.toFloat(), y2 + py))
        return crop to crop.resize(W, H)
    }
}

/**
 * rec_71 corner keypoints. Input: plate box expanded by a context margin, 96×48 BGR in [-1,1].
 * Output: 8 sigmoids = TL, BL, BR, TR as (x, y) fractions of that context crop. Verified visually
 * on the Malaysian samples; with the tight detector box the model is much less accurate.
 */
class CornerRectifier(private val models: ModelManager) {
    fun corners(img: RgbImage, b: Box, marginX: Float, marginY: Float): FloatArray? {
        val ctx = b.expand(marginX, marginY)
        val crop = img.crop(ctx.x1, ctx.y1, ctx.x2, ctx.y2).resize(PlateCropper.W, PlateCropper.H)
        val p = models.mnn(Ids.CORNERS).runSingle(crop.toTensor(1f / 127.5f, -1f, bgr = true))
        if (p.size < 8) return null
        // Pillow rounds the crop box, so map back through the rounded box.
        val cx1 = ctx.x1.roundHalfEven().toFloat(); val cy1 = ctx.y1.roundHalfEven().toFloat()
        val cw = ctx.x2.roundHalfEven() - cx1; val ch = ctx.y2.roundHalfEven() - cy1
        val q = FloatArray(8) { i -> if (i % 2 == 0) cx1 + p[i] * cw else cy1 + p[i] * ch }
        return q.takeIf { plausible(it, b) }
    }

    /** Warp the quad (expanded by [pad] around its centre) to 192×96, then optional shear, then 96×48. */
    fun warp(img: RgbImage, quad: FloatArray, pad: Float, deshear: Float): Pair<RgbImage, RgbImage> {
        val cx = (quad[0] + quad[2] + quad[4] + quad[6]) / 4
        val cy = (quad[1] + quad[3] + quad[5] + quad[7]) / 4
        val q = FloatArray(8) { i -> if (i % 2 == 0) cx + (quad[i] - cx) * (1 + pad) else cy + (quad[i] - cy) * (1 + pad) }
        var warped = img.quad(PlateCropper.W * 2, PlateCropper.H * 2, q)
        if (deshear != 0f) warped = warped.deshear(deshear)
        return warped to warped.resize(PlateCropper.W, PlateCropper.H)
    }

    companion object {
        /** Reject degenerate quads: must be convex and cover a sensible share of the detector box. */
        fun plausible(q: FloatArray, b: Box): Boolean {
            var sign = 0
            for (i in 0 until 4) {
                val ax = q[i * 2]; val ay = q[i * 2 + 1]
                val bx = q[(i + 1) % 4 * 2]; val by = q[(i + 1) % 4 * 2 + 1]
                val cx = q[(i + 2) % 4 * 2]; val cy = q[(i + 2) % 4 * 2 + 1]
                val cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
                val s = if (cross > 0) 1 else if (cross < 0) -1 else 0
                if (s == 0) return false
                if (sign == 0) sign = s else if (s != sign) return false
            }
            var area = 0f
            for (i in 0 until 4) {
                val j = (i + 1) % 4
                area += q[i * 2] * q[j * 2 + 1] - q[j * 2] * q[i * 2 + 1]
            }
            area = abs(area) / 2
            return area > 0.25f * b.area && area < 4f * b.area
        }
    }
}
