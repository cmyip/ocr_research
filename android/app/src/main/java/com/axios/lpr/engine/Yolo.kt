package com.axios.lpr.engine

import kotlin.math.max
import kotlin.math.roundToInt

/** Letterbox as in read_plate.py: resize longest side to [size], paste top-left on grey 114. */
class Letterbox(val size: Int, val srcW: Int, val srcH: Int) {
    val ratio: Float = size / max(srcW, srcH).toFloat()
    val newW: Int = (srcW * ratio).roundToInt().coerceIn(1, size)
    val newH: Int = (srcH * ratio).roundToInt().coerceIn(1, size)

    fun apply(img: RgbImage): RgbImage {
        val small = img.resize(newW, newH, reducingGap = 2f)
        val canvas = RgbImage.filled(size, size, 114)
        for (y in 0 until newH) System.arraycopy(small.data, y * newW * 3, canvas.data, y * size * 3, newW * 3)
        return canvas
    }

    fun unmap(b: Box) = Box(b.x1 / ratio, b.y1 / ratio, b.x2 / ratio, b.y2 / ratio)
}

/**
 * Decodes Ultralytics YOLOv8 output (1, 4 + nc, N): rows cx, cy, w, h, then one score per class.
 * Class 0 = plate, class 1 = vehicle (rec_73). Greedy per-class NMS.
 */
object YoloDecoder {
    fun decode(
        out: FloatArray, anchors: Int, lb: Letterbox,
        plateScore: Float, vehicleScore: Float, iou: Float, maxPlates: Int, maxVehicles: Int,
        withVehicles: Boolean,
    ): List<Detection> {
        val result = ArrayList<Detection>()
        result += nms(out, anchors, 0, plateScore, iou, maxPlates, lb, DetClass.PLATE)
        if (withVehicles) result += nms(out, anchors, 1, vehicleScore, iou, maxVehicles, lb, DetClass.VEHICLE)
        return result
    }

    private fun nms(out: FloatArray, n: Int, c: Int, thr: Float, iou: Float, maxDet: Int, lb: Letterbox, cls: DetClass): List<Detection> {
        val cand = ArrayList<Detection>()
        val row = (4 + c) * n
        for (i in 0 until n) {
            val s = out[row + i]
            if (s < thr) continue
            val cx = out[i]; val cy = out[n + i]; val w = out[2 * n + i]; val h = out[3 * n + i]
            cand += Detection(cls, s, lb.unmap(Box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
        }
        cand.sortByDescending { it.score }
        val keep = ArrayList<Detection>()
        for (d in cand) {
            if (keep.size >= maxDet) break
            if (keep.none { it.box.iou(d.box) > iou }) keep += d
        }
        return keep
    }
}
