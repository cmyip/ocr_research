package com.axios.lpr.engine

import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Interleaved 8-bit RGB image. Pure Kotlin so the whole preprocessing chain is JVM-testable.
 *
 * The operations mirror Pillow's semantics used by read_plate.py (crop with rounding and
 * zero fill, antialiased bilinear resize, reduce(), AFFINE and QUAD transforms), so the
 * Android pipeline feeds the models the same tensors as the Python PoC.
 */
class RgbImage(val width: Int, val height: Int, val data: ByteArray = ByteArray(width * height * 3)) {
    init {
        require(width > 0 && height > 0) { "empty image ${width}x$height" }
        require(data.size == width * height * 3) { "bad buffer size" }
    }

    fun get(x: Int, y: Int, c: Int): Int = data[(y * width + x) * 3 + c].toInt() and 0xFF

    /** Pillow `Image.crop` with float box: coordinates are rounded, outside pixels are 0. */
    fun crop(x0f: Float, y0f: Float, x1f: Float, y1f: Float): RgbImage {
        val x0 = x0f.roundHalfEven()
        val y0 = y0f.roundHalfEven()
        val x1 = max(x0 + 1, x1f.roundHalfEven())
        val y1 = max(y0 + 1, y1f.roundHalfEven())
        val w = x1 - x0
        val h = y1 - y0
        val out = RgbImage(w, h)
        val sx0 = max(0, x0)
        val sx1 = min(width, x1)
        if (sx1 <= sx0) return out
        for (y in max(0, y0) until min(height, y1)) {
            System.arraycopy(data, (y * width + sx0) * 3, out.data, ((y - y0) * w + (sx0 - x0)) * 3, (sx1 - sx0) * 3)
        }
        return out
    }

    /** Pillow `Image.reduce(factor)`: box average over factor×factor blocks (partial edge blocks averaged). */
    fun reduce(fx: Int, fy: Int = fx): RgbImage {
        if (fx <= 1 && fy <= 1) return this
        val w = ceil(width / fx.toDouble()).toInt()
        val h = ceil(height / fy.toDouble()).toInt()
        val out = RgbImage(w, h)
        for (oy in 0 until h) {
            val y0 = oy * fy
            val y1 = min(height, y0 + fy)
            for (ox in 0 until w) {
                val x0 = ox * fx
                val x1 = min(width, x0 + fx)
                var r = 0
                var g = 0
                var b = 0
                for (y in y0 until y1) {
                    var i = (y * width + x0) * 3
                    for (x in x0 until x1) {
                        r += data[i].toInt() and 0xFF
                        g += data[i + 1].toInt() and 0xFF
                        b += data[i + 2].toInt() and 0xFF
                        i += 3
                    }
                }
                val n = (y1 - y0) * (x1 - x0)
                val o = (oy * w + ox) * 3
                out.data[o] = ((r + n / 2) / n).toByte()
                out.data[o + 1] = ((g + n / 2) / n).toByte()
                out.data[o + 2] = ((b + n / 2) / n).toByte()
            }
        }
        return out
    }

    /**
     * Pillow `Image.resize(size, BILINEAR, reducing_gap)`: optional integer box reduce first,
     * then a separable triangle filter whose support grows with the downscale factor (antialias).
     */
    fun resize(dstW: Int, dstH: Int, reducingGap: Float = 0f): RgbImage {
        if (dstW == width && dstH == height) return this
        var src = this
        // After reduce(), Pillow resamples the fractional box (0, 0, W/fx, H/fy) of the reduced image.
        var boxW = width.toDouble()
        var boxH = height.toDouble()
        if (reducingGap > 0f) {
            val fx = max(1, floor(width / dstW.toDouble() / reducingGap).toInt())
            val fy = max(1, floor(height / dstH.toDouble() / reducingGap).toInt())
            if (fx > 1 || fy > 1) {
                src = reduce(fx, fy)
                boxW = width / fx.toDouble()
                boxH = height / fy.toDouble()
            }
        }
        val horiz = if (dstW != src.width || boxW != src.width.toDouble()) src.resampleH(dstW, boxW) else src
        return if (dstH != horiz.height || boxH != horiz.height.toDouble()) horiz.resampleV(dstH, boxH) else horiz
    }

    private class Kernel(val bounds: IntArray, val weights: Array<FloatArray>)

    private fun kernel(inSize: Int, outSize: Int, boxSize: Double = inSize.toDouble()): Kernel {
        val scale = boxSize / outSize
        val filterScale = max(1.0, scale)
        val support = 1.0 * filterScale // bilinear (triangle) support = 1
        val bounds = IntArray(outSize * 2)
        val weights = Array(outSize) { FloatArray(0) }
        for (i in 0 until outSize) {
            val center = (i + 0.5) * scale
            val xmin = max(0, (center - support + 0.5).toInt())
            val xmax = min(inSize, (center + support + 0.5).toInt())
            val w = FloatArray(xmax - xmin)
            var sum = 0.0
            for (k in w.indices) {
                val t = (k + xmin - center + 0.5) / filterScale
                val v = if (t < 0) 1 + t else 1 - t
                val wv = if (v > 0) v else 0.0
                w[k] = wv.toFloat()
                sum += wv
            }
            if (sum > 0) for (k in w.indices) w[k] = (w[k] / sum).toFloat()
            bounds[i * 2] = xmin
            bounds[i * 2 + 1] = xmax - xmin
            weights[i] = w
        }
        return Kernel(bounds, weights)
    }

    private fun resampleH(dstW: Int, boxW: Double = width.toDouble()): RgbImage {
        val k = kernel(width, dstW, boxW)
        val out = RgbImage(dstW, height)
        for (y in 0 until height) {
            val row = y * width * 3
            for (x in 0 until dstW) {
                val xmin = k.bounds[x * 2]
                val n = k.bounds[x * 2 + 1]
                val w = k.weights[x]
                var r = 0f
                var g = 0f
                var b = 0f
                var i = row + xmin * 3
                for (j in 0 until n) {
                    val wj = w[j]
                    r += (data[i].toInt() and 0xFF) * wj
                    g += (data[i + 1].toInt() and 0xFF) * wj
                    b += (data[i + 2].toInt() and 0xFF) * wj
                    i += 3
                }
                val o = (y * dstW + x) * 3
                out.data[o] = clampByte(r)
                out.data[o + 1] = clampByte(g)
                out.data[o + 2] = clampByte(b)
            }
        }
        return out
    }

    private fun resampleV(dstH: Int, boxH: Double = height.toDouble()): RgbImage {
        val k = kernel(height, dstH, boxH)
        val out = RgbImage(width, dstH)
        val stride = width * 3
        for (y in 0 until dstH) {
            val ymin = k.bounds[y * 2]
            val n = k.bounds[y * 2 + 1]
            val w = k.weights[y]
            for (xc in 0 until stride) {
                var acc = 0f
                var i = ymin * stride + xc
                for (j in 0 until n) {
                    acc += (data[i].toInt() and 0xFF) * w[j]
                    i += stride
                }
                out.data[y * stride + xc] = clampByte(acc)
            }
        }
        return out
    }

    /** Bilinear sample with Pillow's transform semantics: outside the image → 0. */
    private fun sampleInto(out: ByteArray, o: Int, xinRaw: Double, yinRaw: Double) {
        if (xinRaw < 0 || xinRaw >= width || yinRaw < 0 || yinRaw >= height) {
            out[o] = 0; out[o + 1] = 0; out[o + 2] = 0
            return
        }
        val xin = xinRaw - 0.5
        val yin = yinRaw - 0.5
        val x = floor(xin).toInt()
        val y = floor(yin).toInt()
        val dx = (xin - x).toFloat()
        val dy = (yin - y).toFloat()
        val x0 = x.coerceIn(0, width - 1)
        val x1 = (x + 1).coerceIn(0, width - 1)
        val y0 = y.coerceIn(0, height - 1)
        val y1 = (y + 1).coerceIn(0, height - 1)
        val i00 = (y0 * width + x0) * 3
        val i01 = (y0 * width + x1) * 3
        val i10 = (y1 * width + x0) * 3
        val i11 = (y1 * width + x1) * 3
        for (c in 0 until 3) {
            val top = (data[i00 + c].toInt() and 0xFF) * (1 - dx) + (data[i01 + c].toInt() and 0xFF) * dx
            val bot = (data[i10 + c].toInt() and 0xFF) * (1 - dx) + (data[i11 + c].toInt() and 0xFF) * dx
            out[o + c] = clampByte(top * (1 - dy) + bot * dy)
        }
    }

    /** Pillow `Image.transform(size, AFFINE, (a,b,c,d,e,f), BILINEAR)`: output (x,y) samples input (ax+by+c, dx+ey+f). */
    fun affine(outW: Int, outH: Int, a: Double, b: Double, c: Double, d: Double, e: Double, f: Double): RgbImage {
        val out = RgbImage(outW, outH)
        for (y in 0 until outH) {
            val yy = y + 0.5
            for (x in 0 until outW) {
                val xx = x + 0.5
                sampleInto(out.data, (y * outW + x) * 3, a * xx + b * yy + c, d * xx + e * yy + f)
            }
        }
        return out
    }

    /**
     * Pillow `Image.transform(size, QUAD, (nw, sw, se, ne), BILINEAR)`: maps the source quadrilateral
     * (top-left, bottom-left, bottom-right, top-right) onto the output rectangle with bilinear mapping.
     */
    fun quad(outW: Int, outH: Int, q: FloatArray): RgbImage {
        require(q.size == 8)
        val x0 = q[0].toDouble(); val y0 = q[1].toDouble()
        val swx = q[2].toDouble(); val swy = q[3].toDouble()
        val sex = q[4].toDouble(); val sey = q[5].toDouble()
        val nex = q[6].toDouble(); val ney = q[7].toDouble()
        val aS = 1.0 / outW
        val aT = 1.0 / outH
        val a0 = x0; val a1 = (nex - x0) * aS; val a2 = (swx - x0) * aT; val a3 = (sex - swx - nex + x0) * aS * aT
        val b0 = y0; val b1 = (ney - y0) * aS; val b2 = (swy - y0) * aT; val b3 = (sey - swy - ney + y0) * aS * aT
        val out = RgbImage(outW, outH)
        for (y in 0 until outH) {
            val yy = y + 0.5
            for (x in 0 until outW) {
                val xx = x + 0.5
                sampleInto(out.data, (y * outW + x) * 3, a0 + a1 * xx + a2 * yy + a3 * xx * yy, b0 + b1 * xx + b2 * yy + b3 * xx * yy)
            }
        }
        return out
    }

    /** Horizontal shear used by read_plate.py: x_in = x - s·y + s·H/2 (positive s straightens right-leaning italics). */
    fun deshear(s: Float): RgbImage =
        if (s == 0f) this else affine(width, height, 1.0, -s.toDouble(), s * height / 2.0, 0.0, 1.0, 0.0)

    fun rotate(degrees: Int): RgbImage {
        val d = ((degrees % 360) + 360) % 360
        if (d == 0) return this
        val w2 = if (d == 180) width else height
        val h2 = if (d == 180) height else width
        val out = RgbImage(w2, h2)
        for (y in 0 until height) for (x in 0 until width) {
            val (nx, ny) = when (d) {
                90 -> (height - 1 - y) to x
                180 -> (width - 1 - x) to (height - 1 - y)
                else -> y to (width - 1 - x)
            }
            System.arraycopy(data, (y * width + x) * 3, out.data, (ny * w2 + nx) * 3, 3)
        }
        return out
    }

    fun flipHorizontal(): RgbImage {
        val out = RgbImage(width, height)
        for (y in 0 until height) for (x in 0 until width) {
            System.arraycopy(data, (y * width + x) * 3, out.data, (y * width + (width - 1 - x)) * 3, 3)
        }
        return out
    }

    /**
     * NCHW float tensor (batch 1): value = pixel * scale + offset. [bgr] swaps channel order.
     */
    fun toTensor(scale: Float, offset: Float = 0f, bgr: Boolean = false): FloatArray {
        val plane = width * height
        val t = FloatArray(plane * 3)
        val c0 = if (bgr) 2 else 0
        val c2 = if (bgr) 0 else 2
        for (p in 0 until plane) {
            val i = p * 3
            t[p] = (data[i + c0].toInt() and 0xFF) * scale + offset
            t[plane + p] = (data[i + 1].toInt() and 0xFF) * scale + offset
            t[2 * plane + p] = (data[i + c2].toInt() and 0xFF) * scale + offset
        }
        return t
    }

    companion object {
        fun filled(w: Int, h: Int, v: Int): RgbImage = RgbImage(w, h, ByteArray(w * h * 3) { v.toByte() })

        /** Raw HWC uint8 buffers shipped with the SDK (sample inputs); [bgr] = stored as BGR. */
        fun fromRaw(bytes: ByteArray, w: Int, h: Int, bgr: Boolean): RgbImage {
            require(bytes.size == w * h * 3) { "raw sample size ${bytes.size} != ${w}x${h}x3" }
            if (!bgr) return RgbImage(w, h, bytes.copyOf())
            val d = ByteArray(bytes.size)
            for (i in 0 until w * h) {
                d[i * 3] = bytes[i * 3 + 2]; d[i * 3 + 1] = bytes[i * 3 + 1]; d[i * 3 + 2] = bytes[i * 3]
            }
            return RgbImage(w, h, d)
        }

        fun fromArgb(pixels: IntArray, w: Int, h: Int): RgbImage {
            val d = ByteArray(w * h * 3)
            for (i in 0 until w * h) {
                val p = pixels[i]
                d[i * 3] = (p shr 16).toByte(); d[i * 3 + 1] = (p shr 8).toByte(); d[i * 3 + 2] = p.toByte()
            }
            return RgbImage(w, h, d)
        }
    }

    fun toArgb(): IntArray = IntArray(width * height) { i ->
        (0xFF shl 24) or ((data[i * 3].toInt() and 0xFF) shl 16) or ((data[i * 3 + 1].toInt() and 0xFF) shl 8) or (data[i * 3 + 2].toInt() and 0xFF)
    }
}

private fun clampByte(v: Float): Byte {
    val r = (v + 0.5f).toInt()
    return (if (r < 0) 0 else if (r > 255) 255 else r).toByte()
}

/** Python's round(): banker's rounding, as used by Pillow's crop. */
internal fun Float.roundHalfEven(): Int = Math.rint(this.toDouble()).toInt()

internal fun Float.roundInt(): Int = this.roundToInt()
