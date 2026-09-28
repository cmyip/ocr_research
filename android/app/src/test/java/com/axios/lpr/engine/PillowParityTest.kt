package com.axios.lpr.engine

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Compares the pure-Kotlin preprocessing with Pillow outputs from the Python PoC
 * (golden_pipeline.json). JPEG decoders differ slightly (ImageIO vs libjpeg-turbo),
 * so sums are compared with a relative tolerance.
 */
class PillowParityTest {
    private val golden: JsonObject = Json.parseToJsonElement(
        javaClass.classLoader!!.getResource("golden_pipeline.json")!!.readText(),
    ).jsonObject

    private fun g(sample: String, key: String) = golden.getValue(sample).jsonObject.getValue(key)
    private fun box(sample: String) = g(sample, "box").jsonArray.map { it.jsonPrimitive.double.toFloat() }.let { Box(it[0], it[1], it[2], it[3]) }
    private fun sum(t: FloatArray) = t.sumOf { it.toDouble() }
    private fun close(expected: Double, actual: Double, rel: Double, what: String) =
        assertEquals(what, expected, actual, expected * rel)

    @Test fun letterboxMatchesPillow() {
        for (s in listOf("BRL4104", "EV232", "PUTRAJAYA541")) {
            val img = loadSample(s)
            val t = Letterbox(320, img.width, img.height).apply(img).toTensor(1f / 255f)
            close(g(s, "letterbox_sum").jsonPrimitive.double, sum(t), 0.002, "$s letterbox")
        }
    }

    @Test fun plainAndShearCropsMatchPillow() {
        for (s in listOf("BRL4104", "EV232", "PUTRAJAYA541")) {
            val img = loadSample(s)
            val b = box(s)
            val p03 = PlateCropper.plain(img, b, 0.03f, 0f).second.toTensor(1 / 255f, bgr = true)
            close(g(s, "plain_pad03_sum").jsonPrimitive.double, sum(p03), 0.01, "$s pad .03")
            val sh = PlateCropper.plain(img, b, 0.02f, 0.3f).second.toTensor(1 / 255f, bgr = true)
            close(g(s, "shear03_pad02_sum").jsonPrimitive.double, sum(sh), 0.01, "$s shear .3")
        }
    }

    @Test fun quadWarpMatchesPillow() {
        for (s in listOf("BRL4104", "EV232", "PUTRAJAYA541")) {
            val img = loadSample(s)
            val q = g(s, "corners").jsonArray.map { it.jsonPrimitive.double.toFloat() }.toFloatArray()
            assertEquals(true, CornerRectifier.plausible(q, box(s)))
            // Same pad expansion as CornerRectifier.warp, without needing the model.
            val cx = (q[0] + q[2] + q[4] + q[6]) / 4; val cy = (q[1] + q[3] + q[5] + q[7]) / 4
            val qq = FloatArray(8) { i -> if (i % 2 == 0) cx + (q[i] - cx) * 1.03f else cy + (q[i] - cy) * 1.03f }
            val in96 = img.quad(192, 96, qq).resize(96, 48).toTensor(1 / 255f, bgr = true)
            close(g(s, "rect_in96_sum").jsonPrimitive.double, sum(in96), 0.01, "$s rectified")
        }
    }
}
