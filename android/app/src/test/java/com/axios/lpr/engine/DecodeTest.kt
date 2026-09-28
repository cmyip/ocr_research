package com.axios.lpr.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DecodeTest {
    private fun probs(t: Int, c: Int, seq: IntArray): FloatArray {
        val p = FloatArray(t * c)
        for (r in 0 until t) p[r * c + seq[r]] = 1f
        return p
    }

    @Test fun greedyCtcCollapsesRepeatsAndDropsBlank() {
        val c = 37; val b = 36
        // B B _ R L L _ 4 _ 4  -> "BRL44" (repeat split by blank survives)
        val seq = intArrayOf(11, 11, b, 27, 21, 21, b, 4, b, 4)
        val r = Ctc.greedy(probs(seq.size, c, seq), seq.size, c)
        assertEquals("BRL44", Ctc.render(r.indices, Alphabets.ALNUM))
        assertEquals(1f, r.meanConf, 1e-6f)
    }

    @Test fun softmaxAppliedOnlyToLogits() {
        val p = floatArrayOf(0.2f, 0.8f)
        assertTrue(Ctc.softmaxIfLogits(p, 1, 2) === p)
        assertEquals(0.2f, p[0], 1e-6f)
        val l = floatArrayOf(0f, 0f, 3f, 1f)
        Ctc.softmaxIfLogits(l, 2, 2)
        assertEquals(0.5f, l[0], 1e-6f)
        assertEquals(1f, l[2] + l[3], 1e-5f)
    }

    @Test fun separatorAlphabetAndUnknownAlphabet() {
        assertEquals("3 43466", Ctc.render(listOf(3, 36, 4, 3, 4, 6, 6), Alphabets.forClasses(38)))
        assertEquals("[7,6,2]", Ctc.render(listOf(7, 6, 2), Alphabets.forClasses(29)))
    }

    @Test fun yoloNmsKeepsBestPerClassAndUnmapsLetterbox() {
        val n = 4
        val out = FloatArray(6 * n)
        fun set(i: Int, cx: Float, cy: Float, w: Float, h: Float, s0: Float, s1: Float) {
            out[i] = cx; out[n + i] = cy; out[2 * n + i] = w; out[3 * n + i] = h; out[4 * n + i] = s0; out[5 * n + i] = s1
        }
        set(0, 100f, 100f, 40f, 10f, 0.9f, 0f)
        set(1, 101f, 100f, 40f, 10f, 0.8f, 0f)  // overlaps 0 → suppressed
        set(2, 200f, 150f, 30f, 10f, 0.5f, 0f)  // separate plate
        set(3, 150f, 120f, 200f, 150f, 0f, 0.95f) // vehicle
        val lb = Letterbox(320, 1920, 1080)
        val d = YoloDecoder.decode(out, n, lb, 0.25f, 0.4f, 0.45f, 8, 3, true)
        assertEquals(2, d.count { it.cls == DetClass.PLATE })
        assertEquals(1, d.count { it.cls == DetClass.VEHICLE })
        val best = d.first()
        assertEquals(0.9f, best.score, 0f)
        assertEquals(80f / lb.ratio, best.box.x1, 0.01f)
    }

    @Test fun voteBeatsSingleConfidentOutlier() {
        fun r(t: String, c: Float) = OcrRead(0, t, c, emptyList(), emptyList(), 37, true, 0f)
        val v = OcrVote.vote(listOf(r("BRL404", 0.99f), r("BRL4104", 0.9f), r("BRL4104", 0.9f)))!!
        assertEquals("BRL4104", v.text)
    }
}
