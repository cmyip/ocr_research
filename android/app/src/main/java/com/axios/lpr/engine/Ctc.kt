package com.axios.lpr.engine

import kotlinx.serialization.Serializable
import kotlin.math.abs
import kotlin.math.exp

object Alphabets {
    const val ALNUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    /** Known alphabets by class count (incl. blank). C=38 adds a group separator at index 36. */
    fun forClasses(classes: Int): String? = when (classes) {
        37 -> ALNUM
        38 -> "$ALNUM "
        else -> null
    }
}

@Serializable
data class CtcResult(val indices: List<Int>, val charConf: List<Float>, val meanConf: Float)

object Ctc {
    /** In place: softmax each of the T rows unless they already sum to 1 (some CRNNs emit logits). */
    fun softmaxIfLogits(p: FloatArray, t: Int, c: Int): FloatArray {
        var isProb = true
        for (r in 0 until t) {
            var s = 0f
            for (k in 0 until c) s += p[r * c + k]
            if (abs(s - 1f) > 1e-3f) { isProb = false; break }
        }
        if (isProb) return p
        for (r in 0 until t) {
            var m = Float.NEGATIVE_INFINITY
            for (k in 0 until c) m = maxOf(m, p[r * c + k])
            var s = 0.0
            for (k in 0 until c) { val e = exp((p[r * c + k] - m).toDouble()); p[r * c + k] = e.toFloat(); s += e }
            for (k in 0 until c) p[r * c + k] = (p[r * c + k] / s).toFloat()
        }
        return p
    }

    /**
     * Greedy CTC, blank = last index. meanConf = mean over all T frames of the max probability
     * (matches read_plate.decode). charConf = max probability within each emitted run.
     */
    fun greedy(p: FloatArray, t: Int, c: Int): CtcResult {
        val blank = c - 1
        val idx = ArrayList<Int>()
        val cc = ArrayList<Float>()
        var prev = -1
        var sum = 0f
        for (r in 0 until t) {
            var best = 0
            var bp = p[r * c]
            for (k in 1 until c) if (p[r * c + k] > bp) { bp = p[r * c + k]; best = k }
            sum += bp
            if (best != blank) {
                if (best != prev) { idx += best; cc += bp } else if (cc.isNotEmpty()) cc[cc.size - 1] = maxOf(cc.last(), bp)
            }
            prev = best
        }
        return CtcResult(idx, cc, sum / t)
    }

    fun render(indices: List<Int>, alphabet: String?): String =
        if (alphabet == null) indices.joinToString(",", "[", "]")
        else indices.joinToString("") { if (it < alphabet.length) alphabet[it].toString() else "?" }
}
