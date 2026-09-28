package com.axios.lpr.engine

import kotlinx.serialization.Serializable

@Serializable
data class OcrRead(
    val model: Int,
    val text: String,
    val conf: Float,
    val charConf: List<Float>,
    val indices: List<Int>,
    val classes: Int,
    val decodable: Boolean,
    val ms: Float,
)

/** Runs one CRNN (rec_50..69): input 96×48 BGR /255, output (1, 18, C). */
class CrnnRunner(private val models: ModelManager) {
    fun read(id: Int, input96x48: RgbImage): OcrRead {
        val net = models.mnn(id)
        val x = input96x48.toTensor(1f / 255f, 0f, bgr = true)
        val t0 = System.nanoTime()
        val out = net.runSingle(x)
        val ms = (System.nanoTime() - t0) / 1e6f
        val shape = net.outputShapes[0]
        val c = shape.last()
        val t = out.size / c
        Ctc.softmaxIfLogits(out, t, c)
        val r = Ctc.greedy(out, t, c)
        val alphabet = Alphabets.forClasses(c)
        return OcrRead(id, Ctc.render(r.indices, alphabet).trim(), r.meanConf, r.charConf, r.indices, c, alphabet != null, ms)
    }
}

object OcrVote {
    /** Confidence-weighted vote on whole strings; ties go to the single most confident read. */
    fun vote(reads: List<OcrRead>): OcrRead? {
        val ok = reads.filter { it.decodable && it.text.isNotBlank() }
        if (ok.isEmpty()) return null
        val score = ok.groupBy { it.text }.mapValues { (_, v) -> v.sumOf { it.conf.toDouble() } }
        val bestText = score.maxBy { it.value }.key
        val winner = ok.filter { it.text == bestText }.maxBy { it.conf }
        return winner.copy(conf = (score.getValue(bestText) / ok.size).toFloat())
    }
}
