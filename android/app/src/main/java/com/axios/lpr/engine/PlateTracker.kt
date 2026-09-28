package com.axios.lpr.engine

/**
 * IoU tracker across live frames. Each track keeps recent (text, conf) reads and exposes a
 * confidence-weighted vote, so the viewfinder label doesn't flicker and single-frame misreads
 * (e.g. BRL404 vs BRL4104) are outvoted. Also decides when a track is stable enough to auto-record.
 */
class PlateTracker(
    private val iouMatch: Float = 0.3f,
    private val maxMisses: Int = 8,
    private val history: Int = 12,
) {
    class Track(val id: Int, var box: Box) {
        val reads = ArrayDeque<Pair<String, Float>>()
        var hits = 0
        var misses = 0
        var recorded = false
        var announced = false
        var lastPlate: PlateResult? = null

        fun vote(): Pair<String, Float> {
            val valid = reads.filter { it.first.isNotBlank() }
            if (valid.isEmpty()) return "" to 0f
            val scores = valid.groupBy { it.first }.mapValues { e -> e.value.sumOf { it.second.toDouble() } }
            val best = scores.maxBy { it.value }
            return best.key to (best.value / valid.size).toFloat()
        }

        /** Share of reads agreeing with the voted text. */
        fun agreement(): Float {
            val (t, _) = vote()
            val valid = reads.count { it.first.isNotBlank() }
            return if (valid == 0) 0f else reads.count { it.first == t } / valid.toFloat()
        }
    }

    private var nextId = 1
    val tracks = ArrayList<Track>()

    fun update(plates: List<PlateResult>): List<Track> {
        val unmatched = tracks.toMutableSet()
        for (p in plates.sortedByDescending { it.det.score }) {
            val best = unmatched.filter { sameText(it, p) }.maxByOrNull { it.box.iou(p.det.box) }
            val tr = if (best != null && best.box.iou(p.det.box) >= iouMatch) {
                unmatched -= best; best
            } else Track(nextId++, p.det.box).also { tracks += it }
            // light smoothing so the drawn box doesn't jitter
            tr.box = if (tr.hits == 0) p.det.box else Box(
                tr.box.x1 * 0.4f + p.det.box.x1 * 0.6f, tr.box.y1 * 0.4f + p.det.box.y1 * 0.6f,
                tr.box.x2 * 0.4f + p.det.box.x2 * 0.6f, tr.box.y2 * 0.4f + p.det.box.y2 * 0.6f,
            )
            tr.reads.addLast(p.text to p.conf)
            while (tr.reads.size > history) tr.reads.removeFirst()
            tr.hits++
            tr.misses = 0
            tr.lastPlate = p
        }
        unmatched.forEach { it.misses++ }
        tracks.removeAll { it.misses > maxMisses }
        return tracks.filter { it.misses == 0 }
    }

    /**
     * A confident read that is textually far from the track's vote means a different plate took
     * the same spot (next car at a gate), so it must not inherit the old votes.
     */
    private fun sameText(t: Track, p: PlateResult): Boolean {
        val (voted, _) = t.vote()
        if (voted.length < 3 || p.text.length < 3 || p.conf < 0.8f) return true
        return levenshtein(voted, p.text) <= maxOf(2, voted.length / 2)
    }

    fun isStable(t: Track, minHits: Int, minConf: Float): Boolean {
        val (text, conf) = t.vote()
        return text.length >= 2 && t.hits >= minHits && conf >= minConf && t.agreement() >= 0.6f
    }

    fun reset() { tracks.clear() }

    companion object {
        fun levenshtein(a: String, b: String): Int {
            var prev = IntArray(b.length + 1) { it }
            for (i in 1..a.length) {
                val cur = IntArray(b.length + 1)
                cur[0] = i
                for (j in 1..b.length) cur[j] = minOf(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + if (a[i - 1] == b[j - 1]) 0 else 1)
                prev = cur
            }
            return prev[b.length]
        }
    }
}

/**
 * Maps analysis-image coordinates onto the preview view. The analysis frame is cropped to the
 * same ViewPort as the preview, so the mapping is a uniform scale plus centring offset
 * (FILL_CENTER); [mirror] handles the front camera.
 */
class CoordinateMapper(private val imgW: Int, private val imgH: Int, private val viewW: Float, private val viewH: Float, private val mirror: Boolean = false) {
    val scale: Float = maxOf(viewW / imgW, viewH / imgH)
    private val dx = (viewW - imgW * scale) / 2
    private val dy = (viewH - imgH * scale) / 2

    fun x(v: Float): Float { val m = v * scale + dx; return if (mirror) viewW - m else m }
    fun y(v: Float): Float = v * scale + dy

    fun map(b: Box): Box {
        val a = x(b.x1); val c = x(b.x2)
        return Box(minOf(a, c), y(b.y1), maxOf(a, c), y(b.y2))
    }

    fun unmap(b: Box): Box {
        fun ux(v: Float) = ((if (mirror) viewW - v else v) - dx) / scale
        fun uy(v: Float) = (v - dy) / scale
        val a = ux(b.x1); val c = ux(b.x2)
        return Box(minOf(a, c), uy(b.y1), maxOf(a, c), uy(b.y2))
    }
}
