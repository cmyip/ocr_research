package com.axios.lpr.engine

import kotlinx.serialization.Serializable

@Serializable
data class RegionRead(val classifier: Int, val group: String, val label: String, val conf: Float, val canReject: Boolean, val ms: Float) {
    val isReject: Boolean get() = label.startsWith("9999")
    val description: String get() = RegionGroups.describe(label)
}

@Serializable
data class RegionDecision(val reads: List<RegionRead>, val chosen: RegionRead?) {
    val group: RegionGroup? get() = chosen?.let { RegionGroups.byName(it.group) }
}

/**
 * Runs the group region classifiers (MobileNetV3, 96×48 BGR raw 0–255) and picks a group.
 * Only classifiers with a 9999 "other" class can reject a plate, so only they route; the
 * 4xxx region-set classifiers (rec_41/47/82) always answer and are reported but never route.
 */
class RegionRouter(private val models: ModelManager) {
    fun classify(input96x48: RgbImage, enabled: Set<Int>, minConf: Float = 0.5f): RegionDecision {
        val x = input96x48.toTensor(1f, 0f, bgr = true)
        val reads = RegionGroups.all.filter { it.classifier in enabled }.map { g ->
            val net = models.mnn(g.classifier)
            val t0 = System.nanoTime()
            val p = net.runSingle(x.copyOf())
            val ms = (System.nanoTime() - t0) / 1e6f
            val labels = models.files.lines(g.idList)
            var best = 0
            for (k in p.indices) if (p[k] > p[best]) best = k
            RegionRead(g.classifier, g.name, labels.getOrElse(best) { "?" }, p[best], labels.any { it.startsWith("9999") }, ms)
        }
        val chosen = reads.filter { it.canReject && !it.isReject && it.conf >= minConf }.maxByOrNull { it.conf }
        return RegionDecision(reads, chosen)
    }
}
