package com.axios.lpr.engine

import kotlinx.serialization.Serializable

@Serializable
data class Attr(val label: String, val conf: Float)

@Serializable
data class VehicleAttrs(
    val makeModel: Attr? = null,
    val pose: Attr? = null,
    val colour: Attr? = null,
    val type: Attr? = null,
    val refinedBox: Box? = null,
    val ms: Float = 0f,
) {
    fun summary(): String = listOfNotNull(colour?.label, makeModel?.label, type?.label, pose?.label).joinToString(" · ")
}

/**
 * MMC: rec_15 (make/model 4193 + pose 2), rec_19 colour (14), rec_21 type (8), rec_78 box refiner.
 * Input: vehicle crop 224×224 RGB scaled to [-1,1] (found empirically — the HANDOVER's raw 0–255
 * note gives "Large Truck" for every car; [-1,1] gives Toyota Corolla Cross on BRL4104.jpeg).
 */
class VehicleAnalyzer(private val models: ModelManager) {
    fun analyze(img: RgbImage, box: Box, cfg: PipelineConfig): VehicleAttrs {
        val t0 = System.nanoTime()
        val b = box.clip(img.width, img.height)
        val crop = img.crop(b.x1, b.y1, b.x2, b.y2).resize(224, 224)
        val x = crop.toTensor(1f / 127.5f, -1f, bgr = false)
        var mm: Attr? = null
        var pose: Attr? = null
        if (cfg.mmcMakeModel) {
            val net = models.mnn(Ids.MMR)
            val outs = net.run(x.copyOf())
            net.output("mmr", outs)?.let { mm = top(it, models.files.lines(16)) }
            net.output("pose", outs)?.let { pose = top(it, models.files.lines(18)) }
        }
        val colour = if (cfg.mmcColour) top(models.mnn(Ids.COLOUR).runSingle(x.copyOf()), models.files.lines(20)) else null
        val type = if (cfg.mmcType) top(models.mnn(Ids.VTYPE).runSingle(x.copyOf()), models.files.lines(22)) else null
        val refined = if (cfg.vehicleBoxRefine) {
            val r = models.mnn(Ids.VBOX).runSingle(x.copyOf())
            Box(b.x1 + r[0] * b.w, b.y1 + r[1] * b.h, b.x1 + r[2] * b.w, b.y1 + r[3] * b.h)
        } else null
        return VehicleAttrs(mm, pose, colour, type, refined, (System.nanoTime() - t0) / 1e6f)
    }

    companion object {
        /** Label lines are "code<TAB>Display name"; show the display part. */
        fun top(p: FloatArray, labels: List<String>): Attr {
            var best = 0
            for (k in p.indices) if (p[k] > p[best]) best = k
            val line = labels.getOrElse(best) { "#$best" }
            val parts = line.split('\t')
            // "white<TAB>White" and "BIGTRUCK<TAB>Large Truck" carry a code first; "Toyota<TAB>Corolla Cross" is make + model.
            val first = parts[0]
            val isCode = parts.size >= 2 && (first.firstOrNull()?.isLowerCase() == true || first.all { it.isUpperCase() })
            val label = if (isCode) parts.drop(1).joinToString(" ") else parts.joinToString(" ")
            return Attr(label.trim(), p[best])
        }
    }
}
