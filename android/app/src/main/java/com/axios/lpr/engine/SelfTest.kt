package com.axios.lpr.engine

data class SelfTestItem(val name: String, val expected: String, val got: String, val pass: Boolean, val ms: Float)

/**
 * Runs the SDK's bundled sample inputs through the models (same checks as decode_samples.py),
 * so a device can be verified without any camera input.
 */
class SelfTest(private val pipeline: PlatePipeline) {
    private val models = pipeline.models
    private val files = models.files

    private val expectedRegion = mapOf(23 to "1222\t2033", 26 to "1195", 29 to "1187", 32 to "1012\t2207", 35 to "1110",
        38 to "1106", 41 to "4001", 44 to "1044", 47 to "4416", 82 to "4101")

    /** Verified exact reads from HANDOVER.md / samples_report.txt. */
    private val expectedRead = listOf(50 to 25, 57 to 25, 65 to 28, 60 to 28, 50 to 31, 53 to 31, 57 to 31, 50 to 34, 53 to 34, 55 to 37)

    fun run(cfg: PipelineConfig, onItem: (SelfTestItem) -> Unit) {
        models.configure(cfg)
        val router = RegionRouter(models)
        for (g in RegionGroups.all) {
            val crop = RgbImage.fromRaw(files.bytes(g.sample), 96, 48, bgr = true)
            val t = System.nanoTime()
            val r = router.classify(crop, setOf(g.classifier)).reads.single()
            val exp = expectedRegion.getValue(g.classifier)
            onItem(SelfTestItem("rec_${g.classifier} region on rec_${g.sample}", RegionGroups.describe(exp), r.description, r.label == exp, ms(t)))
        }
        val crnn = CrnnRunner(models)
        for ((model, sample) in expectedRead) {
            val g = RegionGroups.all.first { it.sample == sample }
            val truth = if (model == 55) "3 43466" else g.sampleTruth
            val t = System.nanoTime()
            val r = crnn.read(model, RgbImage.fromRaw(files.bytes(sample), 96, 48, bgr = true))
            onItem(SelfTestItem("rec_$model OCR on rec_$sample", truth, "${r.text} (${(r.conf * 100).toInt()}%)", r.text == truth, ms(t)))
        }
        // Detector samples: raw RGB letterboxed inputs shipped with the SDK
        for ((id, sample, size) in listOf(Triple(Ids.DET_320, 74, 320), Triple(Ids.DET_640, 77, 640))) {
            val det = models.detector(id)
            val img = RgbImage.fromRaw(files.bytes(sample), size, size, bgr = false)
            val t = System.nanoTime()
            val out = det.infer(img.toTensor(1f / 255f))
            val d = YoloDecoder.decode(out, det.anchors, Letterbox(size, size, size), 0.25f, 0.4f, 0.45f, 8, 3, true)
            val plates = d.count { it.cls == DetClass.PLATE }
            onItem(SelfTestItem("rec_$id detector on rec_$sample", "≥1 plate", "$plates plate(s), ${d.size - plates} vehicle(s)", plates >= 1, ms(t)))
        }
        // MMC on the bundled 224×224 vehicle (stored RGB)
        val veh = RgbImage.fromRaw(files.bytes(17), 224, 224, bgr = false)
        val t = System.nanoTime()
        val a = VehicleAnalyzer(models).analyze(veh, Box(0f, 0f, 224f, 224f), cfg.copy(mmcMakeModel = true, mmcColour = true, mmcType = true))
        onItem(SelfTestItem("rec_15 make/model on rec_17", "Mitsubishi Outlander", a.makeModel?.label ?: "-", a.makeModel?.label == "Mitsubishi Outlander", ms(t)))
        onItem(SelfTestItem("rec_15 pose on rec_17", "Rear", a.pose?.label ?: "-", a.pose?.label == "Rear", 0f))
        onItem(SelfTestItem("rec_21 type on rec_17", "SUV", a.type?.label ?: "-", a.type?.label == "SUV", 0f))
        onItem(SelfTestItem("rec_19 colour on rec_17", "White / Grey", a.colour?.label ?: "-", a.colour?.label in setOf("White", "Grey", "Silver"), 0f))
    }

    private fun ms(t0: Long) = (System.nanoTime() - t0) / 1e6f
}
