package com.axios.lpr.engine

data class PlateResult(
    val index: Int,
    val det: Detection,
    val text: String,
    val conf: Float,
    val charConf: List<Float>,
    val model: Int?,
    val strategy: String,
    val reads: List<OcrRead>,
    val region: RegionDecision?,
    val crops: PlateCrops,
    val hints: List<LookupHint>,
    val vehicleIndex: Int?,
    val deshear: Float,
    val pad: Float,
) {
    val rectified: Boolean get() = crops.rectified != null
}

data class VehicleResult(val index: Int, val det: Detection, val attrs: VehicleAttrs?)

data class FrameResult(
    val width: Int,
    val height: Int,
    val plates: List<PlateResult>,
    val vehicles: List<VehicleResult>,
    val timings: Map<String, Float>,
    val modelsRun: Set<Int>,
) {
    val totalMs: Float get() = timings["total"] ?: 0f
}

/**
 * Image → plates (+ vehicles). Port of read_plate.read_plate() extended with multi-plate NMS,
 * corner rectification, region routing, OCR strategies, MMC and lookup hints.
 * Thread-safe: one frame at a time.
 */
class PlatePipeline(val models: ModelManager) {
    private val crnn = CrnnRunner(models)
    private val router = RegionRouter(models)
    private val rectifier = CornerRectifier(models)
    private val vehicles = VehicleAnalyzer(models)
    val lookups: Lookups by lazy { Lookups(models.files) }
    private val catalog get() = models.files.catalog

    private fun classesOf(id: Int) = catalog[id].classes ?: 0

    @Synchronized
    fun process(img: RgbImage, cfg: PipelineConfig, capture: Boolean, roi: Box? = null): FrameResult {
        models.configure(cfg)
        val timings = LinkedHashMap<String, Float>()
        val ran = LinkedHashSet<Int>()
        val tStart = System.nanoTime()
        fun lap(key: String, t0: Long) { timings[key] = (timings[key] ?: 0f) + (System.nanoTime() - t0) / 1e6f }

        // 1. Detect
        var t = System.nanoTime()
        val det = models.detector(cfg.detectorId)
        ran += cfg.detectorId
        val lb = Letterbox(det.size, img.width, img.height)
        val x = lb.apply(img).toTensor(1f / 255f)
        lap("det_prep", t)
        t = System.nanoTime()
        val out = det.infer(x)
        lap("det_infer", t)
        t = System.nanoTime()
        var dets = YoloDecoder.decode(out, det.anchors, lb, cfg.plateScore, cfg.vehicleScore, cfg.nmsIou, cfg.maxPlates, cfg.maxVehicles, cfg.detectVehicles)
        if (roi != null) dets = dets.filter { it.cls == DetClass.VEHICLE || roi.contains(it.box.cx, it.box.cy) }
        lap("det_decode", t)

        val vehicleDets = dets.filter { it.cls == DetClass.VEHICLE }

        // 2. Plates
        val plates = dets.filter { it.cls == DetClass.PLATE }.mapIndexed { i, d ->
            t = System.nanoTime()
            val (raw, plain96) = PlateCropper.plain(img, d.box, cfg.cropPad, cfg.deshear)
            lap("crop", t)
            var corners: FloatArray? = null
            var rectified: RgbImage? = null
            var ocrInput = plain96
            var cornerMs = 0f
            if (cfg.rectify) {
                t = System.nanoTime()
                corners = rectifier.corners(img, d.box, cfg.cornerMarginX, cfg.cornerMarginY)
                ran += Ids.CORNERS
                if (corners != null) {
                    val (w, in96) = rectifier.warp(img, corners, cfg.cropPad, cfg.deshear)
                    rectified = w; ocrInput = in96
                }
                cornerMs = (System.nanoTime() - t) / 1e6f
                lap("rectify", t)
            }

            t = System.nanoTime()
            val region = if (cfg.regionEnabled || cfg.ocrMode == OcrMode.ROUTED) {
                router.classify(ocrInput, cfg.enabledClassifiers).also { r -> r.reads.forEach { ran += it.classifier } }
            } else null
            lap("region", t)

            t = System.nanoTime()
            val reads = LinkedHashMap<Int, OcrRead>()
            fun read(id: Int): OcrRead = reads.getOrPut(id) { ran += id; crnn.read(id, ocrInput) }
            val latin = cfg.enabledCrnns.filter { classesOf(it) == 37 }.sorted()
            fun ensembleMax(): OcrRead? = latin.map(::read).maxByOrNull { it.conf }

            var strategy: String
            val primary: OcrRead? = when (cfg.ocrMode) {
                OcrMode.PINNED -> { strategy = "pinned rec_${cfg.pinnedCrnn}"; read(cfg.pinnedCrnn) }
                OcrMode.ROUTED -> {
                    val g = region?.group
                    val routed = g?.crnn?.takeIf { it in cfg.enabledCrnns }
                    if (routed != null) { strategy = "routed ${g.name} → rec_$routed"; read(routed) }
                    else { strategy = if (g != null) "routed ${g.name} (no CRNN) → max" else "unrouted → max"; ensembleMax() }
                }
                OcrMode.ENSEMBLE_MAX -> { strategy = "ensemble max"; ensembleMax() }
                OcrMode.ENSEMBLE_VOTE -> { strategy = "ensemble vote"; OcrVote.vote(latin.map(::read)) }
            }
            if (capture && cfg.compareAllOnCapture) cfg.enabledCrnns.sorted().forEach { read(it) }
            lap("ocr", t)

            val text = primary?.text ?: ""
            val veh = vehicleDets.withIndex().filter { it.value.box.contains(d.box.cx, d.box.cy) }.minByOrNull { it.value.box.area }?.index
            PlateResult(
                index = i, det = d, text = text, conf = primary?.conf ?: 0f, charConf = primary?.charConf ?: emptyList(),
                model = primary?.model, strategy = strategy, reads = reads.values.sortedByDescending { it.conf },
                region = region, crops = PlateCrops(raw, plain96, rectified, ocrInput, corners, cornerMs),
                hints = if (cfg.lookups && text.isNotEmpty()) lookups.hints(text, region?.chosen?.group) else emptyList(),
                vehicleIndex = veh, deshear = cfg.deshear, pad = cfg.cropPad,
            )
        }

        // 3. Vehicles + MMC
        t = System.nanoTime()
        val vehicleResults = vehicleDets.mapIndexed { i, d ->
            val attrs = if (cfg.anyMmc || cfg.vehicleBoxRefine) {
                if (cfg.mmcMakeModel) ran += Ids.MMR
                if (cfg.mmcColour) ran += Ids.COLOUR
                if (cfg.mmcType) ran += Ids.VTYPE
                if (cfg.vehicleBoxRefine) ran += Ids.VBOX
                vehicles.analyze(img, d.box, cfg)
            } else null
            VehicleResult(i, d, attrs)
        }
        lap("mmc", t)
        timings["total"] = (System.nanoTime() - tStart) / 1e6f
        return FrameResult(img.width, img.height, plates, vehicleResults, timings, ran)
    }

    /** Apply runtime settings and unload toggled-off models, never in the middle of a frame. */
    @Synchronized
    fun applySettings(cfg: PipelineConfig) {
        models.configure(cfg)
        models.retainOnly(cfg.activeModels(capture = true))
    }

    @Synchronized
    fun unloadAll() = models.retainOnly(emptySet())

    /** Single-model benchmark for Model Lab: runs [runs] inferences on a representative input. */
    @Synchronized
    fun benchmark(id: Int, runs: Int, cfg: PipelineConfig): List<Float> {
        models.configure(cfg)
        val rec = catalog[id]
        // Realistic inputs: the SDK's own sample where one exists, else uniform noise.
        // (A constant tensor drives activations into slow denormal ranges and misleads timings.)
        val rnd = java.util.Random(7)
        return if (rec.kind == "onnx") {
            val d = models.detector(id)
            val x = rec.sample?.let { RgbImage.fromRaw(models.files.bytes(it), d.size, d.size, bgr = false).toTensor(1f / 255f) }
                ?: FloatArray(3 * d.size * d.size) { rnd.nextFloat() }
            d.infer(x)
            List(runs) { val t = System.nanoTime(); d.infer(x); (System.nanoTime() - t) / 1e6f }
        } else {
            val n = models.mnn(id)
            val shape = n.inputShape
            val sample = rec.sample?.let { s -> runCatching { RgbImage.fromRaw(models.files.bytes(s), shape[3], shape[2], bgr = shape[2] == 48) }.getOrNull() }
            val x = when {
                sample != null && rec.role == "region" -> sample.toTensor(1f, 0f, bgr = true)
                sample != null && rec.role == "mmr" -> sample.toTensor(1f / 127.5f, -1f)
                else -> FloatArray(n.inputSize) { rnd.nextFloat() }
            }
            n.run(x.copyOf())
            List(runs) { val t = System.nanoTime(); n.run(x.copyOf()); (System.nanoTime() - t) / 1e6f }
        }
    }
}
