package com.axios.lpr.engine

import kotlinx.serialization.Serializable

enum class OcrMode(val label: String, val help: String) {
    PINNED("Pinned model", "Always use one CRNN"),
    ROUTED("Region-routed", "Region classifiers pick the group's CRNN; falls back to ensemble"),
    ENSEMBLE_MAX("Ensemble: max confidence", "All enabled 0-9A-Z CRNNs, most confident wins"),
    ENSEMBLE_VOTE("Ensemble: vote", "All enabled 0-9A-Z CRNNs, confidence-weighted vote on the text"),
}

enum class OrtProvider(val label: String) { CPU("CPU"), XNNPACK("XNNPACK"), NNAPI("NNAPI") }

/**
 * Every knob of the pipeline, including the per-model on/off toggles.
 * Persisted as JSON in DataStore and snapshotted into each capture record.
 */
@Serializable
data class PipelineConfig(
    // Detector
    val detector640: Boolean = false,
    val plateScore: Float = 0.25f,
    val vehicleScore: Float = 0.40f,
    val nmsIou: Float = 0.45f,
    val maxPlates: Int = 6,
    val maxVehicles: Int = 3,
    val detectVehicles: Boolean = true,

    // OCR
    val ocrMode: OcrMode = OcrMode.PINNED,
    val pinnedCrnn: Int = Ids.DEFAULT_CRNN,
    val enabledCrnns: Set<Int> = Ids.CRNNS.toSet(),
    /** Run every enabled CRNN for comparison (snap / import only; live uses the strategy alone). */
    val compareAllOnCapture: Boolean = true,

    // Region routing
    val regionEnabled: Boolean = true,
    val enabledClassifiers: Set<Int> = Ids.REGION_CLASSIFIERS.toSet(),

    // Geometry
    val rectify: Boolean = true,
    val cornerMarginX: Float = 0.2f,
    val cornerMarginY: Float = 0.5f,
    val cropPad: Float = 0.03f,
    val deshear: Float = 0.0f,

    // MMC (make/model/colour) + vehicle helpers — off by default (rec_15 is the heaviest model)
    val mmcMakeModel: Boolean = false,
    val mmcColour: Boolean = false,
    val mmcType: Boolean = false,
    val vehicleBoxRefine: Boolean = false,

    val lookups: Boolean = true,

    // Runtimes
    val mnnBackend: MnnBackend = MnnBackend.CPU,
    val mnnThreads: Int = 4,
    val mnnPrecision: MnnPrecision = MnnPrecision.NORMAL,
    val ortProvider: OrtProvider = OrtProvider.CPU,
    val ortThreads: Int = 4,
) {
    val detectorId: Int get() = if (detector640) Ids.DET_640 else Ids.DET_320
    val anyMmc: Boolean get() = mmcMakeModel || mmcColour || mmcType

    /** Models this configuration may run, used by the toggle UI and to unload the rest. */
    fun activeModels(capture: Boolean): Set<Int> = buildSet {
        add(detectorId)
        if (rectify) add(Ids.CORNERS)
        if (regionEnabled || ocrMode == OcrMode.ROUTED) addAll(enabledClassifiers)
        when (ocrMode) {
            OcrMode.PINNED -> add(pinnedCrnn)
            OcrMode.ROUTED -> RegionGroups.all.mapNotNull { it.crnn }.filter { it in enabledCrnns }.forEach(::add)
            else -> addAll(enabledCrnns)
        }
        if (capture && compareAllOnCapture) addAll(enabledCrnns)
        if (detectVehicles) {
            if (mmcMakeModel) add(Ids.MMR)
            if (mmcColour) add(Ids.COLOUR)
            if (mmcType) add(Ids.VTYPE)
            if (vehicleBoxRefine) add(Ids.VBOX)
        }
    }

    companion object {
        /** Verified on all three Malaysian samples: rectify + rec_57, no shear. */
        val MALAYSIA = PipelineConfig()

        /** The original PoC settings from read_plate.py (--ocr-model 57 --deshear 0.3). */
        val POC_SHEAR = PipelineConfig(rectify = false, deshear = 0.3f, cropPad = 0.02f)

        val AUTO_ROUTED = PipelineConfig(ocrMode = OcrMode.ROUTED)

        val EVERYTHING = PipelineConfig(
            ocrMode = OcrMode.ROUTED, mmcMakeModel = true, mmcColour = true, mmcType = true, vehicleBoxRefine = true,
        )

        val presets = listOf(
            "Malaysia (verified)" to MALAYSIA,
            "PoC shear 0.3" to POC_SHEAR,
            "Auto region routing" to AUTO_ROUTED,
            "Everything on" to EVERYTHING,
        )
    }
}

/** Live-scan and recording behaviour. */
@Serializable
data class LiveConfig(
    val frameSkip: Int = 0,
    val stableFrames: Int = 3,
    val minRecordConf: Float = 0.90f,
    val dedupeSeconds: Int = 30,
    val autoRecord: Boolean = false,
    val beep: Boolean = true,
    val haptic: Boolean = true,
    val showVehicles: Boolean = true,
    val showCorners: Boolean = false,
    val showHud: Boolean = true,
    val roiOnly: Boolean = false,
    val analysisWidth: Int = 1280,
    val tagLocation: Boolean = false,
    val useFrontCamera: Boolean = false,
)
