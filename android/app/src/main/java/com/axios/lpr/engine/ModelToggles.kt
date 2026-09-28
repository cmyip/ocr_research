package com.axios.lpr.engine

/** Maps each runnable model to its on/off switch in [PipelineConfig]. */
object ModelToggles {
    fun isOn(id: Int, c: PipelineConfig): Boolean? = when (id) {
        Ids.DET_320 -> !c.detector640
        Ids.DET_640 -> c.detector640
        in Ids.CRNNS -> id in c.enabledCrnns
        in Ids.REGION_CLASSIFIERS -> c.regionEnabled && id in c.enabledClassifiers
        Ids.CORNERS -> c.rectify
        Ids.MMR -> c.mmcMakeModel
        Ids.COLOUR -> c.mmcColour
        Ids.VTYPE -> c.mmcType
        Ids.VBOX -> c.vehicleBoxRefine
        else -> null
    }

    fun set(id: Int, on: Boolean, c: PipelineConfig): PipelineConfig = when (id) {
        Ids.DET_320 -> c.copy(detector640 = !on)
        Ids.DET_640 -> c.copy(detector640 = on)
        in Ids.CRNNS -> {
            val s = if (on) c.enabledCrnns + id else c.enabledCrnns - id
            // Keep the pinned model valid: re-pin to the first remaining 0-9A-Z model if it was switched off.
            c.copy(enabledCrnns = s, pinnedCrnn = if (!on && c.pinnedCrnn == id) (s.sorted().firstOrNull() ?: c.pinnedCrnn) else c.pinnedCrnn)
        }
        in Ids.REGION_CLASSIFIERS -> {
            val s = if (on) c.enabledClassifiers + id else c.enabledClassifiers - id
            c.copy(enabledClassifiers = s, regionEnabled = if (on) true else c.regionEnabled)
        }
        Ids.CORNERS -> c.copy(rectify = on)
        Ids.MMR -> c.copy(mmcMakeModel = on)
        Ids.COLOUR -> c.copy(mmcColour = on)
        Ids.VTYPE -> c.copy(mmcType = on)
        Ids.VBOX -> c.copy(vehicleBoxRefine = on)
        else -> c
    }
}
