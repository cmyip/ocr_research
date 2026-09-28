package com.axios.lpr.engine

import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ConfigTest {
    @Test fun mmcOffByDefaultAndNotActive() {
        val c = PipelineConfig()
        assertFalse(c.anyMmc)
        val active = c.activeModels(capture = true)
        assertTrue(Ids.MMR !in active && Ids.COLOUR !in active && Ids.VTYPE !in active)
        assertTrue(Ids.DET_320 in active && Ids.CORNERS in active && 57 in active)
    }

    @Test fun togglesMapToConfig() {
        var c = PipelineConfig()
        c = ModelToggles.set(Ids.MMR, true, c)
        assertEquals(true, ModelToggles.isOn(Ids.MMR, c))
        assertTrue(Ids.MMR in c.activeModels(false))
        c = ModelToggles.set(Ids.DET_640, true, c)
        assertEquals(false, ModelToggles.isOn(Ids.DET_320, c))
        assertTrue(Ids.DET_640 in c.activeModels(false) && Ids.DET_320 !in c.activeModels(false))
        c = ModelToggles.set(57, false, c)
        assertTrue(57 !in c.enabledCrnns)
        assertTrue(c.pinnedCrnn != 57 && c.pinnedCrnn in c.enabledCrnns)
        c = ModelToggles.set(Ids.CORNERS, false, c)
        assertFalse(c.rectify)
        assertTrue(ModelToggles.isOn(3, c) == null)
    }

    @Test fun configRoundTripsThroughJson() {
        val json = Json { encodeDefaults = true; ignoreUnknownKeys = true }
        val c = PipelineConfig.EVERYTHING.copy(mnnBackend = MnnBackend.VULKAN, enabledCrnns = setOf(50, 57))
        val back = json.decodeFromString(PipelineConfig.serializer(), json.encodeToString(PipelineConfig.serializer(), c))
        assertEquals(c, back)
        assertEquals(PipelineConfig(), json.decodeFromString(PipelineConfig.serializer(), "{\"unknown\":1}"))
    }
}
