package com.axios.lpr

import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.axios.lpr.engine.AndroidModelStore
import com.axios.lpr.engine.CrnnRunner
import com.axios.lpr.engine.Ids
import com.axios.lpr.engine.MnnBackend
import com.axios.lpr.engine.ModelManager
import com.axios.lpr.engine.OcrMode
import com.axios.lpr.engine.PipelineConfig
import com.axios.lpr.engine.PlatePipeline
import com.axios.lpr.engine.RegionGroups
import com.axios.lpr.engine.RegionRouter
import com.axios.lpr.engine.RgbImage
import com.axios.lpr.engine.toRgbImage
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.AfterClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Runs the real models on-device and checks them against the Python PoC goldens
 * (golden_pipeline.json) and the SDK's bundled sample inputs.
 */
@RunWith(AndroidJUnit4::class)
class PipelineParityTest {
    companion object {
        lateinit var store: AndroidModelStore
        lateinit var models: ModelManager
        lateinit var pipeline: PlatePipeline
        lateinit var golden: JsonObject
        val samples = listOf("BRL4104", "EV232", "PUTRAJAYA541")

        @BeforeClass @JvmStatic fun setUp() {
            val inst = InstrumentationRegistry.getInstrumentation()
            store = AndroidModelStore(inst.targetContext)
            models = ModelManager(store)
            pipeline = PlatePipeline(models)
            golden = Json.parseToJsonElement(inst.context.assets.open("golden_pipeline.json").bufferedReader().readText()).jsonObject
        }

        @AfterClass @JvmStatic fun tearDown() = models.close()

        fun sample(name: String): RgbImage {
            val ctx = InstrumentationRegistry.getInstrumentation().context
            val bmp = ctx.assets.open("$name.jpeg").use { BitmapFactory.decodeStream(it) }
            return bmp.toRgbImage()
        }
    }

    private fun g(s: String) = golden.getValue(s).jsonObject
    private fun floats(s: String, k: String) = g(s).getValue(k).jsonArray.map { it.jsonPrimitive.double.toFloat() }

    @Test fun defaultPresetReadsAllThreeMalaysianSamples() {
        for (s in samples) {
            val r = pipeline.process(sample(s), PipelineConfig.MALAYSIA, capture = false)
            val p = r.plates.first()
            assertEquals("$s text", s, p.text)
            assertEquals(57, p.model)
            assertTrue("$s rectified", p.rectified)
            val box = floats(s, "box")
            listOf(p.det.box.x1, p.det.box.y1, p.det.box.x2, p.det.box.y2).zip(box).forEach { (a, e) -> assertEquals("$s box", e, a, 3f) }
            val corners = floats(s, "corners")
            p.crops.corners!!.toList().zip(corners).forEach { (a, e) -> assertEquals("$s corners", e, a, 3f) }
        }
    }

    @Test fun pocShearPresetMatchesReadPlatePy() {
        val expected = mapOf("BRL4104" to "BRL4104", "EV232" to "EV232", "PUTRAJAYA541" to "PUTRAJAYA1541")
        for (s in samples) {
            val p = pipeline.process(sample(s), PipelineConfig.POC_SHEAR, capture = false).plates.first()
            assertEquals("$s PoC read", expected[s], p.text)
            assertTrue(!p.rectified)
        }
    }

    @Test fun regionRoutingSendsMalaysianPlateToGroup1187AndRec57() {
        val r = pipeline.process(sample("BRL4104"), PipelineConfig.AUTO_ROUTED, capture = false).plates.first()
        assertEquals("1187", r.region?.chosen?.group)
        assertEquals("1125", r.region?.chosen?.label)
        assertEquals(57, r.model)
        assertEquals("BRL4104", r.text)
    }

    @Test fun captureModeComparesEveryEnabledCrnn() {
        val r = pipeline.process(sample("EV232"), PipelineConfig.MALAYSIA, capture = true).plates.first()
        assertEquals(20, r.reads.size)
        val gr = g("EV232").getValue("rect_reads").jsonObject
        for (id in listOf(50, 53, 57, 63, 66)) {
            val read = r.reads.first { it.model == id }
            assertEquals("rec_$id", gr.getValue(id.toString()).jsonArray[0].jsonPrimitive.content, read.text)
        }
    }

    @Test fun bundledSamplesThroughRegionClassifiersAndCrnns() {
        val router = RegionRouter(models)
        val expectedRegion = mapOf(23 to "1222\t2033", 26 to "1195", 29 to "1187", 32 to "1012\t2207", 35 to "1110",
            38 to "1106", 41 to "4001", 44 to "1044", 47 to "4416", 82 to "4101")
        for (g in RegionGroups.all) {
            val crop = RgbImage.fromRaw(store.bytes(g.sample), 96, 48, bgr = true)
            val d = router.classify(crop, setOf(g.classifier))
            assertEquals("rec_${g.classifier}", expectedRegion[g.classifier], d.reads.single().label)
        }
        val crnn = CrnnRunner(models)
        val expectedRead = mapOf(50 to (25 to "FHW7186"), 57 to (31 to "SH7194K"), 53 to (34 to "RTL015"), 65 to (28 to "0907CDS"), 55 to (37 to "3 43466"))
        for ((model, pair) in expectedRead) {
            val crop = RgbImage.fromRaw(store.bytes(pair.first), 96, 48, bgr = true)
            assertEquals("rec_$model on rec_${pair.first}", pair.second, crnn.read(model, crop).text)
        }
    }

    @Test fun mmcIdentifiesTheCorollaCrossAndIsSkippedWhenOff() {
        val on = pipeline.process(sample("BRL4104"), PipelineConfig.EVERYTHING, capture = false)
        val attrs = on.vehicles.first().attrs!!
        assertTrue(attrs.makeModel!!.label, attrs.makeModel!!.label.contains("Corolla Cross"))
        assertEquals("Frontal", attrs.pose!!.label)
        assertTrue(Ids.MMR in on.modelsRun)

        models.retainOnly(emptySet())
        val off = pipeline.process(sample("BRL4104"), PipelineConfig.MALAYSIA, capture = false)
        assertTrue(off.vehicles.all { it.attrs == null })
        assertTrue(Ids.MMR !in off.modelsRun && !models.isLoaded(Ids.MMR))
        assertTrue(!models.isLoaded(Ids.COLOUR) && !models.isLoaded(Ids.VTYPE))
    }

    @Test fun gpuBackendsFallBackCleanlyAndAgree() {
        for (b in listOf(MnnBackend.OPENCL, MnnBackend.VULKAN, MnnBackend.AUTO)) {
            val cfg = PipelineConfig.MALAYSIA.copy(mnnBackend = b)
            val p = pipeline.process(sample("BRL4104"), cfg, capture = false).plates.first()
            assertEquals("backend $b", "BRL4104", p.text)
        }
        pipeline.process(sample("BRL4104"), PipelineConfig.MALAYSIA, capture = false)
    }

    @Test fun ensembleModesAndDetector640Run() {
        for (mode in OcrMode.entries) {
            val p = pipeline.process(sample("EV232"), PipelineConfig.MALAYSIA.copy(ocrMode = mode), capture = false).plates.first()
            assertEquals("mode $mode", "EV232", p.text)
        }
        val r = pipeline.process(sample("EV232"), PipelineConfig.MALAYSIA.copy(detector640 = true), capture = false)
        assertTrue(r.plates.isNotEmpty())
    }
}
