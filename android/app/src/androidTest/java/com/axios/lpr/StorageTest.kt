package com.axios.lpr

import android.graphics.BitmapFactory
import androidx.room.Room
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.axios.lpr.data.CaptureMeta
import com.axios.lpr.data.CaptureRepository
import com.axios.lpr.data.Exporter
import com.axios.lpr.data.LprDatabase
import com.axios.lpr.engine.AndroidModelStore
import com.axios.lpr.engine.ModelManager
import com.axios.lpr.engine.PipelineConfig
import com.axios.lpr.engine.PlatePipeline
import com.axios.lpr.engine.toRgbImage
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.zip.ZipFile

@RunWith(AndroidJUnit4::class)
class StorageTest {
    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext
    private lateinit var db: LprDatabase
    private lateinit var repo: CaptureRepository
    private lateinit var models: ModelManager

    @Before fun setUp() {
        db = Room.inMemoryDatabaseBuilder(ctx, LprDatabase::class.java).build()
        repo = CaptureRepository(ctx, db)
        models = ModelManager(AndroidModelStore(ctx))
    }

    @After fun tearDown() { db.close(); models.close() }

    @Test fun snapIsStoredWithImagesAndCorrectionKeepsPrediction() = runBlocking {
        val bmp = InstrumentationRegistry.getInstrumentation().context.assets.open("BRL4104.jpeg").use { BitmapFactory.decodeStream(it) }
        val img = bmp.toRgbImage()
        val cfg = PipelineConfig.MALAYSIA.copy(mmcMakeModel = true, mmcColour = true, mmcType = true)
        val frame = PlatePipeline(models).process(img, cfg, capture = true)
        val id = repo.save(frame, img, cfg, CaptureMeta(source = "snap"))

        val cap = repo.dao.capture(id)!!
        assertEquals(1, cap.plates.size)
        val plate = cap.plates[0].plate
        assertEquals("BRL4104", plate.predictedText)
        assertEquals(20, cap.plates[0].reads.size)
        assertNull(plate.correctedText)
        for (path in listOf(cap.capture.imagePath, plate.rawCropPath, plate.inputCropPath, plate.rectifiedCropPath!!)) {
            assertTrue(path, repo.file(path).length() > 0)
        }
        val input = BitmapFactory.decodeFile(repo.file(plate.inputCropPath).path)
        assertEquals(96, input.width); assertEquals(48, input.height)
        assertTrue(cap.vehicles.isNotEmpty())
        assertTrue(cap.vehicles[0].makeModel!!.contains("Corolla Cross"))
        assertEquals(cap.vehicles[0].id, plate.vehicleId)

        // Correction is stored separately; the prediction stays untouched.
        repo.setCorrection(plate.id, " brl 4104x ")
        val corrected = repo.dao.plate(plate.id)!!
        assertEquals("BRL 4104X", corrected.correctedText)
        assertEquals("BRL4104", corrected.predictedText)
        assertNotNull(corrected.correctedAt)
        assertEquals(1, repo.dao.correctedCount().first())
        assertEquals(1, repo.dao.plates(needsReview = false, onlyCorrected = true).first().size)
        assertEquals(0, repo.dao.plates(needsReview = true).first().size)
        assertEquals(1, repo.dao.plates(q = "4104X").first().size)

        // Exports
        val exporter = Exporter(ctx, repo)
        val csv = exporter.csv().readText()
        assertTrue(csv.lines()[0].startsWith("plate_id,capture_id"))
        assertTrue(csv.contains("BRL4104") && csv.contains("BRL 4104X"))
        val (zipFile, n) = exporter.trainingZip(onlyCorrected = true)
        assertEquals(1, n)
        ZipFile(zipFile).use { z ->
            val labels = z.getInputStream(z.getEntry("labels.csv")).bufferedReader().readText()
            assertTrue(labels.contains("BRL 4104X"))
            assertNotNull(z.getEntry("images/${plate.id}_input96x48.png"))
        }
        assertTrue(exporter.json().readText().contains("\"corrected\":\"BRL 4104X\""))

        // Clearing the correction
        repo.setCorrection(plate.id, "  ")
        assertNull(repo.dao.plate(plate.id)!!.correctedText)

        // Cascade delete removes rows and files
        val dir = File(ctx.filesDir, cap.capture.dir)
        assertTrue(dir.exists())
        repo.delete(id)
        assertNull(repo.dao.capture(id))
        assertNull(repo.dao.plate(plate.id))
        assertFalse(dir.exists())
    }
}
