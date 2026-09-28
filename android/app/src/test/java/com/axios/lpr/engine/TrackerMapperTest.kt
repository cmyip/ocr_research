package com.axios.lpr.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class TrackerMapperTest {
    private val dummy = RgbImage.filled(2, 2, 0)
    private fun plate(text: String, conf: Float, x: Float) = PlateResult(
        0, Detection(DetClass.PLATE, 0.8f, Box(x, 10f, x + 100f, 40f)), text, conf, emptyList(), 57, "t",
        emptyList(), null, PlateCrops(dummy, dummy, null, dummy, null, 0f), emptyList(), null, 0f, 0.03f,
    )

    @Test fun trackerVotesAcrossFramesAndBecomesStable() {
        val tr = PlateTracker()
        tr.update(listOf(plate("BRL4104", 0.95f, 0f)))
        tr.update(listOf(plate("BRL404", 0.99f, 2f)))
        val live = tr.update(listOf(plate("BRL4104", 0.95f, 4f)))
        assertEquals(1, live.size)
        assertEquals("BRL4104", live[0].vote().first)
        assertFalse(tr.isStable(live[0], 4, 0.9f))
        tr.update(listOf(plate("BRL4104", 0.95f, 5f)))
        assertTrue(tr.isStable(tr.tracks[0], 4, 0.6f))
    }

    @Test fun trackerStartsNewTrackForDistantBoxAndDropsStale() {
        val tr = PlateTracker(maxMisses = 1)
        tr.update(listOf(plate("A1", 0.9f, 0f)))
        tr.update(listOf(plate("B2", 0.9f, 500f)))
        assertEquals(2, tr.tracks.size)
        tr.update(listOf(plate("B2", 0.9f, 500f)))
        assertEquals(1, tr.tracks.size)
    }

    @Test fun differentPlateAtSameSpotStartsNewTrack() {
        val tr = PlateTracker()
        repeat(5) { tr.update(listOf(plate("EV232", 0.99f, 0f))) }
        val live = tr.update(listOf(plate("PUTRAJAYA541", 0.95f, 0f)))
        assertEquals("PUTRAJAYA541", live.single().vote().first)
        // A near-miss read (one char off) stays on the same track and is outvoted
        val tr2 = PlateTracker()
        repeat(5) { tr2.update(listOf(plate("BRL4104", 0.97f, 0f))) }
        val l2 = tr2.update(listOf(plate("BRL404", 0.99f, 0f)))
        assertEquals("BRL4104", l2.single().vote().first)
        assertEquals(2, PlateTracker.levenshtein("BRL4104", "BR404"))
    }

    @Test fun mapperFillCenterAndMirror() {
        // 16:9 analysis frame on a 9:16 view → scale by height, crop sides
        val m = CoordinateMapper(1280, 720, 1080f, 1920f)
        assertEquals(1920f / 720f, m.scale, 1e-4f)
        val b = m.map(Box(640f, 0f, 640f, 720f))
        assertEquals(540f, b.x1, 0.01f)
        assertEquals(1920f, b.y2, 0.01f)
        val mm = CoordinateMapper(100, 100, 100f, 100f, mirror = true)
        assertEquals(90f, mm.map(Box(0f, 0f, 10f, 10f)).x1, 1e-4f)
        val back = m.unmap(m.map(Box(10f, 20f, 30f, 40f)))
        assertEquals(10f, back.x1, 1e-3f); assertEquals(40f, back.y2, 1e-3f)
    }

    @Test fun lookupsGiveFormatHints() {
        val l = Lookups(FileModelFiles())
        assertTrue(l.hints("990SKR09").any { it.table == "Kazakh region" && it.value.contains("Karagandy") })
        assertTrue(l.hints("M1234AB").any { it.table == "Spanish province" })
        assertTrue(l.hints("BRL4104").none { it.table == "Kazakh region" })
        // Routed to the Malaysia/Singapore group → no foreign-table hints
        assertTrue(l.hints("BRL4104", routedGroup = "1187").isEmpty())
        assertTrue(l.hints("990SKR09", routedGroup = "1106").any { it.table == "Kazakh region" })
    }

    @Test fun catalogCoversAllRecordsAndPipelineIds() {
        val c = FileModelFiles().catalog
        assertEquals(87, c.records.size)
        Ids.CRNNS.forEach { assertEquals("ocr", c[it].role) }
        Ids.REGION_CLASSIFIERS.forEach { assertEquals("region", c[it].role) }
        assertEquals(37, c[57].classes)
        RegionGroups.all.forEach { assertEquals("idlist", c[it.idList].role) }
    }

    @Test fun labelParsingKeepsMakeAndDropsCodes() {
        assertEquals("Toyota Corolla Cross", VehicleAnalyzer.top(floatArrayOf(1f), listOf("Toyota\tCorolla Cross")).label)
        assertEquals("White", VehicleAnalyzer.top(floatArrayOf(1f), listOf("white\tWhite")).label)
        assertEquals("Large Truck", VehicleAnalyzer.top(floatArrayOf(1f), listOf("BIGTRUCK\tLarge Truck")).label)
    }
}
