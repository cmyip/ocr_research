package com.axios.lpr.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RgbImageTest {
    private fun gradient(w: Int, h: Int) = RgbImage(w, h, ByteArray(w * h * 3) { i -> ((i / 3 % w) * 255 / (w - 1)).toByte() })

    @Test fun resizeOfConstantImageStaysConstant() {
        val img = RgbImage.filled(97, 53, 77)
        val r = img.resize(31, 17, reducingGap = 2f)
        assertTrue(r.data.all { (it.toInt() and 0xFF) == 77 })
    }

    @Test fun cropRoundsHalfEvenAndZeroFillsOutside() {
        val img = RgbImage.filled(10, 10, 200)
        val c = img.crop(-2.5f, 0f, 3.5f, 2f) // Python round(-2.5) = -2, round(3.5) = 4
        assertEquals(6, c.width)
        assertEquals(0, c.get(0, 0, 0))
        assertEquals(0, c.get(1, 0, 0))
        assertEquals(200, c.get(2, 0, 0))
    }

    @Test fun reduceAveragesBlocks() {
        val img = RgbImage(4, 2, byteArrayOf(0, 0, 0, 10, 10, 10, 20, 20, 20, 30, 30, 30, 0, 0, 0, 10, 10, 10, 20, 20, 20, 30, 30, 30))
        val r = img.reduce(2)
        assertEquals(2, r.width); assertEquals(1, r.height)
        assertEquals(5, r.get(0, 0, 0)); assertEquals(25, r.get(1, 0, 0))
    }

    @Test fun zeroShearIsIdentityAndQuadOfFullFrameIsIdentity() {
        val img = gradient(40, 20)
        assertTrue(img.deshear(0f) === img)
        val q = img.quad(40, 20, floatArrayOf(0f, 0f, 0f, 20f, 40f, 20f, 40f, 0f))
        assertTrue(q.data.contentEquals(img.data))
    }

    @Test fun rotateRoundTrips() {
        val img = gradient(7, 3)
        assertTrue(img.rotate(90).rotate(270).data.contentEquals(img.data))
        assertEquals(3, img.rotate(90).width)
    }

    @Test fun tensorLayoutIsNchwWithOptionalBgr() {
        val img = RgbImage(1, 1, byteArrayOf(10, 20, 30))
        assertEquals(listOf(10f, 20f, 30f), img.toTensor(1f).toList())
        assertEquals(listOf(30f, 20f, 10f), img.toTensor(1f, bgr = true).toList())
        assertEquals(-1f, RgbImage(1, 1, byteArrayOf(0, 0, 0)).toTensor(1 / 127.5f, -1f)[0], 1e-6f)
    }
}
