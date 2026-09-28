package com.axios.lpr.engine

import kotlinx.serialization.Serializable
import kotlin.math.max
import kotlin.math.min

@Serializable
data class Box(val x1: Float, val y1: Float, val x2: Float, val y2: Float) {
    val w: Float get() = x2 - x1
    val h: Float get() = y2 - y1
    val cx: Float get() = (x1 + x2) / 2
    val cy: Float get() = (y1 + y2) / 2
    val area: Float get() = max(0f, w) * max(0f, h)

    fun iou(o: Box): Float {
        val ix = max(0f, min(x2, o.x2) - max(x1, o.x1))
        val iy = max(0f, min(y2, o.y2) - max(y1, o.y1))
        val inter = ix * iy
        val u = area + o.area - inter
        return if (u <= 0f) 0f else inter / u
    }

    fun contains(x: Float, y: Float) = x in x1..x2 && y in y1..y2
    fun expand(fx: Float, fy: Float) = Box(x1 - w * fx, y1 - h * fy, x2 + w * fx, y2 + h * fy)
    fun clip(w: Int, h: Int) = Box(x1.coerceIn(0f, w.toFloat()), y1.coerceIn(0f, h.toFloat()), x2.coerceIn(0f, w.toFloat()), y2.coerceIn(0f, h.toFloat()))
    fun scale(s: Float) = Box(x1 * s, y1 * s, x2 * s, y2 * s)
    fun toList() = listOf(x1, y1, x2, y2)
}

enum class DetClass { PLATE, VEHICLE }

@Serializable
data class Detection(val cls: DetClass, val score: Float, val box: Box)
