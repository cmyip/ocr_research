package com.axios.lpr.ui.scan

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.TextMeasurer
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.axios.lpr.engine.Box
import com.axios.lpr.engine.CoordinateMapper
import com.axios.lpr.engine.LiveConfig
import com.axios.lpr.ui.theme.AxiosColors

/**
 * Draws each tracked plate's box labelled with its (vote-stabilised) reading and confidence,
 * vehicle boxes with MMC results, optional rec_71 corner quads, and the ROI guide.
 */
@Composable
fun PlateOverlay(live: LiveState, cfg: LiveConfig, viewW: Float, viewH: Float, roi: Box?, modifier: Modifier) {
    val tm = rememberTextMeasurer()
    Canvas(modifier) {
        if (roi != null) drawRoi(roi)
        if (live.frameW == 0 || viewW == 0f) return@Canvas
        val m = CoordinateMapper(live.frameW, live.frameH, viewW, viewH, cfg.useFrontCamera && !live.simulated)
        if (cfg.showVehicles) live.vehicles.forEach { v ->
            val b = m.map(v.box)
            drawRect(AxiosColors.Vehicle, Offset(b.x1, b.y1), Size(b.w, b.h), style = Stroke(2.dp.toPx(), pathEffect = PathEffect.dashPathEffect(floatArrayOf(18f, 10f))))
            v.label?.let { label(tm, it, b.x1, b.y2, AxiosColors.Vehicle, Color.White, 12f, below = true) }
        }
        live.tracks.forEach { t ->
            val b = m.map(t.box)
            val colour = when {
                t.recorded -> AxiosColors.Stable
                t.stable -> AxiosColors.Scan
                else -> AxiosColors.Plate
            }
            drawRoundRect(colour, Offset(b.x1, b.y1), Size(b.w, b.h), CornerRadius(6f), style = Stroke(3.dp.toPx()))
            if (cfg.showCorners) t.corners?.let { q ->
                val p = Path().apply {
                    moveTo(m.x(q[0]), m.y(q[1])); lineTo(m.x(q[2]), m.y(q[3])); lineTo(m.x(q[4]), m.y(q[5])); lineTo(m.x(q[6]), m.y(q[7])); close()
                }
                drawPath(p, Color.Magenta, style = Stroke(1.5.dp.toPx()))
            }
            val text = (t.text.ifBlank { "…" }) + "  " + (t.conf * 100).toInt() + "%" + if (t.recorded) "  ✓" else ""
            label(tm, text, b.x1, b.y1, colour, AxiosColors.Ink, 18f, below = false)
            t.region?.let { label(tm, it, b.x1, b.y2, Color(0xCC000000), Color.White, 11f, below = true) }
        }
    }
}

private fun DrawScope.drawRoi(roi: Box) {
    val dim = Color(0x66000000)
    drawRect(dim, Offset.Zero, Size(size.width, roi.y1))
    drawRect(dim, Offset(0f, roi.y2), Size(size.width, size.height - roi.y2))
    drawRect(dim, Offset(0f, roi.y1), Size(roi.x1, roi.h))
    drawRect(dim, Offset(roi.x2, roi.y1), Size(size.width - roi.x2, roi.h))
    drawRoundRect(Color.White, Offset(roi.x1, roi.y1), Size(roi.w, roi.h), CornerRadius(12f), style = Stroke(2.dp.toPx()))
}

/** Filled tag with text, anchored above (or below) a box edge and kept on screen. */
private fun DrawScope.label(tm: TextMeasurer, text: String, x: Float, y: Float, bg: Color, fg: Color, sp: Float, below: Boolean) {
    val style = TextStyle(color = fg, fontSize = sp.sp, fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
    val layout = tm.measure(text, style)
    val padX = 6.dp.toPx(); val padY = 3.dp.toPx()
    val w = layout.size.width + padX * 2
    val h = layout.size.height + padY * 2
    val lx = x.coerceIn(0f, (size.width - w).coerceAtLeast(0f))
    val ly = (if (below) y + 2 else y - h - 2).coerceIn(0f, (size.height - h).coerceAtLeast(0f))
    drawRoundRect(bg, Offset(lx, ly), Size(w, h), CornerRadius(6f))
    drawText(layout, topLeft = Offset(lx + padX, ly + padY))
}
