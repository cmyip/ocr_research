package com.axios.lpr.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

/** Plate yellow + scan cyan on a slate background. */
object AxiosColors {
    val Plate = Color(0xFFFACC15)
    val Scan = Color(0xFF22D3EE)
    val Stable = Color(0xFF4ADE80)
    val Vehicle = Color(0xFF60A5FA)
    val Warn = Color(0xFFF97316)
    val Ink = Color(0xFF0F172A)
}

private val Dark = darkColorScheme(
    primary = AxiosColors.Plate, onPrimary = AxiosColors.Ink,
    secondary = AxiosColors.Scan, onSecondary = AxiosColors.Ink,
    tertiary = AxiosColors.Stable,
    background = Color(0xFF0B1016), onBackground = Color(0xFFE2E8F0),
    surface = Color(0xFF111821), onSurface = Color(0xFFE2E8F0),
    surfaceVariant = Color(0xFF1C2530), onSurfaceVariant = Color(0xFFA7B3C2),
    surfaceContainer = Color(0xFF151D27), surfaceContainerHigh = Color(0xFF1A2330),
    outline = Color(0xFF3B4757),
)

val PlateText = TextStyle(fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold, fontSize = 22.sp, letterSpacing = 1.5.sp)

/** Always dark: it's a camera app and plate overlays read best on dark chrome. */
@Composable
fun AxiosTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = Dark, typography = Typography(), content = content)
}
