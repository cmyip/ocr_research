package com.axios.lpr.ui.common

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.axios.lpr.ui.theme.AxiosColors
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.ConcurrentHashMap

/** Tiny bitmap cache for thumbnails. */
object BitmapCache {
    private val map = ConcurrentHashMap<String, Bitmap>()
    fun get(path: String, maxSide: Int): Bitmap? = map["$path@$maxSide"] ?: run {
        val f = File(path)
        if (!f.exists()) return null
        val b = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, b)
        var s = 1
        while (maxOf(b.outWidth, b.outHeight) / (s * 2) >= maxSide) s *= 2
        BitmapFactory.decodeFile(path, BitmapFactory.Options().apply { inSampleSize = s })?.also {
            if (map.size > 300) map.clear()
            map["$path@$maxSide"] = it
        }
    }
    fun clear() = map.clear()
}

@Composable
fun FileImage(
    path: String?, modifier: Modifier = Modifier, maxSide: Int = 1024, contentScale: ContentScale = ContentScale.Fit,
    pixelated: Boolean = false,
) {
    var bmp by remember(path, maxSide) { mutableStateOf<Bitmap?>(null) }
    LaunchedEffect(path, maxSide) { bmp = path?.let { withContext(Dispatchers.IO) { BitmapCache.get(it, maxSide) } } }
    val b = bmp
    if (b != null) {
        Image(b.asImageBitmap(), null, modifier, contentScale = contentScale,
            filterQuality = if (pixelated) FilterQuality.None else FilterQuality.Medium)
    } else Box(modifier.background(MaterialTheme.colorScheme.surfaceVariant))
}

@Composable
fun BitmapImage(bitmap: Bitmap, modifier: Modifier = Modifier, pixelated: Boolean = false, contentScale: ContentScale = ContentScale.Fit) {
    Image(bitmap.asImageBitmap(), null, modifier, contentScale = contentScale, filterQuality = if (pixelated) FilterQuality.None else FilterQuality.Medium)
}

@Composable
fun SectionTitle(text: String, modifier: Modifier = Modifier) {
    Text(text.uppercase(), modifier.padding(top = 16.dp, bottom = 6.dp), style = MaterialTheme.typography.labelMedium,
        color = MaterialTheme.colorScheme.secondary, letterSpacing = 1.2.sp)
}

@Composable
fun KeyValue(key: String, value: String, modifier: Modifier = Modifier) {
    Row(modifier.padding(vertical = 2.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(key, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.weight(0.4f))
        Text(value, style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(0.6f))
    }
}

fun confColor(conf: Float): Color = when {
    conf >= 0.95f -> AxiosColors.Stable
    conf >= 0.85f -> AxiosColors.Plate
    else -> AxiosColors.Warn
}

@Composable
fun ConfBadge(conf: Float, modifier: Modifier = Modifier) {
    Text("${(conf * 100).toInt()}%", modifier
        .background(confColor(conf).copy(alpha = 0.18f), RoundedCornerShape(6.dp))
        .padding(horizontal = 6.dp, vertical = 2.dp),
        color = confColor(conf), style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
}

/** Yellow plate-style text chip. */
@Composable
fun PlateChip(text: String, modifier: Modifier = Modifier, big: Boolean = false) {
    Box(modifier.background(AxiosColors.Plate, RoundedCornerShape(6.dp)).padding(horizontal = if (big) 12.dp else 8.dp, vertical = if (big) 6.dp else 3.dp),
        contentAlignment = Alignment.Center) {
        Text(text.ifBlank { "—" }, color = AxiosColors.Ink, fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold,
            fontSize = if (big) 24.sp else 15.sp, letterSpacing = 1.sp)
    }
}

@Composable
fun Caption(text: String, modifier: Modifier = Modifier) {
    Text(text, modifier, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
}

@Composable
fun Labeled(label: String, content: @Composable () -> Unit) {
    Column { Caption(label); content() }
}
