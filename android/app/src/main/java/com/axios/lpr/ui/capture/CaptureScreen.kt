package com.axios.lpr.ui.capture

import android.content.Intent
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.FileProvider
import com.axios.lpr.AppContainer
import com.axios.lpr.data.CaptureWithPlates
import com.axios.lpr.data.PlateWithReads
import com.axios.lpr.data.VehicleEntity
import com.axios.lpr.engine.LookupHint
import com.axios.lpr.engine.RegionDecision
import com.axios.lpr.ui.common.Caption
import com.axios.lpr.ui.common.ConfBadge
import com.axios.lpr.ui.common.FileImage
import com.axios.lpr.ui.common.KeyValue
import com.axios.lpr.ui.common.PlateChip
import com.axios.lpr.ui.common.SectionTitle
import com.axios.lpr.ui.common.confColor
import com.axios.lpr.ui.theme.AxiosColors
import kotlinx.coroutines.launch
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.json.Json
import java.text.DateFormat
import java.util.Date

private val json = Json { ignoreUnknownKeys = true }

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CaptureScreen(c: AppContainer, captureId: Long, onBack: () -> Unit) {
    val data by remember(captureId) { c.repo.dao.observeCapture(captureId) }.collectAsState(initial = null)
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var confirmDelete by remember { mutableStateOf(false) }

    Scaffold(topBar = {
        TopAppBar(
            title = { Text(data?.let { "${it.plates.size} plate(s) · ${it.capture.source}" } ?: "Capture") },
            navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") } },
            actions = {
                data?.let { d ->
                    IconButton(onClick = {
                        val f = c.repo.file(d.capture.imagePath)
                        val uri = FileProvider.getUriForFile(context, "${context.packageName}.files", f)
                        val text = d.plates.joinToString(", ") { it.plate.finalText }
                        context.startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
                            type = "image/jpeg"; putExtra(Intent.EXTRA_STREAM, uri); putExtra(Intent.EXTRA_TEXT, text)
                            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                        }, "Share capture"))
                    }) { Icon(Icons.Filled.Share, "Share") }
                    IconButton(onClick = { confirmDelete = true }) { Icon(Icons.Filled.Delete, "Delete") }
                }
            },
        )
    }) { pad ->
        val d = data
        if (d == null) {
            Box(Modifier.fillMaxSize().padding(pad), contentAlignment = Alignment.Center) { Caption("Loading…") }
            return@Scaffold
        }
        LazyColumn(Modifier.fillMaxSize().padding(pad).imePadding().padding(horizontal = 12.dp)) {
            item { FrameWithBoxes(c, d) }
            if (d.plates.isEmpty()) item {
                Card(Modifier.fillMaxWidth().padding(vertical = 8.dp)) {
                    Column(Modifier.padding(16.dp)) {
                        Text("No plate detected", style = MaterialTheme.typography.titleMedium)
                        Caption("Try the 640 detector, a lower plate score threshold, or move closer.")
                    }
                }
            }
            items(d.plates, key = { it.plate.id }) { p -> PlateCard(c, p) }
            if (d.vehicles.isNotEmpty()) item { SectionTitle("Vehicles") }
            items(d.vehicles, key = { "v${it.id}" }) { v -> VehicleCard(c, v) }
            item { CaptureInfo(d) }
            item { Spacer(Modifier.height(40.dp)) }
        }
    }
    if (confirmDelete) AlertDialog(
        onDismissRequest = { confirmDelete = false },
        title = { Text("Delete capture?") },
        text = { Text("Removes the images and every plate record from this capture.") },
        confirmButton = { TextButton(onClick = { confirmDelete = false; scope.launch { c.repo.delete(captureId); onBack() } }) { Text("Delete") } },
        dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text("Cancel") } },
    )
}

@Composable
private fun FrameWithBoxes(c: AppContainer, d: CaptureWithPlates) {
    val cap = d.capture
    Box(Modifier.fillMaxWidth().padding(vertical = 8.dp).aspectRatio(cap.width / cap.height.toFloat())) {
        FileImage(c.repo.file(cap.imagePath).path, Modifier.fillMaxSize(), maxSide = 1600)
        Canvas(Modifier.fillMaxSize()) {
            val s = size.width / cap.width
            d.vehicles.forEach { v -> drawRect(AxiosColors.Vehicle, Offset(v.x1 * s, v.y1 * s), Size((v.x2 - v.x1) * s, (v.y2 - v.y1) * s), style = Stroke(2.dp.toPx())) }
            d.plates.forEach { p ->
                val b = p.plate
                val col = if (b.correctedText != null) AxiosColors.Stable else AxiosColors.Plate
                drawRect(col, Offset(b.x1 * s, b.y1 * s), Size((b.x2 - b.x1) * s, (b.y2 - b.y1) * s), style = Stroke(2.5.dp.toPx()))
            }
        }
    }
}

@Composable
private fun PlateCard(c: AppContainer, pr: PlateWithReads) {
    val p = pr.plate
    val scope = rememberCoroutineScope()
    var field by remember(p.id, p.correctedText) { mutableStateOf(p.correctedText ?: "") }
    val region = remember(p.regionJson) { p.regionJson?.let { runCatching { json.decodeFromString(RegionDecision.serializer(), it) }.getOrNull() } }
    val hints = remember(p.hintsJson) { p.hintsJson?.let { runCatching { json.decodeFromString(ListSerializer(LookupHint.serializer()), it) }.getOrNull() } ?: emptyList() }
    val charConf = remember(p.charConfJson) { runCatching { json.decodeFromString(ListSerializer(Float.serializer()), p.charConfJson) }.getOrDefault(emptyList()) }
    var showRegions by remember { mutableStateOf(false) }

    Card(Modifier.fillMaxWidth().padding(vertical = 6.dp)) {
        Column(Modifier.padding(12.dp)) {
            Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.Bottom) {
                Column { Caption("Detector crop"); FileImage(c.repo.file(p.rawCropPath).path, Modifier.height(48.dp).width(120.dp), maxSide = 600) }
                p.rectifiedCropPath?.let { Column { Caption("Rectified"); FileImage(c.repo.file(it).path, Modifier.height(56.dp).width(112.dp), maxSide = 400) } }
                Column { Caption("CRNN input"); FileImage(c.repo.file(p.inputCropPath).path, Modifier.size(width = 112.dp, height = 56.dp), pixelated = true) }
            }
            Spacer(Modifier.height(10.dp))
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                PlateChip(p.predictedText, big = true)
                ConfBadge(p.predictedConf)
                if (p.correctedText != null) Icon(Icons.Filled.Check, "Corrected", tint = AxiosColors.Stable)
            }
            if (charConf.isNotEmpty() && charConf.size == p.predictedText.replace(" ", "").length) {
                Row(Modifier.padding(top = 4.dp), horizontalArrangement = Arrangement.spacedBy(2.dp)) {
                    p.predictedText.replace(" ", "").forEachIndexed { i, ch ->
                        Text("$ch", Modifier.background(confColor(charConf[i]).copy(alpha = 0.25f), RoundedCornerShape(3.dp)).padding(horizontal = 5.dp),
                            fontFamily = FontFamily.Monospace, fontSize = 13.sp, color = confColor(charConf[i]))
                    }
                }
            }
            Caption("${p.ocrModel?.let { "rec_$it" } ?: "no model"} · ${p.strategy} · det ${(p.detScore * 100).toInt()}%" +
                (if (p.rectified) " · rectified" else "") + (if (p.deshear != 0f) " · shear ${p.deshear}" else ""))
            region?.chosen?.let { Caption("Region: ${it.description} (${(it.conf * 100).toInt()}%, group ${it.group})") }
                ?: region?.let { Caption("Region: not routed") }
            hints.forEach { Caption("Hint · ${it.table}: ${it.key} = ${it.value}") }

            SectionTitle("Corrected reading (for training)")
            Row(verticalAlignment = Alignment.CenterVertically) {
                OutlinedTextField(
                    value = field,
                    onValueChange = { v -> field = v.uppercase().filter { it.isLetterOrDigit() || it == ' ' }.take(16) },
                    modifier = Modifier.weight(1f),
                    singleLine = true,
                    placeholder = { Text(p.predictedText, fontFamily = FontFamily.Monospace) },
                    textStyle = MaterialTheme.typography.titleMedium.copy(fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold),
                    keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.Characters, autoCorrectEnabled = false, imeAction = ImeAction.Done),
                    keyboardActions = KeyboardActions(onDone = { if (field.isNotBlank()) scope.launch { c.repo.setCorrection(p.id, field) } }),
                )
                Spacer(Modifier.width(8.dp))
                Button(onClick = { scope.launch { c.repo.setCorrection(p.id, field) } }, enabled = field.isNotBlank() && field != p.correctedText) { Text("Save") }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(top = 4.dp)) {
                OutlinedButton(onClick = { field = p.predictedText; scope.launch { c.repo.setCorrection(p.id, p.predictedText) } }) { Text("Prediction is correct") }
                if (p.correctedText != null) TextButton(onClick = { field = ""; scope.launch { c.repo.setCorrection(p.id, null) } }) { Text("Clear") }
            }
            p.correctedText?.let {
                Caption("Saved: $it" + (p.correctedAt?.let { t -> " · " + DateFormat.getDateTimeInstance().format(Date(t)) } ?: "") +
                    if (it != p.predictedText) " · differs from prediction" else " · confirms prediction")
            }

            if (pr.reads.isNotEmpty()) {
                SectionTitle("Every model's read · tap to use")
                pr.reads.sortedByDescending { it.conf }.forEach { r ->
                    Row(Modifier.fillMaxWidth().clickable { if (r.decodable) field = r.text.uppercase() }.padding(vertical = 3.dp), verticalAlignment = Alignment.CenterVertically) {
                        Text("rec_${r.model}", Modifier.width(64.dp), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Text(r.text, Modifier.weight(1f), fontFamily = FontFamily.Monospace, fontWeight = if (r.model == p.ocrModel) FontWeight.Bold else FontWeight.Normal,
                            color = if (r.decodable) MaterialTheme.colorScheme.onSurface else MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 14.sp)
                        Text("C=${r.classes}", Modifier.width(48.dp), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        ConfBadge(r.conf)
                    }
                }
            }
            region?.let { rd ->
                TextButton(onClick = { showRegions = !showRegions }) { Text(if (showRegions) "Hide region classifiers" else "Show all region classifiers (${rd.reads.size})") }
                if (showRegions) rd.reads.forEach { r ->
                    KeyValue("rec_${r.classifier} · ${r.group}", "${r.description}  ${(r.conf * 100).toInt()}%" + if (!r.canReject) "  (no reject class)" else "")
                }
            }
        }
    }
}

@Composable
private fun VehicleCard(c: AppContainer, v: VehicleEntity) {
    Card(Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
        Row(Modifier.padding(12.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            FileImage(v.cropPath?.let { c.repo.file(it).path }, Modifier.size(96.dp), maxSide = 300)
            Column {
                Text("Vehicle ${(v.score * 100).toInt()}%", style = MaterialTheme.typography.titleSmall)
                if (v.makeModel == null && v.colour == null && v.type == null) Caption("MMC was off for this capture")
                v.makeModel?.let { KeyValue("Make / model", "$it (${pct(v.makeModelConf)})") }
                v.pose?.let { KeyValue("Pose", "$it (${pct(v.poseConf)})") }
                v.colour?.let { KeyValue("Colour", "$it (${pct(v.colourConf)})") }
                v.type?.let { KeyValue("Type", "$it (${pct(v.typeConf)})") }
                v.refinedBox?.let { Caption("Refined box: $it") }
            }
        }
    }
}

private fun pct(v: Float?) = v?.let { "${(it * 100).toInt()}%" } ?: "–"

@Composable
private fun CaptureInfo(d: CaptureWithPlates) {
    val cap = d.capture
    var showCfg by remember { mutableStateOf(false) }
    Column(Modifier.padding(vertical = 8.dp)) {
        SectionTitle("Capture")
        KeyValue("Time", DateFormat.getDateTimeInstance().format(Date(cap.timestamp)))
        KeyValue("Source", cap.source + (cap.originalName?.let { " · $it" } ?: ""))
        KeyValue("Frame", "${cap.width}×${cap.height}")
        cap.sessionId?.let { KeyValue("Session", "#$it") }
        if (cap.lat != null) KeyValue("Location", "%.5f, %.5f".format(cap.lat, cap.lon))
        cap.videoPath?.let { KeyValue("Video", "$it @ ${(cap.videoOffsetMs ?: 0) / 1000.0}s") }
        KeyValue("Total", "%.1f ms".format(cap.totalMs))
        KeyValue("Timings", cap.timingsJson.trim('{', '}').replace("\"", "").replace(",", "  "))
        KeyValue("Models run", cap.modelsRun.split(",").joinToString(" ") { "rec_$it" })
        TextButton(onClick = { showCfg = !showCfg }) { Text(if (showCfg) "Hide pipeline config" else "Show pipeline config") }
        if (showCfg) Text(cap.configJson.replace(",\"", ",\n\""), fontFamily = FontFamily.Monospace, fontSize = 11.sp)
    }
}
