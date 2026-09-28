package com.axios.lpr.ui.lab

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Error
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.axios.lpr.AppContainer
import com.axios.lpr.engine.ModelRecord
import com.axios.lpr.engine.ModelToggles
import com.axios.lpr.engine.RgbImage
import com.axios.lpr.engine.SelfTest
import com.axios.lpr.engine.SelfTestItem
import com.axios.lpr.engine.toBitmap
import com.axios.lpr.ui.ImportMenu
import com.axios.lpr.ui.common.BitmapImage
import com.axios.lpr.ui.common.Caption
import com.axios.lpr.ui.common.SectionTitle
import com.axios.lpr.ui.theme.AxiosColors
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private val roleOrder = listOf("detector", "corners", "ocr", "region", "mmr", "colour", "vtype", "vbox", "sample", "labels", "idlist", "lookup", "config", "unknown", "dummy")
private val roleNames = mapOf(
    "detector" to "Detectors (ONNX)", "corners" to "Corner keypoints", "ocr" to "CRNN OCR", "region" to "Region classifiers",
    "mmr" to "MMC: make/model", "colour" to "MMC: colour", "vtype" to "MMC: vehicle type", "vbox" to "Vehicle box refiner",
    "sample" to "Bundled sample inputs", "labels" to "Label files", "idlist" to "Region ID lists", "lookup" to "Lookup tables",
    "config" to "Config", "unknown" to "Not runnable (custom float blobs)", "dummy" to "Dummy records",
)

@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun LabScreen(c: AppContainer, onTables: () -> Unit) {
    val scope = rememberCoroutineScope()
    val settings by c.settings.collectAsState()
    val cfg = settings.pipeline
    val results = remember { mutableStateListOf<SelfTestItem>() }
    var testing by remember { mutableStateOf(false) }
    val bench = remember { mutableStateMapOf<Int, String>() }
    var loaded by remember { mutableStateOf(c.models.loaded()) }
    var roleFilter by remember { mutableStateOf("runnable") }
    LaunchedEffect(Unit) { while (true) { loaded = c.models.loaded(); delay(1000) } }

    val records = c.modelStore.catalog.records.filter {
        when (roleFilter) { "runnable" -> it.runnable; "all" -> true; else -> !it.runnable }
    }.sortedWith(compareBy({ roleOrder.indexOf(it.role).let { i -> if (i < 0) 99 else i } }, { it.id }))

    Scaffold(topBar = { TopAppBar(title = { Column { Text("Model Lab"); Caption("${loaded.size} loaded · ${c.modelStore.catalog.records.size} SDK records") } }) }) { pad ->
        LazyColumn(Modifier.fillMaxSize().padding(pad).padding(horizontal = 12.dp)) {
            item {
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(12.dp)) {
                        Text("Self-test on the SDK's bundled samples", style = MaterialTheme.typography.titleSmall)
                        Caption("Region classifiers, CRNNs, both detectors and MMC on inputs shipped inside the weights. Uses the current runtime settings.")
                        Row(Modifier.padding(top = 8.dp), horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                            Button(enabled = !testing, onClick = {
                                results.clear(); testing = true
                                scope.launch(Dispatchers.Default) {
                                    runCatching { SelfTest(c.pipeline).run(cfg) { item -> scope.launch { results += item } } }
                                        .onFailure { e -> scope.launch { results += SelfTestItem("error", "", e.message ?: "$e", false, 0f) } }
                                    testing = false
                                }
                            }) { Text("Run self-test") }
                            if (testing) CircularProgressIndicator(Modifier.size(20.dp), strokeWidth = 2.dp)
                            if (results.isNotEmpty()) Text("${results.count { it.pass }}/${results.size} pass", color = if (results.all { it.pass }) AxiosColors.Stable else AxiosColors.Warn)
                        }
                        results.forEach { r ->
                            Row(Modifier.padding(vertical = 2.dp), verticalAlignment = Alignment.CenterVertically) {
                                Icon(if (r.pass) Icons.Filled.CheckCircle else Icons.Filled.Error, null, Modifier.size(16.dp), tint = if (r.pass) AxiosColors.Stable else AxiosColors.Warn)
                                Spacer(Modifier.width(6.dp))
                                Column(Modifier.weight(1f)) {
                                    Text(r.name, style = MaterialTheme.typography.bodySmall)
                                    Caption("expected ${r.expected} · got ${r.got}")
                                }
                                if (r.ms > 0) Caption("%.0f ms".format(r.ms))
                            }
                        }
                    }
                }
            }
            item {
                Row(Modifier.padding(vertical = 8.dp), horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                    ImportMenu(c)
                    OutlinedButton(onClick = onTables) { Text("Reference tables") }
                    TextButton(onClick = { scope.launch(Dispatchers.Default) { c.pipeline.unloadAll(); loaded = c.models.loaded() } }) { Text("Unload all") }
                }
                Caption("Analyse an image: imports it and opens the capture with every intermediate crop and every model's read.")
            }
            item {
                FlowRow(Modifier.padding(top = 8.dp), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    listOf("runnable" to "Runnable models", "data" to "Data records", "all" to "All 87").forEach { (k, l) ->
                        FilterChip(roleFilter == k, { roleFilter = k }, { Text(l) })
                    }
                }
            }
            var lastRole: String? = null
            records.forEach { rec ->
                if (rec.role != lastRole) {
                    val role = rec.role
                    item(key = "h_$role") { SectionTitle(roleNames[role] ?: role) }
                    lastRole = role
                }
                item(key = rec.id) {
                    ModelCard(c, rec, loaded[rec.id], ModelToggles.isOn(rec.id, cfg), bench[rec.id],
                        onToggle = { on -> scope.launch { c.settingsRepo.updatePipeline { ModelToggles.set(rec.id, on, it) } } },
                        onBench = {
                            bench[rec.id] = "running…"
                            scope.launch {
                                val r = withContext(Dispatchers.Default) { runCatching { c.pipeline.benchmark(rec.id, 30, cfg) } }
                                bench[rec.id] = r.fold({ v ->
                                    val s = v.sorted()
                                    "median %.2f ms · min %.2f · max %.2f (30 runs)".format(s[s.size / 2], s.first(), s.last())
                                }, { "failed: ${it.message}" })
                                loaded = c.models.loaded()
                            }
                        })
                }
            }
            item { Spacer(Modifier.height(32.dp)) }
        }
    }
}

@Composable
private fun ModelCard(c: AppContainer, rec: ModelRecord, loadedAs: String?, on: Boolean?, bench: String?, onToggle: (Boolean) -> Unit, onBench: () -> Unit) {
    Card(Modifier.fillMaxWidth().padding(vertical = 3.dp)) {
        Column(Modifier.padding(10.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("rec_%02d · %s".format(rec.id, rec.title), style = MaterialTheme.typography.bodyMedium)
                    Caption("${rec.kind} · ${"%.1f".format(rec.size / 1e6)} MB" + (rec.input?.let { " · in ${it.joinToString("×")}" } ?: "") +
                        (rec.classes?.let { " · C=$it" } ?: "") + (rec.group?.let { " · group $it" } ?: ""))
                }
                if (on != null) Switch(on, onToggle)
            }
            rec.preprocess?.let { Caption("Input: $it") }
            rec.notes?.let { Caption(it) }
            rec.truth?.let { Caption("Truth: $it") }
            if (rec.runnable) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        loadedAs?.let { "loaded · $it" + (c.models.loadTimeMs(rec.id)?.let { t -> " · load %.0f ms".format(t) } ?: "") } ?: "not loaded",
                        style = MaterialTheme.typography.labelSmall, color = if (loadedAs != null) AxiosColors.Scan else MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.weight(1f),
                    )
                    TextButton(onClick = onBench) { Text("Benchmark") }
                }
                c.models.note(rec.id)?.let { Caption(it) }
                bench?.let { Text(it, fontFamily = FontFamily.Monospace, fontSize = 11.sp) }
            }
            if (rec.role == "sample" && rec.shape != null) SamplePreview(c, rec)
        }
    }
}

@Composable
private fun SamplePreview(c: AppContainer, rec: ModelRecord) {
    val bmp = remember(rec.id) {
        runCatching {
            val (h, w) = rec.shape!![0] to rec.shape[1]
            // Plate crops are stored BGR; the 224 vehicle and detector samples are RGB.
            RgbImage.fromRaw(c.modelStore.bytes(rec.id), w, h, bgr = h == 48).toBitmap()
        }.getOrNull()
    }
    bmp?.let { BitmapImage(it, Modifier.padding(top = 6.dp).height(if (rec.shape!![0] == 48) 48.dp else 120.dp), pixelated = rec.shape[0] == 48) }
}
