package com.axios.lpr.ui.settings

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
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AssistChip
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.axios.lpr.AppContainer
import com.axios.lpr.engine.Ids
import com.axios.lpr.engine.LiveConfig
import com.axios.lpr.engine.MnnBackend
import com.axios.lpr.engine.MnnPrecision
import com.axios.lpr.engine.OcrMode
import com.axios.lpr.engine.OrtProvider
import com.axios.lpr.engine.PipelineConfig
import com.axios.lpr.engine.RegionGroups
import com.axios.lpr.ui.common.Caption
import com.axios.lpr.ui.common.SectionTitle
import com.axios.lpr.ui.scan.ToggleRow
import kotlinx.coroutines.launch
import kotlin.math.roundToInt

@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun SettingsScreen(c: AppContainer) {
    val s by c.settings.collectAsState()
    val scope = rememberCoroutineScope()
    val p = s.pipeline
    val l = s.live
    fun pipe(f: (PipelineConfig) -> PipelineConfig) { scope.launch { c.settingsRepo.updatePipeline(f) } }
    fun live(f: (LiveConfig) -> LiveConfig) { scope.launch { c.settingsRepo.updateLive(f) } }
    val catalog = c.modelStore.catalog

    Scaffold(topBar = { TopAppBar(title = { Text("Settings") }) }) { pad ->
        Column(Modifier.fillMaxSize().padding(pad).verticalScroll(rememberScrollState()).padding(horizontal = 16.dp)) {
            SectionTitle("Presets")
            FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                PipelineConfig.presets.forEach { (name, cfg) -> AssistChip({ pipe { cfg } }, { Text(name) }) }
            }
            Caption("Malaysia (verified): rectify + rec_57 read all three sample plates. PoC shear reproduces read_plate.py.")

            SectionTitle("Detector")
            FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                FilterChip(!p.detector640, { pipe { it.copy(detector640 = false) } }, { Text("rec_72 · 320") })
                FilterChip(p.detector640, { pipe { it.copy(detector640 = true) } }, { Text("rec_75 · 640") })
            }
            SliderRow("Plate score threshold", p.plateScore, 0.05f..0.9f) { v -> pipe { it.copy(plateScore = v) } }
            SliderRow("Vehicle score threshold", p.vehicleScore, 0.05f..0.9f) { v -> pipe { it.copy(vehicleScore = v) } }
            SliderRow("NMS IoU", p.nmsIou, 0.1f..0.9f) { v -> pipe { it.copy(nmsIou = v) } }
            SliderRow("Max plates per frame", p.maxPlates.toFloat(), 1f..12f, steps = 10, fmt = { it.roundToInt().toString() }) { v -> pipe { it.copy(maxPlates = v.roundToInt()) } }
            ToggleRow("Detect vehicles", p.detectVehicles) { v -> pipe { it.copy(detectVehicles = v) } }

            SectionTitle("OCR strategy")
            OcrMode.entries.forEach { m ->
                Row(verticalAlignment = Alignment.CenterVertically) {
                    FilterChip(p.ocrMode == m, { pipe { it.copy(ocrMode = m) } }, { Text(m.label) })
                    Spacer(Modifier.width(8.dp)); Caption(m.help)
                }
            }
            Caption("Pinned model")
            FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                Ids.CRNNS.forEach { id -> FilterChip(p.pinnedCrnn == id, { pipe { it.copy(pinnedCrnn = id) } }, { Text("$id") }) }
            }
            ToggleRow("Compare every enabled CRNN on snap/import", p.compareAllOnCapture) { v -> pipe { it.copy(compareAllOnCapture = v) } }

            SectionTitle("CRNN toggles (ensembles use the enabled 0-9A-Z models)")
            FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                Ids.CRNNS.forEach { id ->
                    val cls = catalog[id].classes
                    FilterChip(id in p.enabledCrnns, {
                        pipe { cfg -> com.axios.lpr.engine.ModelToggles.set(id, id !in cfg.enabledCrnns, cfg) }
                    }, { Text("$id·C$cls") })
                }
            }
            Row { AssistChip({ pipe { it.copy(enabledCrnns = Ids.CRNNS.toSet()) } }, { Text("All on") }); Spacer(Modifier.width(6.dp))
                AssistChip({ pipe { it.copy(enabledCrnns = Ids.CRNNS.filter { id -> catalog[id].classes == 37 }.toSet()) } }, { Text("Only 0-9A-Z") }) }

            SectionTitle("Region classifiers")
            ToggleRow("Run region classifiers", p.regionEnabled, "Routing mode always runs them") { v -> pipe { it.copy(regionEnabled = v) } }
            FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                RegionGroups.all.forEach { g ->
                    FilterChip(g.classifier in p.enabledClassifiers, {
                        pipe { cfg -> cfg.copy(enabledClassifiers = if (g.classifier in cfg.enabledClassifiers) cfg.enabledClassifiers - g.classifier else cfg.enabledClassifiers + g.classifier) }
                    }, { Text("${g.classifier}·${g.name}" + (g.crnn?.let { "→$it" } ?: "")) })
                }
            }

            SectionTitle("Plate geometry")
            ToggleRow("Corner rectification (rec_71)", p.rectify, "Perspective-warp the plate from its 4 predicted corners") { v -> pipe { it.copy(rectify = v) } }
            SliderRow("Corner context margin X", p.cornerMarginX, 0f..0.5f) { v -> pipe { it.copy(cornerMarginX = v) } }
            SliderRow("Corner context margin Y", p.cornerMarginY, 0f..1f) { v -> pipe { it.copy(cornerMarginY = v) } }
            SliderRow("Crop pad", p.cropPad, 0f..0.15f, fmt = { "%.3f".format(it) }) { v -> pipe { it.copy(cropPad = v) } }
            SliderRow("De-shear (italic fonts)", p.deshear, -0.6f..0.6f) { v -> pipe { it.copy(deshear = v) } }

            SectionTitle("MMC · vehicle models")
            ToggleRow("Make / model + pose (rec_15)", p.mmcMakeModel, "4193 make/model classes, heaviest model") { v -> pipe { it.copy(mmcMakeModel = v) } }
            ToggleRow("Colour (rec_19)", p.mmcColour) { v -> pipe { it.copy(mmcColour = v) } }
            ToggleRow("Vehicle type (rec_21)", p.mmcType) { v -> pipe { it.copy(mmcType = v) } }
            ToggleRow("Vehicle box refiner (rec_78, experimental)", p.vehicleBoxRefine) { v -> pipe { it.copy(vehicleBoxRefine = v) } }
            ToggleRow("Lookup-table hints", p.lookups, "DE district, ES province, JO category, KZ region") { v -> pipe { it.copy(lookups = v) } }

            SectionTitle("Runtimes")
            Caption("MNN backend (GPU sessions are checked against CPU and fall back if they disagree)")
            FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                MnnBackend.entries.forEach { b -> FilterChip(p.mnnBackend == b, { pipe { it.copy(mnnBackend = b) } }, { Text(b.label) }) }
            }
            FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                MnnPrecision.entries.forEach { pr -> FilterChip(p.mnnPrecision == pr, { pipe { it.copy(mnnPrecision = pr) } }, { Text("Precision ${pr.name.lowercase()}") }) }
            }
            SliderRow("MNN threads", p.mnnThreads.toFloat(), 1f..8f, steps = 6, fmt = { it.roundToInt().toString() }) { v -> pipe { it.copy(mnnThreads = v.roundToInt()) } }
            Caption("ONNX Runtime execution provider (detector)")
            FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                OrtProvider.entries.forEach { o -> FilterChip(p.ortProvider == o, { pipe { it.copy(ortProvider = o) } }, { Text(o.label) }) }
            }
            SliderRow("ORT threads", p.ortThreads.toFloat(), 1f..8f, steps = 6, fmt = { it.roundToInt().toString() }) { v -> pipe { it.copy(ortThreads = v.roundToInt()) } }

            SectionTitle("Live scan & recording")
            SliderRow("Skip frames between analyses", l.frameSkip.toFloat(), 0f..5f, steps = 4, fmt = { it.roundToInt().toString() }) { v -> live { it.copy(frameSkip = v.roundToInt()) } }
            SliderRow("Frames before a plate is stable", l.stableFrames.toFloat(), 1f..10f, steps = 8, fmt = { it.roundToInt().toString() }) { v -> live { it.copy(stableFrames = v.roundToInt()) } }
            SliderRow("Min confidence to auto-record", l.minRecordConf, 0.5f..0.99f) { v -> live { it.copy(minRecordConf = v) } }
            SliderRow("Ignore repeats within (s)", l.dedupeSeconds.toFloat(), 0f..300f, fmt = { it.roundToInt().toString() }) { v -> live { it.copy(dedupeSeconds = v.roundToInt()) } }
            Caption("Analysis resolution")
            FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                listOf(640, 960, 1280, 1920).forEach { w -> FilterChip(l.analysisWidth == w, { live { it.copy(analysisWidth = w) } }, { Text("${w}p-wide") }) }
            }
            ToggleRow("Auto-record stable plates", l.autoRecord) { v -> live { it.copy(autoRecord = v) } }
            ToggleRow("Beep on new plate", l.beep) { v -> live { it.copy(beep = v) } }
            ToggleRow("Vibrate on new plate", l.haptic) { v -> live { it.copy(haptic = v) } }
            ToggleRow("Show vehicle boxes", l.showVehicles) { v -> live { it.copy(showVehicles = v) } }
            ToggleRow("Show corner quads", l.showCorners) { v -> live { it.copy(showCorners = v) } }
            ToggleRow("Performance HUD", l.showHud) { v -> live { it.copy(showHud = v) } }
            ToggleRow("Scan only inside guide box", l.roiOnly) { v -> live { it.copy(roiOnly = v) } }
            ToggleRow("Tag captures with location", l.tagLocation, "Uses the last known fix; stays on device") { v -> live { it.copy(tagLocation = v) } }
            ToggleRow("Use front camera", l.useFrontCamera) { v -> live { it.copy(useFrontCamera = v) } }

            SectionTitle("About these models")
            Caption(
                "Weights are the unpacked assets of a commercial ANPR SDK, used here for local research. " +
                    "Only the 37- and 38-class CRNN alphabets decode to text; other alphabets show class indices. " +
                    "The SDK's region-ID → name tables are missing, so names shown are inferred. " +
                    "rec_04–14 are custom float blobs with an unknown loader and are not runnable. " +
                    "All captures stay on this device.",
            )
            Spacer(Modifier.height(40.dp))
        }
    }
}

@Composable
private fun SliderRow(label: String, value: Float, range: ClosedFloatingPointRange<Float>, steps: Int = 0, fmt: (Float) -> String = { "%.2f".format(it) }, onDone: (Float) -> Unit) {
    var v by remember(value) { mutableFloatStateOf(value) }
    Column(Modifier.fillMaxWidth().padding(vertical = 2.dp)) {
        Row { Text(label, style = MaterialTheme.typography.bodyMedium, modifier = Modifier.weight(1f)); Text(fmt(v), style = MaterialTheme.typography.labelLarge) }
        Slider(v, { v = it }, valueRange = range, steps = steps, onValueChangeFinished = { onDone(v) })
    }
}
