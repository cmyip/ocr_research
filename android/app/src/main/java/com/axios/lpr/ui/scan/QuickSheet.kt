package com.axios.lpr.ui.scan

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.axios.lpr.AppContainer
import com.axios.lpr.engine.Ids
import com.axios.lpr.engine.OcrMode
import com.axios.lpr.engine.PipelineConfig
import com.axios.lpr.ui.common.Caption
import com.axios.lpr.ui.common.SectionTitle
import kotlinx.coroutines.launch

/** Viewfinder quick toggles: MMC models, OCR mode, geometry and overlay options. */
@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun QuickSheet(c: AppContainer, simulating: Boolean, onSimulate: () -> Unit, onStopSimulation: () -> Unit, onDismiss: () -> Unit) {
    val s by c.settings.collectAsState()
    val scope = rememberCoroutineScope()
    val p = s.pipeline
    val l = s.live
    fun pipe(f: (PipelineConfig) -> PipelineConfig) = scope.launch { c.settingsRepo.updatePipeline(f) }
    fun live(f: (com.axios.lpr.engine.LiveConfig) -> com.axios.lpr.engine.LiveConfig) = scope.launch { c.settingsRepo.updateLive(f) }

    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(Modifier.fillMaxWidth().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp).padding(bottom = 32.dp)) {
            SectionTitle("MMC (make / model / colour)")
            Caption("Runs on each detected vehicle. Make/model (rec_15) is the heaviest model in the SDK.")
            ToggleRow("Make + model + pose (rec_15)", p.mmcMakeModel) { v -> pipe { it.copy(mmcMakeModel = v) } }
            ToggleRow("Colour (rec_19)", p.mmcColour) { v -> pipe { it.copy(mmcColour = v) } }
            ToggleRow("Vehicle type (rec_21)", p.mmcType) { v -> pipe { it.copy(mmcType = v) } }
            ToggleRow("Detect vehicles", p.detectVehicles) { v -> pipe { it.copy(detectVehicles = v) } }

            SectionTitle("OCR")
            FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                OcrMode.entries.forEach { m -> FilterChip(p.ocrMode == m, { pipe { it.copy(ocrMode = m) } }, { Text(m.label) }) }
            }
            if (p.ocrMode == OcrMode.PINNED) {
                Caption("Pinned model (0-9A-Z CRNNs)")
                FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    Ids.CRNNS.filter { c.modelStore.catalog[it].classes in setOf(37, 38) }.forEach { id ->
                        FilterChip(p.pinnedCrnn == id, { pipe { it.copy(pinnedCrnn = id) } }, { Text("rec_$id") })
                    }
                }
            }
            ToggleRow("Corner rectification (rec_71)", p.rectify) { v -> pipe { it.copy(rectify = v) } }
            ToggleRow("Region classifiers", p.regionEnabled) { v -> pipe { it.copy(regionEnabled = v) } }
            ToggleRow("640 detector (small/distant plates)", p.detector640) { v -> pipe { it.copy(detector640 = v) } }

            SectionTitle("Viewfinder")
            ToggleRow("Show vehicle boxes", l.showVehicles) { v -> live { it.copy(showVehicles = v) } }
            ToggleRow("Show corner quads", l.showCorners) { v -> live { it.copy(showCorners = v) } }
            ToggleRow("Scan only inside guide box", l.roiOnly) { v -> live { it.copy(roiOnly = v) } }
            ToggleRow("Performance HUD", l.showHud) { v -> live { it.copy(showHud = v) } }
            ToggleRow("Beep on new plate", l.beep) { v -> live { it.copy(beep = v) } }

            SectionTitle("Simulated camera")
            Caption("Replay photos through the live viewfinder path (tracker, labelled boxes, auto-record, snap). Useful without a car or on an emulator.")
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                androidx.compose.material3.OutlinedButton(onClick = onSimulate) { Text(if (simulating) "Pick other photos" else "Simulate with photos…") }
                if (simulating) androidx.compose.material3.TextButton(onClick = onStopSimulation) { Text("Back to camera") }
            }

            SectionTitle("Presets")
            FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                PipelineConfig.presets.forEach { (name, cfg) -> FilterChip(false, { pipe { cfg } }, { Text(name) }) }
            }
        }
    }
}

@Composable
fun ToggleRow(label: String, checked: Boolean, sub: String? = null, onChange: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth().padding(vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(label, style = MaterialTheme.typography.bodyMedium)
            sub?.let { Caption(it) }
        }
        Switch(checked, onChange)
    }
}
