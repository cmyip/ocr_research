package com.axios.lpr.ui.scan

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Cameraswitch
import androidx.compose.material.icons.filled.FiberManualRecord
import androidx.compose.material.icons.filled.FlashOff
import androidx.compose.material.icons.filled.FlashOn
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.outlined.PhotoLibrary
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilledIconToggleButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.axios.lpr.AppContainer
import com.axios.lpr.engine.Box as PBox
import com.axios.lpr.ui.ImportMenu
import com.axios.lpr.ui.common.PlateChip
import com.axios.lpr.ui.theme.AxiosColors
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

@Composable
fun ScanScreen(c: AppContainer, onOpenCapture: (Long) -> Unit, onError: (String) -> Unit) {
    val context = LocalContext.current
    var granted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    val askCamera = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }
    LaunchedEffect(Unit) { if (!granted) askCamera.launch(Manifest.permission.CAMERA) }

    if (!granted) {
        Column(Modifier.fillMaxSize().padding(24.dp), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) {
            Text("Camera permission is needed to scan plates.", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(12.dp))
            Text("You can still import photos from your library or files.", color = MaterialTheme.colorScheme.onSurfaceVariant)
            Spacer(Modifier.height(16.dp))
            Button(onClick = { askCamera.launch(Manifest.permission.CAMERA) }) { Text("Grant camera access") }
            Spacer(Modifier.height(8.dp))
            ImportMenu(c)
        }
        return
    }
    CameraContent(c, onOpenCapture, onError)
}

@Composable
private fun CameraContent(c: AppContainer, onOpenCapture: (Long) -> Unit, onError: (String) -> Unit) {
    val context = LocalContext.current
    val owner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()
    val controller = remember { CameraController(context.applicationContext, c) }
    val previewView = remember { PreviewView(context).apply { scaleType = PreviewView.ScaleType.FILL_CENTER; implementationMode = PreviewView.ImplementationMode.COMPATIBLE } }
    val settings by c.settings.collectAsState()
    val live by controller.live.collectAsState()
    val recorded by controller.recorded.collectAsState()
    val recording by controller.recordingVideo.collectAsState()
    val torch by controller.torch.collectAsState()
    val hasFlash by controller.hasFlash.collectAsState()
    val prepare by c.prepare.collectAsState()
    val simFrame by controller.simFrame.collectAsState()
    val simPicker = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia(20)) { uris ->
        if (uris.isNotEmpty()) scope.launch(kotlinx.coroutines.Dispatchers.IO) {
            val bmps = uris.mapNotNull { u -> runCatching { com.axios.lpr.engine.ImageLoader.decode(context.contentResolver, u, maxSide = 1280) }.getOrNull() }
            controller.startSimulation(bmps)
        }
    }
    var mode by remember { mutableStateOf(CaptureMode.PHOTO) }
    var viewW by remember { mutableStateOf(0f) }
    var viewH by remember { mutableStateOf(0f) }
    var snapping by remember { mutableStateOf(false) }
    var showSheet by remember { mutableStateOf(false) }
    val askAudio = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        scope.launch { runCatching { controller.startVideo(withAudio = ok) }.onFailure { onError(it.message ?: "Recording failed") } }
    }
    val askLocation = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { }
    LaunchedEffect(settings.live.tagLocation) {
        if (settings.live.tagLocation && ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            askLocation.launch(Manifest.permission.ACCESS_FINE_LOCATION)
        }
    }

    DisposableEffect(Unit) { onDispose { controller.release() } }
    LaunchedEffect(mode, settings.live.useFrontCamera, settings.live.analysisWidth) {
        // The ViewPort needs a laid-out PreviewView.
        kotlinx.coroutines.flow.flow { while (true) { emit(previewView.width); kotlinx.coroutines.delay(30) } }.first { it > 0 }
        runCatching { controller.bind(owner, previewView, mode) }.onFailure { onError("Camera: ${it.message}") }
    }

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        AndroidView({ previewView }, Modifier.fillMaxSize())
        simFrame?.let { com.axios.lpr.ui.common.BitmapImage(it, Modifier.fillMaxSize(), contentScale = androidx.compose.ui.layout.ContentScale.Crop) }
        Box(
            Modifier.fillMaxSize()
                .onSizeChanged { viewW = it.width.toFloat(); viewH = it.height.toFloat(); controller.viewSize = viewW to viewH }
                .pointerInput(Unit) { detectTransformGestures { _, _, zoom, _ -> if (zoom != 1f) controller.zoomBy(zoom) } }
                .pointerInput(Unit) { detectTapGestures(onTap = { o: Offset -> controller.focus(previewView, o.x, o.y) }) },
        ) {
            val roi = if (settings.live.roiOnly && viewW > 0) PBox(viewW * 0.08f, viewH * 0.32f, viewW * 0.92f, viewH * 0.62f) else null
            controller.roiView = roi
            PlateOverlay(live, settings.live, viewW, viewH, roi, Modifier.fillMaxSize())
        }

        // Top bar: HUD + toggles
        Column(Modifier.fillMaxWidth().safeDrawingPadding().padding(8.dp)) {
            Row(verticalAlignment = Alignment.Top) {
                if (settings.live.showHud) Hud(live, Modifier.weight(1f)) else Spacer(Modifier.weight(1f))
                Column(horizontalAlignment = Alignment.End) {
                    Row {
                        if (hasFlash) RoundIcon(if (torch) Icons.Filled.FlashOn else Icons.Filled.FlashOff, "Torch", torch) { controller.setTorch(!torch) }
                        RoundIcon(Icons.Filled.Cameraswitch, "Switch camera", false) {
                            scope.launch { c.settingsRepo.updateLive { it.copy(useFrontCamera = !it.useFrontCamera) } }
                        }
                        RoundIcon(Icons.Filled.Tune, "Quick settings", showSheet) { showSheet = true }
                    }
                    AutoRecordToggle(settings.live.autoRecord || recording) {
                        scope.launch {
                            c.settingsRepo.updateLive { it.copy(autoRecord = !it.autoRecord) }
                            if (settings.live.autoRecord) controller.endAutoSession()
                        }
                    }
                }
            }
            prepare?.let { (d, t) ->
                Text("Preparing models $d/$t…", Modifier.padding(top = 6.dp).background(Color(0xCC000000), RoundedCornerShape(6.dp)).padding(8.dp),
                    color = AxiosColors.Plate, style = MaterialTheme.typography.labelMedium)
            }
            live.error?.let { Text("⚠ $it", color = AxiosColors.Warn, style = MaterialTheme.typography.labelSmall) }
            if (simFrame != null) Row(Modifier.padding(top = 6.dp).background(Color(0xCC7C3AED), RoundedCornerShape(6.dp))
                .clickable { controller.stopSimulation() }.padding(horizontal = 10.dp, vertical = 6.dp)) {
                Text("SIMULATED CAMERA · tap to stop", color = Color.White, style = MaterialTheme.typography.labelMedium)
            }
        }

        // Bottom controls
        Column(Modifier.align(Alignment.BottomCenter).fillMaxWidth().background(Color(0x99000000)).safeDrawingPadding().padding(bottom = 8.dp)) {
            if (recorded.isNotEmpty()) {
                Row(Modifier.horizontalScroll(rememberScrollState()).padding(horizontal = 8.dp, vertical = 6.dp), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    recorded.forEach { r -> PlateChip(r.text, Modifier.clickable { onOpenCapture(r.captureId) }) }
                }
            }
            Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.weight(1f)) { ImportMenu(c, compact = true, icon = Icons.Outlined.PhotoLibrary) }
                Shutter(mode, recording, snapping) {
                    when {
                        mode == CaptureMode.PHOTO && !snapping -> scope.launch {
                            snapping = true
                            runCatching { controller.snap() }.onSuccess(onOpenCapture).onFailure { onError("Snap failed: ${it.message}") }
                            snapping = false
                        }
                        mode == CaptureMode.VIDEO && recording -> controller.stopVideo()
                        mode == CaptureMode.VIDEO -> askAudio.launch(Manifest.permission.RECORD_AUDIO)
                    }
                }
                Box(Modifier.weight(1f), contentAlignment = Alignment.CenterEnd) {
                    SingleChoiceSegmentedButtonRow(Modifier.width(150.dp)) {
                        CaptureMode.entries.forEachIndexed { i, m ->
                            SegmentedButton(selected = mode == m, onClick = { if (!recording) mode = m },
                                shape = SegmentedButtonDefaults.itemShape(i, 2)) { Text(if (m == CaptureMode.PHOTO) "Photo" else "Video", fontSize = 12.sp) }
                        }
                    }
                }
            }
        }
    }
    if (showSheet) QuickSheet(
        c, simulating = simFrame != null,
        onSimulate = { showSheet = false; simPicker.launch(androidx.activity.result.PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
        onStopSimulation = { showSheet = false; controller.stopSimulation() },
        onDismiss = { showSheet = false },
    )
}

@Composable
private fun RoundIcon(icon: androidx.compose.ui.graphics.vector.ImageVector, desc: String, on: Boolean, onClick: () -> Unit) {
    FilledIconToggleButton(checked = on, onCheckedChange = { onClick() },
        colors = IconButtonDefaults.filledIconToggleButtonColors(containerColor = Color(0x88000000), contentColor = Color.White,
            checkedContainerColor = AxiosColors.Plate, checkedContentColor = AxiosColors.Ink)) { Icon(icon, desc) }
}

@Composable
private fun AutoRecordToggle(on: Boolean, onClick: () -> Unit) {
    Row(Modifier.padding(top = 4.dp).background(if (on) Color(0xCCDC2626) else Color(0x88000000), RoundedCornerShape(16.dp))
        .clickable(onClick = onClick).padding(horizontal = 10.dp, vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
        Icon(Icons.Filled.FiberManualRecord, null, Modifier.size(12.dp), tint = if (on) Color.White else Color(0xFFDC2626))
        Spacer(Modifier.width(6.dp))
        Text(if (on) "Auto-record ON" else "Auto-record", color = Color.White, style = MaterialTheme.typography.labelMedium)
    }
}

@Composable
private fun Shutter(mode: CaptureMode, recording: Boolean, busy: Boolean, onClick: () -> Unit) {
    val ring = if (mode == CaptureMode.VIDEO) Color(0xFFDC2626) else Color.White
    Box(Modifier.size(76.dp).border(4.dp, ring, CircleShape).padding(8.dp).background(if (mode == CaptureMode.VIDEO) Color(0xFFDC2626) else Color.White, CircleShape)
        .clickable(onClick = onClick), contentAlignment = Alignment.Center) {
        when {
            busy -> CircularProgressIndicator(Modifier.size(28.dp), color = AxiosColors.Ink, strokeWidth = 3.dp)
            recording -> Icon(Icons.Filled.Stop, "Stop", tint = Color.White, modifier = Modifier.size(32.dp))
        }
    }
}

@Composable
private fun Hud(live: LiveState, modifier: Modifier) {
    val t = live.timings
    fun ms(k: String) = t[k]?.let { "%.0f".format(it) } ?: "–"
    Column(modifier.background(Color(0x99000000), RoundedCornerShape(8.dp)).padding(8.dp)) {
        Text("%.1f fps · %s ms".format(live.fps, ms("total")), color = AxiosColors.Scan, fontFamily = FontFamily.Monospace, fontSize = 13.sp)
        Text("det ${ms("det_prep")}+${ms("det_infer")} rect ${ms("rectify")} reg ${ms("region")} ocr ${ms("ocr")} mmc ${ms("mmc")}",
            color = Color.White, fontFamily = FontFamily.Monospace, fontSize = 10.sp)
        Text("${live.frameW}×${live.frameH} · models ${live.modelsRun.sorted().joinToString(",") { it.toString() }}",
            color = Color(0xFFB0BEC5), fontFamily = FontFamily.Monospace, fontSize = 10.sp)
    }
}
