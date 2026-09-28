package com.axios.lpr.ui.scan

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Log
import android.util.Size
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.lifecycle.awaitInstance
import androidx.camera.video.FallbackStrategy
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.axios.lpr.AppContainer
import com.axios.lpr.data.CaptureMeta
import com.axios.lpr.engine.Box
import com.axios.lpr.engine.FrameResult
import com.axios.lpr.engine.PlateTracker
import com.axios.lpr.engine.RgbImage
import com.axios.lpr.engine.toRgbImage
import com.axios.lpr.ui.LocationHelper
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

data class LiveTrack(
    val id: Int, val box: Box, val text: String, val conf: Float, val stable: Boolean, val recorded: Boolean,
    val corners: FloatArray?, val region: String?, val model: Int?,
)

data class LiveVehicle(val box: Box, val label: String?)

data class LiveState(
    val frameW: Int = 0, val frameH: Int = 0,
    val tracks: List<LiveTrack> = emptyList(),
    val vehicles: List<LiveVehicle> = emptyList(),
    val timings: Map<String, Float> = emptyMap(),
    val fps: Float = 0f,
    val modelsRun: Set<Int> = emptySet(),
    val error: String? = null,
    val frames: Long = 0,
    val simulated: Boolean = false,
)

enum class CaptureMode { PHOTO, VIDEO }

data class RecordedPlate(val text: String, val conf: Float, val captureId: Long, val at: Long)

/**
 * CameraX wiring. Preview, ImageAnalysis and ImageCapture (photo mode) or VideoCapture
 * (video mode) share one ViewPort, so analysis frames cover exactly what the viewfinder shows.
 */
class CameraController(private val context: Context, private val c: AppContainer) {
    private val analysisExecutor = Executors.newSingleThreadExecutor { Thread(it, "axios-analysis") }
    private val tracker = PlateTracker()
    private val busy = AtomicBoolean(false)
    private var camera: Camera? = null
    private var imageCapture: ImageCapture? = null
    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null
    private var recordingSession: Long? = null
    private var recordingFile: File? = null
    private var recordingStart = 0L
    private var autoSession: Long? = null
    private var frameIndex = 0L
    private var lastFrameNs = 0L
    private var fpsEma = 0f
    private val recentTexts = HashMap<String, Long>()
    private val tone by lazy { runCatching { ToneGenerator(AudioManager.STREAM_NOTIFICATION, 70) }.getOrNull() }

    @Volatile var roiView: Box? = null
    @Volatile var viewSize: Pair<Float, Float> = 0f to 0f
    @Volatile var paused = false

    private val _live = MutableStateFlow(LiveState())
    val live: StateFlow<LiveState> = _live
    private val _recorded = MutableStateFlow<List<RecordedPlate>>(emptyList())
    val recorded: StateFlow<List<RecordedPlate>> = _recorded
    private val _recordingVideo = MutableStateFlow(false)
    val recordingVideo: StateFlow<Boolean> = _recordingVideo
    val zoom = MutableStateFlow(1f)
    val torch = MutableStateFlow(false)
    val hasFlash = MutableStateFlow(false)

    suspend fun bind(owner: LifecycleOwner, previewView: PreviewView, mode: CaptureMode) {
        val live = c.settings.value.live
        val provider = ProcessCameraProvider.awaitInstance(context)
        provider.unbindAll()
        tracker.reset()
        val aspect = ResolutionSelector.Builder().setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY).build()
        val analysisSel = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
            .setResolutionStrategy(ResolutionStrategy(Size(live.analysisWidth, live.analysisWidth * 9 / 16), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
            .build()
        val preview = Preview.Builder().setResolutionSelector(aspect).build().also { it.surfaceProvider = previewView.surfaceProvider }
        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(analysisSel)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
        analysis.setAnalyzer(analysisExecutor, ::analyze)
        val group = UseCaseGroup.Builder().addUseCase(preview).addUseCase(analysis)
        previewView.viewPort?.let { group.setViewPort(it) }
        imageCapture = null; videoCapture = null
        if (mode == CaptureMode.PHOTO) {
            imageCapture = ImageCapture.Builder().setResolutionSelector(aspect).setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY).build()
                .also { group.addUseCase(it) }
        } else {
            val recorder = Recorder.Builder().setQualitySelector(QualitySelector.from(Quality.HD, FallbackStrategy.lowerQualityOrHigherThan(Quality.SD))).build()
            videoCapture = VideoCapture.withOutput(recorder).also { group.addUseCase(it) }
        }
        val selector = if (live.useFrontCamera) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
        camera = provider.bindToLifecycle(owner, selector, group.build()).also { cam ->
            hasFlash.value = cam.cameraInfo.hasFlashUnit()
            cam.cameraControl.enableTorch(torch.value && cam.cameraInfo.hasFlashUnit())
            cam.cameraControl.setZoomRatio(zoom.value)
        }
    }

    fun unbind() {
        stopVideo()
        runCatching { ProcessCameraProvider.getInstance(context).get().unbindAll() }
    }

    fun setTorch(on: Boolean) { torch.value = on; camera?.cameraControl?.enableTorch(on) }

    fun zoomBy(factor: Float) {
        val cam = camera ?: return
        val zs = cam.cameraInfo.zoomState.value ?: return
        val r = (zoom.value * factor).coerceIn(zs.minZoomRatio, zs.maxZoomRatio)
        zoom.value = r
        cam.cameraControl.setZoomRatio(r)
    }

    fun focus(previewView: PreviewView, x: Float, y: Float) {
        val p = previewView.meteringPointFactory.createPoint(x, y)
        camera?.cameraControl?.startFocusAndMetering(FocusMeteringAction.Builder(p).build())
    }

    /** Upright RGB frame cropped to the ViewPort (cropRect is in buffer coordinates). */
    private fun ImageProxy.toUpright(): RgbImage {
        val bmp = toBitmap()
        val crop = cropRect
        val m = Matrix().apply { postRotate(imageInfo.rotationDegrees.toFloat()) }
        val needsCrop = !(crop.width() == bmp.width && crop.height() == bmp.height) &&
            crop.right <= bmp.width && crop.bottom <= bmp.height && crop.width() > 0 && crop.height() > 0
        val out = if (needsCrop) Bitmap.createBitmap(bmp, crop.left, crop.top, crop.width(), crop.height(), m, false)
        else if (imageInfo.rotationDegrees != 0) Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, false)
        else bmp
        return out.toRgbImage()
    }

    private fun analyze(proxy: ImageProxy) {
        try {
            val live = c.settings.value.live
            frameIndex++
            if (simulating || paused || c.prepare.value != null || (live.frameSkip > 0 && frameIndex % (live.frameSkip + 1) != 0L)) return
            analyzeFrame(proxy.toUpright())
        } catch (e: Throwable) {
            Log.e("AxiosCamera", "analysis failed", e)
            _live.value = _live.value.copy(error = e.message ?: e.javaClass.simpleName)
        } finally {
            proxy.close()
        }
    }

    /** One live frame: pipeline → tracker → announce/auto-record → overlay state. Camera and simulator share it. */
    private fun analyzeFrame(img: RgbImage) {
        val live = c.settings.value.live
        val cfg = c.settings.value.pipeline
        val (vw, vh) = viewSize
        val roi = if (live.roiOnly && vw > 0) {
            roiView?.let { com.axios.lpr.engine.CoordinateMapper(img.width, img.height, vw, vh, live.useFrontCamera && !simulating).unmap(it) }
        } else null
        val frame = c.pipeline.process(img, cfg, capture = false, roi = roi)
        lastFrame = img
        val active = tracker.update(frame.plates)
        val now = System.nanoTime()
        if (lastFrameNs != 0L) {
            val fps = 1e9f / (now - lastFrameNs)
            fpsEma = if (fpsEma == 0f) fps else fpsEma * 0.85f + fps * 0.15f
        }
        lastFrameNs = now

        val autoOn = live.autoRecord || _recordingVideo.value
        val nowMs = System.currentTimeMillis()
        for (t in active) {
            if (!tracker.isStable(t, live.stableFrames, live.minRecordConf)) continue
            val (text, conf) = t.vote()
            val fresh = recentTexts[text]?.let { nowMs - it >= live.dedupeSeconds * 1000L } ?: true
            if (!t.announced) {
                t.announced = true
                if (fresh) notifyNew()
            }
            if (autoOn && !t.recorded) {
                t.recorded = true
                if (fresh) {
                    recentTexts[text] = nowMs
                    val p = t.lastPlate!!
                    record(frame, img, p.copy(text = text, conf = conf, strategy = p.strategy + " + vote×${t.reads.size}"))
                }
            }
        }

        _live.value = LiveState(
            frameW = img.width, frameH = img.height,
            tracks = active.map { t ->
                val (text, conf) = t.vote()
                val p = t.lastPlate
                LiveTrack(t.id, t.box, text, conf, tracker.isStable(t, live.stableFrames, live.minRecordConf), t.recorded,
                    p?.crops?.corners, p?.region?.chosen?.description, p?.model)
            },
            vehicles = frame.vehicles.map { v -> LiveVehicle(v.det.box, v.attrs?.summary()?.takeIf { it.isNotBlank() }) },
            timings = frame.timings, fps = fpsEma, modelsRun = frame.modelsRun, frames = frameIndex, simulated = simulating,
        )
    }

    // --- Simulated camera: replay photos through the live path (no car needed) ---
    @Volatile private var simulating = false
    @Volatile private var lastFrame: RgbImage? = null
    private var simJob: kotlinx.coroutines.Job? = null
    private val _simFrame = MutableStateFlow<Bitmap?>(null)
    val simFrame: StateFlow<Bitmap?> = _simFrame

    /** Feeds each image for [framesPerImage] frames, cycling, until [stopSimulation]. */
    fun startSimulation(images: List<Bitmap>, framesPerImage: Int = 12) {
        if (images.isEmpty()) return
        stopSimulation()
        simulating = true
        tracker.reset()
        val frames = images.map { it to it.toRgbImage() }
        simJob = c.scope.launch(Dispatchers.Default) {
            var i = 0
            while (isActive && simulating) {
                if (c.prepare.value != null || paused) { kotlinx.coroutines.delay(200); continue }
                val (bmp, img) = frames[(i / framesPerImage) % frames.size]
                if (i % framesPerImage == 0 && frames.size > 1) tracker.reset() // scene cut
                _simFrame.value = bmp
                runCatching { analyzeFrame(img) }.onFailure { e ->
                    Log.e("AxiosCamera", "simulated frame failed", e)
                    _live.value = _live.value.copy(error = e.message ?: e.javaClass.simpleName)
                }
                i++
                kotlinx.coroutines.delay(33)
            }
        }
    }

    fun stopSimulation() {
        simulating = false
        simJob?.cancel(); simJob = null
        _simFrame.value = null
        tracker.reset()
        _live.value = LiveState()
    }

    private fun record(frame: FrameResult, img: RgbImage, plate: com.axios.lpr.engine.PlateResult) {
        val cfg = c.settings.value.pipeline
        val live = c.settings.value.live
        val videoSession = recordingSession
        val source = if (videoSession != null) "video" else "auto"
        val single = frame.copy(plates = listOf(plate.copy(index = 0)))
        c.scope.launch {
            val session = videoSession ?: autoSession ?: c.repo.startSession("live_auto").also { autoSession = it }
            val loc = if (live.tagLocation) LocationHelper.last(context) else null
            val id = c.repo.save(single, img, cfg, CaptureMeta(
                source = source, sessionId = session, lat = loc?.first, lon = loc?.second,
                videoPath = recordingFile?.takeIf { videoSession != null }?.let { "videos/${it.name}" },
                videoOffsetMs = if (videoSession != null) System.currentTimeMillis() - recordingStart else null,
            ))
            _recorded.value = (listOf(RecordedPlate(plate.text, plate.conf, id, System.currentTimeMillis())) + _recorded.value).take(30)
        }
    }

    private fun notifyNew() {
        val live = c.settings.value.live
        if (live.beep) tone?.startTone(ToneGenerator.TONE_PROP_BEEP, 90)
        if (live.haptic) vibrator()?.vibrate(VibrationEffect.createOneShot(40, VibrationEffect.DEFAULT_AMPLITUDE))
    }

    private fun vibrator(): Vibrator? = if (Build.VERSION.SDK_INT >= 31) {
        context.getSystemService(VibratorManager::class.java)?.defaultVibrator
    } else @Suppress("DEPRECATION") context.getSystemService(Vibrator::class.java)

    fun endAutoSession() {
        autoSession?.let { s -> c.scope.launch { c.repo.endSession(s) } }
        autoSession = null
    }

    /** Full-resolution still → pipeline (all enabled models) → SQLite. Returns the capture id. */
    suspend fun snap(): Long {
        if (simulating) {
            val img = lastFrame ?: error("No simulated frame yet")
            return withContext(Dispatchers.Default) {
                val cfg = c.settings.value.pipeline
                val frame = c.pipeline.process(img, cfg, capture = true)
                c.repo.save(frame, img, cfg, CaptureMeta(source = "snap", originalName = "simulated camera"))
            }
        }
        val ic = imageCapture ?: error("Switch to photo mode to snap")
        val done = CompletableDeferred<RgbImage>()
        ic.takePicture(analysisExecutor, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                try { done.complete(image.toUpright()) } catch (e: Throwable) { done.completeExceptionally(e) } finally { image.close() }
            }
            override fun onError(exception: ImageCaptureException) { done.completeExceptionally(exception) }
        })
        val img = done.await()
        return withContext(Dispatchers.Default) {
            val cfg = c.settings.value.pipeline
            val frame = c.pipeline.process(img, cfg, capture = true)
            val loc = if (c.settings.value.live.tagLocation) LocationHelper.last(context) else null
            c.repo.save(frame, img, cfg, CaptureMeta(source = "snap", sessionId = recordingSession, lat = loc?.first, lon = loc?.second))
        }
    }

    @SuppressLint("MissingPermission")
    suspend fun startVideo(withAudio: Boolean) {
        val vc = videoCapture ?: error("Switch to video mode to record")
        val dir = File(context.filesDir, "videos").apply { mkdirs() }
        val file = File(dir, "axios-${System.currentTimeMillis()}.mp4")
        recordingSession = c.repo.startSession("video", file.name)
        recordingFile = file
        recordingStart = System.currentTimeMillis()
        val pending = vc.output.prepareRecording(context, FileOutputOptions.Builder(file).build())
        if (withAudio) pending.withAudioEnabled()
        recording = pending.start(ContextCompat.getMainExecutor(context)) { e ->
            if (e is VideoRecordEvent.Finalize) {
                _recordingVideo.value = false
                if (e.hasError()) Log.w("AxiosCamera", "recording finished with error ${e.error}")
            }
        }
        _recordingVideo.value = true
    }

    fun stopVideo() {
        recording?.stop(); recording = null
        recordingSession?.let { s -> c.scope.launch { c.repo.endSession(s) } }
        recordingSession = null
        _recordingVideo.value = false
    }

    fun release() {
        stopSimulation()
        unbind()
        endAutoSession()
        analysisExecutor.shutdown()
        tone?.release()
    }
}
