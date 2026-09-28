package com.axios.lpr.ui

import android.net.Uri
import android.util.Log
import com.axios.lpr.AppContainer
import com.axios.lpr.data.CaptureMeta
import com.axios.lpr.engine.ImageLoader
import com.axios.lpr.engine.toRgbImage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

data class ImportState(
    val running: Boolean = false,
    val total: Int = 0,
    val done: Int = 0,
    val current: String? = null,
    val captureIds: List<Long> = emptyList(),
    val errors: List<String> = emptyList(),
    val sessionId: Long? = null,
    val finishedAt: Long = 0,
)

/**
 * Runs images from the Photo Picker, SAF or share intents through the same pipeline as a snap.
 * Several images form a batch session; each result is stored like a snap with source = import.
 */
class ImportController(private val c: AppContainer) {
    private val _state = MutableStateFlow(ImportState())
    val state: StateFlow<ImportState> = _state
    private val lock = Mutex()

    fun import(uris: List<Uri>, via: String) {
        if (uris.isEmpty()) return
        c.scope.launch(Dispatchers.Default) {
            lock.withLock {
                val session = if (uris.size > 1) c.repo.startSession("import_batch", "${uris.size} images via $via") else null
                _state.value = ImportState(running = true, total = uris.size, sessionId = session)
                val ids = ArrayList<Long>()
                val errors = ArrayList<String>()
                val cr = c.app.contentResolver
                uris.forEachIndexed { i, uri ->
                    val name = ImageLoader.displayName(cr, uri) ?: uri.lastPathSegment ?: "image"
                    _state.value = _state.value.copy(done = i, current = name)
                    try {
                        val bmp = ImageLoader.decode(cr, uri)
                        val img = bmp.toRgbImage()
                        val cfg = c.settings.value.pipeline
                        val frame = c.pipeline.process(img, cfg, capture = true)
                        val id = c.repo.save(frame, img, cfg, CaptureMeta(source = "import", sessionId = session, originalName = name)) { dir ->
                            // Keep the untouched original too: content URIs can expire.
                            val ext = name.substringAfterLast('.', "jpg").take(5)
                            cr.openInputStream(uri)?.use { input -> java.io.File(dir, "original.$ext").outputStream().use { input.copyTo(it) } }
                        }
                        ids += id
                    } catch (e: Throwable) {
                        Log.e("AxiosImport", "failed $uri", e)
                        errors += "$name: ${e.message ?: e.javaClass.simpleName}"
                    }
                    _state.value = _state.value.copy(done = i + 1, captureIds = ids.toList(), errors = errors.toList())
                }
                session?.let { c.repo.endSession(it) }
                _state.value = _state.value.copy(running = false, current = null, finishedAt = System.currentTimeMillis())
            }
        }
    }

    fun dismiss() { _state.value = ImportState() }
}
