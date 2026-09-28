package com.axios.lpr.data

import android.content.Context
import androidx.room.withTransaction
import com.axios.lpr.engine.FrameResult
import com.axios.lpr.engine.PipelineConfig
import com.axios.lpr.engine.RgbImage
import com.axios.lpr.engine.saveJpeg
import com.axios.lpr.engine.savePng
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.builtins.MapSerializer
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.json.Json
import java.io.File
import java.util.UUID

data class CaptureMeta(
    val source: String,
    val sessionId: Long? = null,
    val originalName: String? = null,
    val videoPath: String? = null,
    val videoOffsetMs: Long? = null,
    val lat: Double? = null,
    val lon: Double? = null,
    val timestamp: Long = System.currentTimeMillis(),
)

/**
 * Stores every recognised plate with its images. Image files live under
 * filesDir/captures/<uuid>/; SQLite (Room) holds relative paths and all metadata.
 */
class CaptureRepository(private val context: Context, val db: LprDatabase) {
    val dao = db.dao()
    val root: File = File(context.filesDir, "captures").apply { mkdirs() }
    private val json = Json { encodeDefaults = true }

    fun file(rel: String): File = File(context.filesDir, rel)

    suspend fun save(frame: FrameResult, image: RgbImage, cfg: PipelineConfig, meta: CaptureMeta, extraFiles: ((File) -> Unit)? = null): Long = withContext(Dispatchers.IO) {
        val dirName = UUID.randomUUID().toString()
        val dir = File(root, dirName).apply { mkdirs() }
        fun rel(name: String) = "captures/$dirName/$name"

        image.saveJpeg(File(dir, "frame.jpg"), 90)
        extraFiles?.invoke(dir)
        val vehicleFiles = frame.vehicles.map { v ->
            val b = v.det.box.clip(image.width, image.height)
            val name = "vehicle_${v.index}.jpg"
            if (b.w >= 8 && b.h >= 8) { image.crop(b.x1, b.y1, b.x2, b.y2).saveJpeg(File(dir, name), 85); rel(name) } else null
        }
        val plateFiles = frame.plates.map { p ->
            p.crops.raw.saveJpeg(File(dir, "plate_${p.index}_raw.jpg"), 95)
            p.crops.ocrInput.savePng(File(dir, "plate_${p.index}_input96x48.png"))
            p.crops.rectified?.savePng(File(dir, "plate_${p.index}_rectified.png"))
            Triple(rel("plate_${p.index}_raw.jpg"), rel("plate_${p.index}_input96x48.png"), p.crops.rectified?.let { rel("plate_${p.index}_rectified.png") })
        }

        db.withTransaction {
            val captureId = dao.insertCapture(
                CaptureEntity(
                    timestamp = meta.timestamp, source = meta.source, dir = "captures/$dirName", imagePath = rel("frame.jpg"),
                    width = image.width, height = image.height, lat = meta.lat, lon = meta.lon, sessionId = meta.sessionId,
                    videoPath = meta.videoPath, videoOffsetMs = meta.videoOffsetMs, originalName = meta.originalName,
                    configJson = json.encodeToString(PipelineConfig.serializer(), cfg),
                    timingsJson = json.encodeToString(MapSerializer(String.serializer(), Float.serializer()), frame.timings),
                    totalMs = frame.totalMs, modelsRun = frame.modelsRun.sorted().joinToString(","),
                ),
            )
            val vehicleIds = frame.vehicles.mapIndexed { i, v ->
                val a = v.attrs
                dao.insertVehicle(
                    VehicleEntity(
                        captureId = captureId, x1 = v.det.box.x1, y1 = v.det.box.y1, x2 = v.det.box.x2, y2 = v.det.box.y2, score = v.det.score,
                        makeModel = a?.makeModel?.label, makeModelConf = a?.makeModel?.conf, pose = a?.pose?.label, poseConf = a?.pose?.conf,
                        colour = a?.colour?.label, colourConf = a?.colour?.conf, type = a?.type?.label, typeConf = a?.type?.conf,
                        refinedBox = a?.refinedBox?.let { json.encodeToString(com.axios.lpr.engine.Box.serializer(), it) },
                        cropPath = vehicleFiles[i],
                    ),
                )
            }
            frame.plates.forEachIndexed { i, p ->
                val (raw, input, rect) = plateFiles[i]
                val chosen = p.region?.chosen
                val plateId = dao.insertPlate(
                    PlateEntity(
                        captureId = captureId, vehicleId = p.vehicleIndex?.let { vehicleIds.getOrNull(it) },
                        x1 = p.det.box.x1, y1 = p.det.box.y1, x2 = p.det.box.x2, y2 = p.det.box.y2, detScore = p.det.score,
                        predictedText = p.text, predictedConf = p.conf, ocrModel = p.model, strategy = p.strategy,
                        charConfJson = json.encodeToString(ListSerializer(Float.serializer()), p.charConf),
                        regionGroup = chosen?.group, regionLabel = chosen?.label, regionConf = chosen?.conf,
                        regionJson = p.region?.let { json.encodeToString(com.axios.lpr.engine.RegionDecision.serializer(), it) },
                        rectified = p.rectified, deshear = p.deshear, pad = p.pad,
                        cornersJson = p.crops.corners?.let { json.encodeToString(ListSerializer(Float.serializer()), it.toList()) },
                        hintsJson = if (p.hints.isEmpty()) null else json.encodeToString(ListSerializer(com.axios.lpr.engine.LookupHint.serializer()), p.hints),
                        rawCropPath = raw, inputCropPath = input, rectifiedCropPath = rect,
                    ),
                )
                dao.insertReads(p.reads.map { r ->
                    PlateReadEntity(
                        plateId = plateId, model = r.model, text = r.text, conf = r.conf, classes = r.classes, decodable = r.decodable,
                        indicesJson = json.encodeToString(ListSerializer(Int.serializer()), r.indices), ms = r.ms,
                    )
                })
            }
            captureId
        }
    }

    /** Stores the corrected reading; blank clears it. The prediction column is never touched. */
    suspend fun setCorrection(plateId: Long, text: String?) {
        val t = text?.trim()?.uppercase()?.takeIf { it.isNotEmpty() }
        dao.setCorrection(plateId, t, t?.let { System.currentTimeMillis() })
    }

    suspend fun delete(captureId: Long) = withContext(Dispatchers.IO) {
        val dir = dao.captureDir(captureId)
        dao.deleteCapture(captureId)
        dir?.let { File(context.filesDir, it).deleteRecursively() }
    }

    suspend fun deleteAll() = withContext(Dispatchers.IO) {
        dao.deleteAllCaptures(); dao.deleteAllSessions()
        root.deleteRecursively(); root.mkdirs()
    }

    suspend fun startSession(mode: String, note: String? = null) = dao.insertSession(SessionEntity(startedAt = System.currentTimeMillis(), mode = mode, note = note))
    suspend fun endSession(id: Long) = dao.endSession(id, System.currentTimeMillis())
}
