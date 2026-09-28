package com.axios.lpr.data

import androidx.room.ColumnInfo
import androidx.room.Embedded
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey
import androidx.room.Relation

@Entity(tableName = "session")
data class SessionEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    @ColumnInfo(name = "started_at") val startedAt: Long,
    @ColumnInfo(name = "ended_at") val endedAt: Long? = null,
    val mode: String, // live_auto | video | import_batch
    val note: String? = null,
)

@Entity(
    tableName = "capture",
    foreignKeys = [ForeignKey(entity = SessionEntity::class, parentColumns = ["id"], childColumns = ["session_id"], onDelete = ForeignKey.SET_NULL)],
    indices = [Index("session_id"), Index("timestamp")],
)
data class CaptureEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val timestamp: Long,
    val source: String, // snap | import | auto | video
    /** Directory under filesDir/captures holding this capture's images. */
    val dir: String,
    @ColumnInfo(name = "image_path") val imagePath: String,
    val width: Int,
    val height: Int,
    val lat: Double? = null,
    val lon: Double? = null,
    @ColumnInfo(name = "session_id") val sessionId: Long? = null,
    @ColumnInfo(name = "video_path") val videoPath: String? = null,
    @ColumnInfo(name = "video_offset_ms") val videoOffsetMs: Long? = null,
    @ColumnInfo(name = "original_name") val originalName: String? = null,
    @ColumnInfo(name = "config_json") val configJson: String,
    @ColumnInfo(name = "timings_json") val timingsJson: String,
    @ColumnInfo(name = "total_ms") val totalMs: Float,
    @ColumnInfo(name = "models_run") val modelsRun: String,
)

@Entity(
    tableName = "vehicle",
    foreignKeys = [ForeignKey(entity = CaptureEntity::class, parentColumns = ["id"], childColumns = ["capture_id"], onDelete = ForeignKey.CASCADE)],
    indices = [Index("capture_id")],
)
data class VehicleEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    @ColumnInfo(name = "capture_id") val captureId: Long,
    val x1: Float, val y1: Float, val x2: Float, val y2: Float,
    val score: Float,
    @ColumnInfo(name = "make_model") val makeModel: String? = null,
    @ColumnInfo(name = "make_model_conf") val makeModelConf: Float? = null,
    val pose: String? = null,
    @ColumnInfo(name = "pose_conf") val poseConf: Float? = null,
    val colour: String? = null,
    @ColumnInfo(name = "colour_conf") val colourConf: Float? = null,
    val type: String? = null,
    @ColumnInfo(name = "type_conf") val typeConf: Float? = null,
    @ColumnInfo(name = "refined_box") val refinedBox: String? = null,
    @ColumnInfo(name = "crop_path") val cropPath: String? = null,
)

@Entity(
    tableName = "plate",
    foreignKeys = [
        ForeignKey(entity = CaptureEntity::class, parentColumns = ["id"], childColumns = ["capture_id"], onDelete = ForeignKey.CASCADE),
        ForeignKey(entity = VehicleEntity::class, parentColumns = ["id"], childColumns = ["vehicle_id"], onDelete = ForeignKey.SET_NULL),
    ],
    indices = [Index("capture_id"), Index("vehicle_id"), Index("predicted_text"), Index("corrected_text")],
)
data class PlateEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    @ColumnInfo(name = "capture_id") val captureId: Long,
    @ColumnInfo(name = "vehicle_id") val vehicleId: Long? = null,
    val x1: Float, val y1: Float, val x2: Float, val y2: Float,
    @ColumnInfo(name = "det_score") val detScore: Float,
    @ColumnInfo(name = "predicted_text") val predictedText: String,
    @ColumnInfo(name = "predicted_conf") val predictedConf: Float,
    @ColumnInfo(name = "ocr_model") val ocrModel: Int?,
    val strategy: String,
    @ColumnInfo(name = "char_conf_json") val charConfJson: String,
    @ColumnInfo(name = "region_group") val regionGroup: String? = null,
    @ColumnInfo(name = "region_label") val regionLabel: String? = null,
    @ColumnInfo(name = "region_conf") val regionConf: Float? = null,
    @ColumnInfo(name = "region_json") val regionJson: String? = null,
    val rectified: Boolean,
    val deshear: Float,
    val pad: Float,
    @ColumnInfo(name = "corners_json") val cornersJson: String? = null,
    @ColumnInfo(name = "hints_json") val hintsJson: String? = null,
    /** User-entered true reading for future fine-tuning. The prediction is never overwritten. */
    @ColumnInfo(name = "corrected_text") val correctedText: String? = null,
    @ColumnInfo(name = "corrected_at") val correctedAt: Long? = null,
    /** Padded detector crop at native resolution. */
    @ColumnInfo(name = "raw_crop_path") val rawCropPath: String,
    /** Exact 96×48 image the CRNN saw. */
    @ColumnInfo(name = "input_crop_path") val inputCropPath: String,
    @ColumnInfo(name = "rectified_crop_path") val rectifiedCropPath: String? = null,
) {
    val finalText: String get() = correctedText ?: predictedText
}

@Entity(
    tableName = "plate_read",
    foreignKeys = [ForeignKey(entity = PlateEntity::class, parentColumns = ["id"], childColumns = ["plate_id"], onDelete = ForeignKey.CASCADE)],
    indices = [Index("plate_id")],
)
data class PlateReadEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    @ColumnInfo(name = "plate_id") val plateId: Long,
    val model: Int,
    val text: String,
    val conf: Float,
    val classes: Int,
    val decodable: Boolean,
    @ColumnInfo(name = "indices_json") val indicesJson: String,
    val ms: Float,
)

data class PlateWithReads(
    @Embedded val plate: PlateEntity,
    @Relation(parentColumn = "id", entityColumn = "plate_id") val reads: List<PlateReadEntity>,
)

data class CaptureWithPlates(
    @Embedded val capture: CaptureEntity,
    @Relation(entity = PlateEntity::class, parentColumn = "id", entityColumn = "capture_id") val plates: List<PlateWithReads>,
    @Relation(parentColumn = "id", entityColumn = "capture_id") val vehicles: List<VehicleEntity>,
)

/** Flat row for history lists and exports. */
data class PlateRow(
    val plateId: Long,
    val captureId: Long,
    val timestamp: Long,
    val source: String,
    val sessionId: Long?,
    val predictedText: String,
    val predictedConf: Float,
    val correctedText: String?,
    val ocrModel: Int?,
    val strategy: String,
    val regionLabel: String?,
    val regionGroup: String?,
    val regionConf: Float?,
    val detScore: Float,
    val rectified: Boolean,
    val deshear: Float,
    val pad: Float,
    val rawCropPath: String,
    val inputCropPath: String,
    val imagePath: String,
    val lat: Double?,
    val lon: Double?,
    val originalName: String?,
    val makeModel: String?,
    val colour: String?,
    val type: String?,
)
