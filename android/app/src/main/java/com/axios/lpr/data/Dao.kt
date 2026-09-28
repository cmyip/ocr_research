package com.axios.lpr.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import androidx.room.Transaction
import kotlinx.coroutines.flow.Flow

@Dao
interface LprDao {
    @Insert suspend fun insertSession(s: SessionEntity): Long
    @Query("UPDATE session SET ended_at = :at WHERE id = :id") suspend fun endSession(id: Long, at: Long)
    @Query("SELECT * FROM session ORDER BY started_at DESC") fun sessions(): Flow<List<SessionEntity>>

    @Insert suspend fun insertCapture(c: CaptureEntity): Long
    @Insert suspend fun insertVehicle(v: VehicleEntity): Long
    @Insert suspend fun insertPlate(p: PlateEntity): Long
    @Insert suspend fun insertReads(r: List<PlateReadEntity>)

    @Transaction
    @Query("SELECT * FROM capture WHERE id = :id")
    fun observeCapture(id: Long): Flow<CaptureWithPlates?>

    @Transaction
    @Query("SELECT * FROM capture WHERE id = :id")
    suspend fun capture(id: Long): CaptureWithPlates?

    @Query("SELECT dir FROM capture WHERE id = :id") suspend fun captureDir(id: Long): String?
    @Query("DELETE FROM capture WHERE id = :id") suspend fun deleteCapture(id: Long)
    @Query("SELECT dir FROM capture") suspend fun allDirs(): List<String>
    @Query("DELETE FROM capture") suspend fun deleteAllCaptures()
    @Query("DELETE FROM session") suspend fun deleteAllSessions()

    @Query("UPDATE plate SET corrected_text = :text, corrected_at = :at WHERE id = :id")
    suspend fun setCorrection(id: Long, text: String?, at: Long?)

    @Query("SELECT * FROM plate WHERE id = :id") suspend fun plate(id: Long): PlateEntity?

    @Query(
        """
        SELECT p.id AS plateId, c.id AS captureId, c.timestamp, c.source, c.session_id AS sessionId,
               p.predicted_text AS predictedText, p.predicted_conf AS predictedConf, p.corrected_text AS correctedText,
               p.ocr_model AS ocrModel, p.strategy, p.region_label AS regionLabel, p.region_group AS regionGroup,
               p.region_conf AS regionConf, p.det_score AS detScore, p.rectified, p.deshear, p.pad,
               p.raw_crop_path AS rawCropPath, p.input_crop_path AS inputCropPath, c.image_path AS imagePath,
               c.lat, c.lon, c.original_name AS originalName,
               v.make_model AS makeModel, v.colour, v.type
        FROM plate p JOIN capture c ON c.id = p.capture_id LEFT JOIN vehicle v ON v.id = p.vehicle_id
        WHERE (:q = '' OR p.predicted_text LIKE '%' || :q || '%' OR p.corrected_text LIKE '%' || :q || '%')
          AND (:needsReview = 0 OR p.corrected_text IS NULL)
          AND (:onlyCorrected = 0 OR p.corrected_text IS NOT NULL)
          AND (:sessionId IS NULL OR c.session_id = :sessionId)
          AND (:source = '' OR c.source = :source)
          AND c.timestamp BETWEEN :from AND :to
        ORDER BY c.timestamp DESC, p.id
        """,
    )
    fun plates(
        q: String = "", needsReview: Boolean = false, onlyCorrected: Boolean = false, sessionId: Long? = null,
        source: String = "", from: Long = 0, to: Long = Long.MAX_VALUE,
    ): Flow<List<PlateRow>>

    @Query(
        """
        SELECT p.id AS plateId, c.id AS captureId, c.timestamp, c.source, c.session_id AS sessionId,
               p.predicted_text AS predictedText, p.predicted_conf AS predictedConf, p.corrected_text AS correctedText,
               p.ocr_model AS ocrModel, p.strategy, p.region_label AS regionLabel, p.region_group AS regionGroup,
               p.region_conf AS regionConf, p.det_score AS detScore, p.rectified, p.deshear, p.pad,
               p.raw_crop_path AS rawCropPath, p.input_crop_path AS inputCropPath, c.image_path AS imagePath,
               c.lat, c.lon, c.original_name AS originalName,
               v.make_model AS makeModel, v.colour, v.type
        FROM plate p JOIN capture c ON c.id = p.capture_id LEFT JOIN vehicle v ON v.id = p.vehicle_id
        WHERE (:onlyCorrected = 0 OR p.corrected_text IS NOT NULL)
        ORDER BY c.timestamp, p.id
        """,
    )
    suspend fun exportRows(onlyCorrected: Boolean): List<PlateRow>

    @Query("SELECT * FROM plate_read WHERE plate_id = :plateId ORDER BY conf DESC") suspend fun reads(plateId: Long): List<PlateReadEntity>

    @Query("SELECT COUNT(*) FROM plate") fun plateCount(): Flow<Int>
    @Query("SELECT COUNT(*) FROM plate WHERE corrected_text IS NOT NULL") fun correctedCount(): Flow<Int>
    @Query("SELECT COUNT(*) FROM plate WHERE corrected_text IS NOT NULL AND corrected_text != predicted_text") fun disagreeCount(): Flow<Int>

    /** For auto-record dedupe: was this text recorded recently? */
    @Query("SELECT COUNT(*) FROM plate p JOIN capture c ON c.id = p.capture_id WHERE p.predicted_text = :text AND c.timestamp >= :since")
    suspend fun recentCount(text: String, since: Long): Int
}
