package com.axios.lpr.data

import android.content.Context
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/** CSV / JSON / training-set exports written to cacheDir/exports and shared via FileProvider. */
class Exporter(private val context: Context, private val repo: CaptureRepository) {
    private val dir get() = File(context.cacheDir, "exports").apply { mkdirs() }
    private fun stamp() = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
    private val iso = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSX", Locale.US).apply { timeZone = TimeZone.getTimeZone("UTC") }

    suspend fun csv(onlyCorrected: Boolean = false): File {
        val rows = repo.dao.exportRows(onlyCorrected)
        val f = File(dir, "axios-lpr-plates-${stamp()}.csv")
        f.bufferedWriter().use { w ->
            w.appendLine(PLATE_HEADER.joinToString(","))
            rows.forEach { w.appendLine(plateCsv(it)) }
        }
        return f
    }

    fun plateCsv(r: PlateRow): String = listOf(
        r.plateId, r.captureId, iso.format(Date(r.timestamp)), r.source, r.sessionId ?: "", r.predictedText, "%.4f".format(Locale.US, r.predictedConf),
        r.correctedText ?: "", r.ocrModel?.let { "rec_$it" } ?: "", r.strategy, r.regionLabel?.replace('\t', ' ') ?: "", "%.3f".format(Locale.US, r.detScore),
        r.rectified, r.makeModel ?: "", r.colour ?: "", r.type ?: "", r.lat ?: "", r.lon ?: "", r.originalName ?: "", r.imagePath,
    ).joinToString(",") { csvCell(it.toString()) }

    suspend fun json(): File {
        val rows = repo.dao.exportRows(false)
        val f = File(dir, "axios-lpr-${stamp()}.json")
        val sb = StringBuilder("[\n")
        rows.forEachIndexed { i, r ->
            val reads = repo.dao.reads(r.plateId).joinToString(",") { """{"model":${it.model},"text":${q(it.text)},"conf":${it.conf},"classes":${it.classes}}""" }
            sb.append(
                """  {"plate_id":${r.plateId},"capture_id":${r.captureId},"timestamp":${q(iso.format(Date(r.timestamp)))},"source":${q(r.source)},""" +
                    """"predicted":${q(r.predictedText)},"conf":${r.predictedConf},"corrected":${r.correctedText?.let(::q) ?: "null"},""" +
                    """"ocr_model":${r.ocrModel ?: "null"},"strategy":${q(r.strategy)},"region":${r.regionLabel?.let(::q) ?: "null"},""" +
                    """"vehicle":{"make_model":${r.makeModel?.let(::q) ?: "null"},"colour":${r.colour?.let(::q) ?: "null"},"type":${r.type?.let(::q) ?: "null"}},""" +
                    """"lat":${r.lat ?: "null"},"lon":${r.lon ?: "null"},"image":${q(r.imagePath)},"reads":[$reads]}""",
            )
            sb.append(if (i < rows.size - 1) ",\n" else "\n")
        }
        sb.append("]\n")
        f.writeText(sb.toString())
        return f
    }

    /**
     * Training set for future fine-tuning: the exact 96×48 CRNN inputs, the raw crops, and
     * labels.csv. [onlyCorrected] keeps rows with a user-entered reading.
     */
    suspend fun trainingZip(onlyCorrected: Boolean): Pair<File, Int> {
        val rows = repo.dao.exportRows(onlyCorrected)
        val f = File(dir, "axios-lpr-training-${stamp()}.zip")
        ZipOutputStream(f.outputStream().buffered()).use { z ->
            val labels = StringBuilder("image,raw_image,predicted,corrected,label,ocr_model,deshear,pad,rectified,region_id,conf\n")
            rows.forEach { r ->
                val input = repo.file(r.inputCropPath)
                val raw = repo.file(r.rawCropPath)
                if (!input.exists()) return@forEach
                val inName = "images/${r.plateId}_input96x48.png"
                val rawName = "images/${r.plateId}_raw.jpg"
                z.putNextEntry(ZipEntry(inName)); input.inputStream().use { it.copyTo(z) }; z.closeEntry()
                if (raw.exists()) { z.putNextEntry(ZipEntry(rawName)); raw.inputStream().use { it.copyTo(z) }; z.closeEntry() }
                labels.append(
                    listOf(inName, rawName, r.predictedText, r.correctedText ?: "", r.correctedText ?: r.predictedText, r.ocrModel?.let { "rec_$it" } ?: "",
                        r.deshear, r.pad, r.rectified, r.regionLabel?.replace('\t', ' ') ?: "", "%.4f".format(Locale.US, r.predictedConf))
                        .joinToString(",") { csvCell(it.toString()) },
                ).append('\n')
            }
            z.putNextEntry(ZipEntry("labels.csv")); z.write(labels.toString().toByteArray()); z.closeEntry()
            z.putNextEntry(ZipEntry("README.txt"))
            z.write(
                ("Axios LPR training export\n" +
                    "images/*_input96x48.png = exact CRNN input (BGR order is applied at inference; files are RGB)\n" +
                    "label = corrected reading if present, else prediction. Alphabet 0-9A-Z; CTC blank = last class.\n").toByteArray(),
            )
            z.closeEntry()
        }
        return f to rows.size
    }

    companion object {
        val PLATE_HEADER = listOf("plate_id", "capture_id", "timestamp_utc", "source", "session_id", "predicted", "conf", "corrected", "ocr_model",
            "strategy", "region", "det_score", "rectified", "make_model", "colour", "vehicle_type", "lat", "lon", "original_name", "image")

        fun csvCell(s: String): String = if (s.any { it == ',' || it == '"' || it == '\n' }) "\"" + s.replace("\"", "\"\"") + "\"" else s
        private fun q(s: String) = "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\t", "\\t") + "\""
    }
}
