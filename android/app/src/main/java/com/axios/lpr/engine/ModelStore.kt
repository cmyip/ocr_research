package com.axios.lpr.engine

import android.content.Context
import java.io.File
import java.nio.charset.Charset
import java.nio.charset.CodingErrorAction

/** Access to SDK records: raw bytes (models load from memory), text for labels. */
interface ModelFiles {
    val catalog: ModelCatalog
    fun bytes(id: Int): ByteArray
    fun lines(id: Int): List<String>
}

/**
 * Reads records straight from the APK's uncompressed assets/models/ (no copy to storage),
 * and decodes label files as UTF-8 with a cp1252 fallback (rec_16 mixes both).
 */
class AndroidModelStore(private val context: Context) : ModelFiles {
    private val textCache = HashMap<Int, List<String>>()

    override val catalog: ModelCatalog by lazy {
        ModelCatalog.parse(context.assets.open("models.json").bufferedReader().use { it.readText() })
    }

    override fun bytes(id: Int): ByteArray = context.assets.open("models/${catalog[id].file}").use { it.readBytes() }

    @Synchronized
    override fun lines(id: Int): List<String> = textCache.getOrPut(id) { decodeLines(bytes(id)) }

    /** Removes the model copies older builds kept in filesDir. */
    fun cleanLegacyCopies() { File(context.filesDir, "models").deleteRecursively() }

    companion object {
        fun decodeLines(bytes: ByteArray): List<String> = decodeText(bytes).lines().map { it.trimEnd('\r') }.filter { it.isNotBlank() }

        fun decodeText(bytes: ByteArray): String {
            val utf8 = Charsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT).onUnmappableCharacter(CodingErrorAction.REPORT)
            return try {
                utf8.decode(java.nio.ByteBuffer.wrap(bytes)).toString()
            } catch (_: Exception) {
                // Mixed files: decode line by line, falling back to cp1252 for bad lines.
                val cp = Charset.forName("windows-1252")
                String(bytes, Charsets.ISO_8859_1).split('\n').joinToString("\n") { l ->
                    val raw = l.toByteArray(Charsets.ISO_8859_1)
                    try {
                        Charsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
                            .decode(java.nio.ByteBuffer.wrap(raw)).toString()
                    } catch (_: Exception) {
                        String(raw, cp)
                    }
                }
            }
        }
    }
}
