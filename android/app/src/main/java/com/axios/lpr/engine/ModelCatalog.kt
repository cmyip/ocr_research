package com.axios.lpr.engine

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/** One SDK asset as described by assets/models.json. */
@Serializable
data class ModelRecord(
    val id: Int,
    val file: String,
    val kind: String,          // onnx | mnn | text | raw | floatblob | dummy
    val role: String,          // detector | ocr | region | corners | mmr | colour | vtype | vbox | labels | idlist | sample | lookup | config | unknown | dummy
    val title: String,
    val size: Long = 0,
    val input: List<Int>? = null,
    val outputs: List<String>? = null,
    val preprocess: String? = null,
    val labels: List<Int>? = null,
    val sample: Int? = null,
    val shape: List<Int>? = null,
    val truth: String? = null,
    val group: String? = null,
    val crnn: Int? = null,
    val classes: Int? = null,
    val lookup: String? = null,
    val notes: String? = null,
) {
    val runnable: Boolean get() = kind == "onnx" || kind == "mnn"
}

@Serializable
data class ModelCatalog(val records: List<ModelRecord>) {
    operator fun get(id: Int): ModelRecord = records[id]
    fun byRole(role: String) = records.filter { it.role == role }

    companion object {
        val json = Json { ignoreUnknownKeys = true }
        fun parse(text: String): ModelCatalog = json.decodeFromString(serializer(), text)
    }
}

/** Model IDs used by the pipeline (see HANDOVER.md and model_resolver.py). */
object Ids {
    const val DET_320 = 72
    const val DET_640 = 75
    const val CORNERS = 71
    const val VBOX = 78
    const val MMR = 15
    const val COLOUR = 19
    const val VTYPE = 21
    val CRNNS = (50..69).toList()
    val REGION_CLASSIFIERS = listOf(23, 26, 29, 32, 35, 38, 41, 44, 47, 82)
    const val DEFAULT_CRNN = 57
}
