package com.axios.lpr.engine

/**
 * OCR "country groups" recovered from the SDK catalog (model_resolver.py GROUPS).
 * Each group has a region classifier whose label file lists the region IDs it serves,
 * and optionally a dedicated CRNN.
 */
data class RegionGroup(val name: String, val classifier: Int, val idList: Int, val crnn: Int?, val sample: Int, val sampleTruth: String)

object RegionGroups {
    val all = listOf(
        RegionGroup("2033", 23, 24, 50, 25, "FHW7186"),
        RegionGroup("1195", 26, 27, null, 28, "0907CDS"),
        RegionGroup("1187", 29, 30, 57, 31, "SH7194K"),
        RegionGroup("2207", 32, 33, 53, 34, "RTL015"),
        RegionGroup("1110", 35, 36, 55, 37, "343466"),
        RegionGroup("1106", 38, 39, 60, 40, "990SKR09"),
        RegionGroup("4001", 41, 42, null, 43, "(Thai)8572"),
        RegionGroup("1044", 44, 45, null, 46, "ASI927"),
        RegionGroup("4416", 47, 48, null, 49, "C9998"),
        RegionGroup("4101", 82, 83, null, 84, "3581KEA"),
    )

    fun byClassifier(id: Int) = all.first { it.classifier == id }
    fun byName(name: String) = all.firstOrNull { it.name == name }

    /**
     * Region-ID names. The SDK ships no ID→name table; these come from the bundled samples
     * and today's tests on Malaysian plates, so treat them as inferred.
     */
    val knownNames = mapOf(
        "1012" to "Australia", "2207" to "Victoria",
        "1106" to "Kazakhstan", "1110" to "Kuwait",
        "1125" to "Malaysia", "1187" to "Singapore", "1195" to "Spain",
        "1222" to "USA", "2033" to "North Carolina",
        "4001" to "Thailand (Bangkok)", "9999" to "Unknown / other",
    )

    fun describe(label: String): String {
        val parts = label.split('\t').filter { it.isNotBlank() }
        return parts.joinToString(" / ") { id -> knownNames[id]?.let { "$id $it" } ?: id }
    }
}
