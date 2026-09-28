package com.axios.lpr.engine

import kotlinx.serialization.Serializable

@Serializable
data class LookupHint(val table: String, val key: String, val value: String)

/**
 * SDK post-processing tables (rec_02 DE districts, rec_03 ES provinces, rec_85 JO categories,
 * rec_86 KZ regions). The SDK's rules for applying them are unknown, so these are shown as
 * format-based hints only.
 */
class Lookups(files: ModelFiles) {
    data class Table(val id: Int, val name: String, val entries: List<Pair<String, String>>) {
        val map: Map<String, String> = entries.groupBy({ it.first }, { it.second }).mapValues { it.value.distinct().joinToString(" / ") }
    }

    val tables: List<Table> = listOf(
        2 to "German district", 3 to "Spanish province", 85 to "Jordanian category", 86 to "Kazakh region",
    ).map { (id, name) ->
        Table(id, name, files.lines(id).mapNotNull { l ->
            val i = l.indexOf(';')
            if (i <= 0) null else l.substring(0, i).trim() to l.substring(i + 1).trim()
        }.filter { it.second.isNotEmpty() })
    }

    private val de = tables[0].map
    private val es = tables[1].map
    private val jo = tables[2].map
    private val kz = tables[3].map

    /**
     * [routedGroup]: the region group the classifiers chose, if any. A routed plate only gets
     * hints for its own country (KZ = group 1106, ES = group 1195); DE/JO have no known group,
     * so they are only suggested for unrouted plates.
     */
    fun hints(text: String, routedGroup: String? = null): List<LookupHint> {
        val t = text.replace(" ", "").uppercase()
        fun allowed(table: String) = routedGroup == null || (table == "Kazakh region" && routedGroup == "1106") ||
            (table == "Spanish province" && routedGroup == "1195")
        return rawHints(t).filter { allowed(it.table) }
    }

    private fun rawHints(t: String): List<LookupHint> {
        val out = ArrayList<LookupHint>()
        Regex("^\\d{3}[A-Z]{2,3}(\\d{2})$").find(t)?.let { m ->
            kz[m.groupValues[1]]?.let { out += LookupHint("Kazakh region", m.groupValues[1], it) }
        }
        Regex("^([A-Z]{1,2})\\d{4}[A-Z]{0,2}$").find(t)?.let { m ->
            es[m.groupValues[1]]?.let { out += LookupHint("Spanish province", m.groupValues[1], it) }
        }
        Regex("^([A-Z]{2,5})\\d{1,4}[EH]?$").find(t)?.let { m ->
            val letters = m.groupValues[1]
            for (n in 3 downTo 1) {
                if (letters.length - n < 1) continue // German plates keep 1–2 letters after the district
                val k = letters.substring(0, n)
                de[k]?.let { out += LookupHint("German district", k, it) }
            }
        }
        Regex("^(\\d{1,2})\\d{4,5}$").find(t)?.let { m ->
            jo[m.groupValues[1]]?.let { out += LookupHint("Jordanian category", m.groupValues[1], it) }
        }
        return out
    }
}
