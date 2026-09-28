package com.axios.lpr.ui.tables

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.axios.lpr.AppContainer
import com.axios.lpr.engine.RegionGroups

/** Browse the SDK's text records: lookup tables, region-ID lists, label files, ADR config. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TablesScreen(c: AppContainer, onBack: () -> Unit) {
    val sources: List<Pair<String, Int>> = remember {
        listOf("DE districts" to 2, "ES provinces" to 3, "JO categories" to 85, "KZ regions" to 86) +
            RegionGroups.all.map { "Region IDs ${it.name}" to it.idList } +
            listOf("Country IDs" to 0, "Make/model" to 16, "Colours" to 20, "Vehicle types" to 22, "Detector classes" to 73, "ADR config" to 1)
    }
    var sel by remember { mutableStateOf(sources.first()) }
    var q by remember { mutableStateOf("") }
    val lines = remember(sel) { runCatching { c.modelStore.lines(sel.second) }.getOrDefault(emptyList()) }
    val shown = remember(lines, q) { if (q.isBlank()) lines else lines.filter { it.contains(q, ignoreCase = true) } }

    Scaffold(topBar = {
        TopAppBar(title = { Text("Reference tables") }, navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") } })
    }) { pad ->
        Column(Modifier.fillMaxSize().padding(pad).imePadding()) {
            Row(Modifier.horizontalScroll(rememberScrollState()).padding(horizontal = 12.dp), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                sources.forEach { s -> FilterChip(sel == s, { sel = s }, { Text(s.first) }) }
            }
            OutlinedTextField(q, { q = it }, Modifier.fillMaxWidth().padding(12.dp), placeholder = { Text("Filter ${lines.size} lines of rec_%02d".format(sel.second)) }, singleLine = true)
            LazyColumn(Modifier.fillMaxSize().padding(horizontal = 12.dp)) {
                items(shown.take(5000)) { line ->
                    val parts = line.split(';', '\t', limit = 2)
                    Row(Modifier.padding(vertical = 2.dp)) {
                        Text(parts[0], Modifier.width(110.dp), fontFamily = FontFamily.Monospace, fontSize = 12.sp, color = MaterialTheme.colorScheme.secondary)
                        Text(parts.getOrElse(1) { "" }.let { if (sel.second in RegionGroups.all.map { g -> g.idList } + 0) RegionGroups.describe(line) else it },
                            fontSize = 12.sp)
                    }
                }
            }
        }
    }
}
