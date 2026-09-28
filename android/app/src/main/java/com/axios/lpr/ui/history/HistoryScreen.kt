package com.axios.lpr.ui.history

import android.content.Context
import android.content.Intent
import android.widget.Toast
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import com.axios.lpr.AppContainer
import com.axios.lpr.data.PlateRow
import com.axios.lpr.ui.ImportMenu
import com.axios.lpr.ui.common.BitmapCache
import com.axios.lpr.ui.common.Caption
import com.axios.lpr.ui.common.ConfBadge
import com.axios.lpr.ui.common.FileImage
import com.axios.lpr.ui.common.PlateChip
import com.axios.lpr.ui.theme.AxiosColors
import kotlinx.coroutines.launch
import java.io.File
import java.text.DateFormat
import java.util.Date

enum class Review(val label: String) { ALL("All"), NEEDS("Needs review"), CORRECTED("Corrected") }

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HistoryScreen(c: AppContainer, initialSession: Long?, onOpen: (Long) -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var query by remember { mutableStateOf("") }
    var review by remember { mutableStateOf(Review.ALL) }
    var source by remember { mutableStateOf("") }
    var session by remember(initialSession) { mutableStateOf(initialSession) }
    var menu by remember { mutableStateOf(false) }
    var confirmWipe by remember { mutableStateOf(false) }
    val rows by remember(query, review, source, session) {
        c.repo.dao.plates(query.trim(), review == Review.NEEDS, review == Review.CORRECTED, session, source)
    }.collectAsState(initial = emptyList())
    val total by c.repo.dao.plateCount().collectAsState(0)
    val corrected by c.repo.dao.correctedCount().collectAsState(0)
    val disagree by c.repo.dao.disagreeCount().collectAsState(0)
    val sessions by c.repo.dao.sessions().collectAsState(emptyList())

    fun export(label: String, block: suspend () -> Pair<File, String>) = scope.launch {
        runCatching { block() }.onSuccess { (f, mime) -> share(context, f, mime, label) }
            .onFailure { Toast.makeText(context, "Export failed: ${it.message}", Toast.LENGTH_LONG).show() }
    }

    Scaffold(topBar = {
        TopAppBar(title = { Column { Text("History"); Caption("$total plates · $corrected corrected · $disagree fixes") } }, actions = {
            ImportMenu(c, compact = true)
            Box {
                IconButton(onClick = { menu = true }) { Icon(Icons.Filled.MoreVert, "More") }
                DropdownMenu(menu, { menu = false }) {
                    DropdownMenuItem({ Text("Export CSV") }, leadingIcon = { Icon(Icons.Filled.Download, null) }, onClick = {
                        menu = false; export("CSV") { c.exporter.csv() to "text/csv" }
                    })
                    DropdownMenuItem({ Text("Export JSON (with every model's read)") }, onClick = {
                        menu = false; export("JSON") { c.exporter.json() to "application/json" }
                    })
                    DropdownMenuItem({ Text("Training set: corrected only") }, onClick = {
                        menu = false; export("training zip") { c.exporter.trainingZip(onlyCorrected = true).first to "application/zip" }
                    })
                    DropdownMenuItem({ Text("Training set: all plates") }, onClick = {
                        menu = false; export("training zip") { c.exporter.trainingZip(onlyCorrected = false).first to "application/zip" }
                    })
                    HorizontalDivider()
                    DropdownMenuItem({ Text("Delete all records", color = AxiosColors.Warn) }, onClick = { menu = false; confirmWipe = true })
                }
            }
        })
    }) { pad ->
        Column(Modifier.fillMaxSize().padding(pad).imePadding()) {
            OutlinedTextField(query, { query = it.uppercase() }, Modifier.fillMaxWidth().padding(horizontal = 12.dp),
                leadingIcon = { Icon(Icons.Filled.Search, null) }, placeholder = { Text("Search predicted or corrected text") }, singleLine = true)
            Row(Modifier.horizontalScroll(rememberScrollState()).padding(horizontal = 12.dp, vertical = 6.dp), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                Review.entries.forEach { r -> FilterChip(review == r, { review = r }, { Text(r.label) }) }
                listOf("" to "Any source", "snap" to "Snap", "import" to "Import", "auto" to "Auto", "video" to "Video").forEach { (k, l) ->
                    FilterChip(source == k, { source = k }, { Text(l) })
                }
            }
            if (sessions.isNotEmpty()) {
                Row(Modifier.horizontalScroll(rememberScrollState()).padding(horizontal = 12.dp), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    FilterChip(session == null, { session = null }, { Text("All sessions") })
                    sessions.take(20).forEach { s ->
                        FilterChip(session == s.id, { session = s.id }, { Text("#${s.id} ${s.mode.replace('_', ' ')}") })
                    }
                }
            }
            if (rows.isEmpty()) {
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text("No plates yet", style = MaterialTheme.typography.titleMedium)
                        Caption("Snap on the Scan tab, turn on auto-record, or import images.")
                    }
                }
            } else LazyColumn(Modifier.fillMaxSize()) {
                items(rows, key = { it.plateId }) { r -> PlateRowItem(c, r) { onOpen(r.captureId) }; HorizontalDivider() }
                item { Spacer(Modifier.height(24.dp)) }
            }
        }
    }
    if (confirmWipe) AlertDialog(
        onDismissRequest = { confirmWipe = false },
        title = { Text("Delete all records?") },
        text = { Text("Deletes every capture, plate, correction and image stored by Axios LPR. This cannot be undone.") },
        confirmButton = { TextButton(onClick = { confirmWipe = false; scope.launch { c.repo.deleteAll(); BitmapCache.clear() } }) { Text("Delete all") } },
        dismissButton = { TextButton(onClick = { confirmWipe = false }) { Text("Cancel") } },
    )
}

@Composable
private fun PlateRowItem(c: AppContainer, r: PlateRow, onClick: () -> Unit) {
    Row(Modifier.fillMaxWidth().clickable(onClick = onClick).padding(horizontal = 12.dp, vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        FileImage(c.repo.file(r.rawCropPath).path, Modifier.width(96.dp).height(40.dp), maxSide = 300)
        Spacer(Modifier.width(10.dp))
        Column(Modifier.weight(1f)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                if (r.correctedText != null && r.correctedText != r.predictedText) {
                    PlateChip(r.correctedText)
                    Text(r.predictedText, fontFamily = FontFamily.Monospace, textDecoration = TextDecoration.LineThrough,
                        color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodySmall)
                } else PlateChip(r.predictedText)
                if (r.correctedText != null) Icon(Icons.Filled.Check, "Reviewed", tint = AxiosColors.Stable)
            }
            Caption(DateFormat.getDateTimeInstance(DateFormat.SHORT, DateFormat.SHORT).format(Date(r.timestamp)) +
                " · ${r.source} · ${r.ocrModel?.let { "rec_$it" } ?: "-"}" + (r.makeModel?.let { " · $it" } ?: ""))
        }
        ConfBadge(r.predictedConf)
    }
}

fun share(context: Context, f: File, mime: String, label: String) {
    val uri = FileProvider.getUriForFile(context, "${context.packageName}.files", f)
    context.startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
        type = mime; putExtra(Intent.EXTRA_STREAM, uri); addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }, "Share $label"))
}
