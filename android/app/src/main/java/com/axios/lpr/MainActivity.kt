package com.axios.lpr

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Science
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.axios.lpr.ui.capture.CaptureScreen
import com.axios.lpr.ui.common.Caption
import com.axios.lpr.ui.history.HistoryScreen
import com.axios.lpr.ui.lab.LabScreen
import com.axios.lpr.ui.scan.ScanScreen
import com.axios.lpr.ui.settings.SettingsScreen
import com.axios.lpr.ui.tables.TablesScreen
import com.axios.lpr.ui.theme.AxiosTheme
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    private val container get() = (application as AxiosApp).container

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        if (savedInstanceState == null) handleShare(intent)
        setContent { AxiosTheme { App(container) } }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleShare(intent)
    }

    /** Images shared from Gallery / Files go through the same import pipeline. */
    private fun handleShare(intent: Intent?) {
        intent ?: return
        val uris: List<Uri> = when (intent.action) {
            Intent.ACTION_SEND -> listOfNotNull(
                if (Build.VERSION.SDK_INT >= 33) intent.getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)
                else @Suppress("DEPRECATION") intent.getParcelableExtra(Intent.EXTRA_STREAM),
            )
            Intent.ACTION_SEND_MULTIPLE ->
                if (Build.VERSION.SDK_INT >= 33) intent.getParcelableArrayListExtra(Intent.EXTRA_STREAM, Uri::class.java).orEmpty()
                else @Suppress("DEPRECATION") intent.getParcelableArrayListExtra<Uri>(Intent.EXTRA_STREAM).orEmpty()
            else -> emptyList()
        }
        if (uris.isNotEmpty()) container.importer.import(uris, "share")
    }
}

private data class Tab(val route: String, val label: String, val icon: ImageVector)

private val tabs = listOf(
    Tab("scan", "Scan", Icons.Filled.CameraAlt),
    Tab("history", "History", Icons.Filled.History),
    Tab("lab", "Lab", Icons.Filled.Science),
    Tab("settings", "Settings", Icons.Filled.Settings),
)

@Composable
private fun App(c: AppContainer) {
    val nav = rememberNavController()
    val snackbar = remember { SnackbarHostState() }
    val scope = rememberCoroutineScope()
    val entry by nav.currentBackStackEntryAsState()
    val route = entry?.destination?.route
    val onError: (String) -> Unit = { msg -> scope.launch { snackbar.showSnackbar(msg) } }

    Scaffold(
        snackbarHost = { SnackbarHost(snackbar) },
        bottomBar = {
            if (route == null || tabs.any { route.startsWith(it.route) }) NavigationBar {
                tabs.forEach { t ->
                    NavigationBarItem(
                        selected = route?.startsWith(t.route) == true,
                        onClick = { nav.navigateTab(t.route) },
                        icon = { Icon(t.icon, t.label) },
                        label = { Text(t.label) },
                    )
                }
            }
        },
    ) { pad ->
        NavHost(nav, startDestination = "scan", Modifier.fillMaxSize().padding(bottom = pad.calculateBottomPadding())) {
            composable("scan") { ScanScreen(c, onOpenCapture = { nav.navigate("capture/$it") }, onError = onError) }
            composable("history?session={session}", arguments = listOf(navArgument("session") { type = NavType.LongType; defaultValue = -1L })) { e ->
                val s = e.arguments?.getLong("session")?.takeIf { it > 0 }
                HistoryScreen(c, s, onOpen = { nav.navigate("capture/$it") })
            }
            composable("lab") { LabScreen(c, onTables = { nav.navigate("tables") }) }
            composable("settings") { SettingsScreen(c) }
            composable("tables") { TablesScreen(c, onBack = { nav.popBackStack() }) }
            composable("capture/{id}", arguments = listOf(navArgument("id") { type = NavType.LongType })) { e ->
                CaptureScreen(c, e.arguments!!.getLong("id"), onBack = { nav.popBackStack() })
            }
        }
    }
    ImportProgress(c, nav, onError)
}

private fun NavHostController.navigateTab(route: String) = navigate(route) {
    popUpTo(graph.findStartDestination().id) { saveState = true }
    launchSingleTop = true
    restoreState = true
}

@Composable
private fun ImportProgress(c: AppContainer, nav: NavHostController, onError: (String) -> Unit) {
    val st by c.importer.state.collectAsState()
    LaunchedEffect(st.finishedAt) {
        if (st.finishedAt == 0L) return@LaunchedEffect
        st.errors.forEach(onError)
        when {
            st.captureIds.size == 1 -> nav.navigate("capture/${st.captureIds[0]}")
            st.captureIds.size > 1 -> nav.navigate("history?session=${st.sessionId ?: -1}") { launchSingleTop = true }
        }
        c.importer.dismiss()
    }
    if (st.running) AlertDialog(
        onDismissRequest = {},
        title = { Text("Reading plates") },
        text = {
            Column {
                LinearProgressIndicator(progress = { if (st.total == 0) 0f else st.done / st.total.toFloat() }, Modifier.padding(vertical = 8.dp))
                Text("${st.done}/${st.total}")
                st.current?.let { Caption(it) }
                if (st.errors.isNotEmpty()) Caption("${st.errors.size} failed")
            }
        },
        confirmButton = { TextButton(onClick = {}, enabled = false) { Text("Working…") } },
    )
}
