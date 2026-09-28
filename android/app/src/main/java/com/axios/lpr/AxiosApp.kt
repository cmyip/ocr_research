package com.axios.lpr

import android.app.Application
import com.axios.lpr.data.AppSettings
import com.axios.lpr.data.CaptureRepository
import com.axios.lpr.data.Exporter
import com.axios.lpr.data.LprDatabase
import com.axios.lpr.data.SettingsRepository
import com.axios.lpr.engine.AndroidModelStore
import com.axios.lpr.engine.ModelManager
import com.axios.lpr.engine.PlatePipeline
import com.axios.lpr.ui.ImportController
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

/** Manual DI: one instance of each service for the whole app. */
class AppContainer(val app: Application) {
    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    val modelStore = AndroidModelStore(app)
    val models = ModelManager(modelStore)
    val pipeline = PlatePipeline(models)
    val db = LprDatabase.create(app)
    val repo = CaptureRepository(app, db)
    val exporter = Exporter(app, repo)
    val settingsRepo = SettingsRepository(app)
    val settings: StateFlow<AppSettings> = settingsRepo.settings.stateIn(scope, SharingStarted.Eagerly, AppSettings())
    val importer = ImportController(this)

    /** Startup warm-up progress (done, total); null when ready. */
    val prepare = MutableStateFlow<Pair<Int, Int>?>(0 to 1)

    init {
        scope.launch(Dispatchers.IO) {
            modelStore.cleanLegacyCopies()
            // Load the default live models once so the first frame isn't slow.
            val warm = settings.value.pipeline.activeModels(capture = false).toList()
            warm.forEachIndexed { i, id ->
                prepare.value = i to warm.size
                runCatching { if (modelStore.catalog[id].kind == "onnx") models.detector(id) else models.mnn(id) }
            }
            prepare.value = null
        }
        // Toggling a model off unloads it; runtime changes reload lazily.
        scope.launch {
            settings.map { it.pipeline }.distinctUntilChanged().collect { cfg -> pipeline.applySettings(cfg) }
        }
    }
}

class AxiosApp : Application() {
    lateinit var container: AppContainer
        private set

    override fun onCreate() {
        super.onCreate()
        container = AppContainer(this)
    }
}
