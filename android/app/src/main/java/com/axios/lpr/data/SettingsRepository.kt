package com.axios.lpr.data

import android.content.Context
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.axios.lpr.engine.LiveConfig
import com.axios.lpr.engine.PipelineConfig
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import kotlinx.serialization.json.Json

private val Context.store by preferencesDataStore("axios_settings")

data class AppSettings(val pipeline: PipelineConfig = PipelineConfig(), val live: LiveConfig = LiveConfig())

/** Pipeline + live settings (including every model toggle) persisted as JSON in DataStore. */
class SettingsRepository(private val context: Context) {
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }
    private val kPipeline = stringPreferencesKey("pipeline")
    private val kLive = stringPreferencesKey("live")

    val settings: Flow<AppSettings> = context.store.data.map { p ->
        AppSettings(
            p[kPipeline]?.let { runCatching { json.decodeFromString(PipelineConfig.serializer(), it) }.getOrNull() } ?: PipelineConfig(),
            p[kLive]?.let { runCatching { json.decodeFromString(LiveConfig.serializer(), it) }.getOrNull() } ?: LiveConfig(),
        )
    }

    suspend fun updatePipeline(f: (PipelineConfig) -> PipelineConfig) = context.store.edit { p ->
        val cur = p[kPipeline]?.let { runCatching { json.decodeFromString(PipelineConfig.serializer(), it) }.getOrNull() } ?: PipelineConfig()
        p[kPipeline] = json.encodeToString(PipelineConfig.serializer(), f(cur))
    }

    suspend fun updateLive(f: (LiveConfig) -> LiveConfig) = context.store.edit { p ->
        val cur = p[kLive]?.let { runCatching { json.decodeFromString(LiveConfig.serializer(), it) }.getOrNull() } ?: LiveConfig()
        p[kLive] = json.encodeToString(LiveConfig.serializer(), f(cur))
    }
}
