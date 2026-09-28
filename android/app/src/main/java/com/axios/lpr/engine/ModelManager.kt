package com.axios.lpr.engine

import android.util.Log
import java.io.Closeable

/**
 * Owns the loaded runtime sessions. Models load lazily on first use and are closed when
 * toggled off ([retainOnly]) or when the runtime settings change ([configure]).
 */
class ModelManager(val files: ModelFiles) : Closeable {
    private val mnn = HashMap<Int, MnnNet>()
    private val ort = HashMap<Int, OrtDetector>()
    private val loadMs = HashMap<Int, Float>()
    private val notes = HashMap<Int, String>()

    /** Check each GPU session against CPU on load and fall back when outputs disagree. */
    var validateGpu = true

    private var backend = MnnBackend.CPU
    private var threads = clampThreads(4)
    private var precision = MnnPrecision.NORMAL
    private var ortProvider = OrtProvider.CPU
    private var ortThreads = clampThreads(4)

    @Synchronized
    fun configure(cfg: PipelineConfig) {
        val mnnChanged = cfg.mnnBackend != backend || clampThreads(cfg.mnnThreads) != threads || cfg.mnnPrecision != precision
        val ortChanged = cfg.ortProvider != ortProvider || clampThreads(cfg.ortThreads) != ortThreads
        if (mnnChanged) {
            mnn.values.forEach { it.close() }; mnn.clear()
            backend = cfg.mnnBackend; threads = clampThreads(cfg.mnnThreads); precision = cfg.mnnPrecision
        }
        if (ortChanged) {
            ort.values.forEach { it.close() }; ort.clear()
            ortProvider = cfg.ortProvider; ortThreads = clampThreads(cfg.ortThreads)
        }
    }

    @Synchronized
    fun mnn(id: Int): MnnNet = mnn.getOrPut(id) {
        val t = System.nanoTime()
        val bytes = files.bytes(id)
        var net = MnnNet.load(bytes, "rec_$id", backend, threads, precision)
        notes.remove(id)
        if (net.activeBackend != MnnBackend.CPU.code && validateGpu) {
            val diff = gpuDiff(net, bytes)
            if (diff > GPU_TOLERANCE) {
                val label = MnnBackend.labelOf(net.activeBackend)
                Log.w(TAG, "rec_$id: $label output differs from CPU by $diff; using CPU")
                net.close()
                net = MnnNet.load(bytes, "rec_$id", MnnBackend.CPU, threads, precision)
                notes[id] = "$label rejected (Δ=${"%.3f".format(diff)} vs CPU)"
            } else {
                notes[id] = "${MnnBackend.labelOf(net.activeBackend)} validated (Δ=${"%.4f".format(diff)})"
            }
        } else if (backend != MnnBackend.CPU && net.activeBackend == MnnBackend.CPU.code) {
            notes[id] = "${backend.label} unavailable → CPU"
        }
        loadMs[id] = (System.nanoTime() - t) / 1e6f
        Log.i(TAG, "loaded rec_$id on ${MnnBackend.labelOf(net.activeBackend)} in ${loadMs[id]} ms ${notes[id] ?: ""}")
        net
    }

    /** Max abs difference between [net] and a CPU session on a fixed pseudo-random input. */
    private fun gpuDiff(net: MnnNet, bytes: ByteArray): Float {
        val rnd = java.util.Random(1234)
        val x = FloatArray(net.inputSize) { rnd.nextFloat() }
        MnnNet.load(bytes, net.name, MnnBackend.CPU, threads, MnnPrecision.HIGH).use { cpu ->
            val a = net.run(x.copyOf())
            val b = cpu.run(x.copyOf())
            var d = 0f
            for (i in a.indices) {
                // Compare after per-output normalisation so logits and probabilities are treated alike
                val scale = maxOf(1f, b[i].maxOf { kotlin.math.abs(it) })
                for (k in a[i].indices) d = maxOf(d, kotlin.math.abs(a[i][k] - b[i][k]) / scale)
            }
            return d
        }
    }

    fun note(id: Int): String? = notes[id]

    @Synchronized
    fun detector(id: Int): OrtDetector = ort.getOrPut(id) {
        val t = System.nanoTime()
        OrtDetector(files.bytes(id), ortProvider, ortThreads).also { loadMs[id] = (System.nanoTime() - t) / 1e6f }
    }

    /** Close every loaded model not in [keep]. */
    @Synchronized
    fun retainOnly(keep: Set<Int>) {
        mnn.keys.filter { it !in keep }.forEach { mnn.remove(it)?.close() }
        ort.keys.filter { it !in keep }.forEach { ort.remove(it)?.close() }
    }

    @Synchronized
    fun loaded(): Map<Int, String> =
        mnn.mapValues { "MNN ${MnnBackend.labelOf(it.value.activeBackend)}" } + ort.mapValues { "ORT ${it.value.providerLabel}" }

    @Synchronized
    fun isLoaded(id: Int) = id in mnn || id in ort

    fun loadTimeMs(id: Int): Float? = loadMs[id]

    @Synchronized
    override fun close() {
        mnn.values.forEach { it.close() }; mnn.clear()
        ort.values.forEach { it.close() }; ort.clear()
    }

    companion object {
        private const val TAG = "AxiosModels"
        const val GPU_TOLERANCE = 0.05f

        /** More threads than cores makes inference much slower (spin-waiting), so cap at the core count. */
        fun clampThreads(n: Int) = n.coerceIn(1, Runtime.getRuntime().availableProcessors().coerceAtLeast(1))
    }
}
