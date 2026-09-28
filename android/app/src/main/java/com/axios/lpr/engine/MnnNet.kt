package com.axios.lpr.engine

import java.io.Closeable

/** MNN forward types (MNNForwardType) and precision modes exposed in Settings. */
enum class MnnBackend(val code: Int, val label: String) {
    CPU(0, "CPU"), OPENCL(3, "OpenCL"), VULKAN(7, "Vulkan"), AUTO(4, "Auto");

    companion object {
        fun labelOf(code: Int) = entries.firstOrNull { it.code == code }?.label ?: "type $code"
    }
}

enum class MnnPrecision(val code: Int) { NORMAL(0), HIGH(1), LOW(2) }

/**
 * One MNN model with its own session. Not thread-safe: [run] is synchronised per instance.
 * Output order follows [outputNames].
 */
class MnnNet private constructor(
    private var handle: Long,
    val name: String,
) : Closeable {
    val inputShape: IntArray = nativeInputShape(handle)
    val outputNames: List<String> = nativeOutputNames(handle).toList()
    val outputShapes: List<IntArray> = outputNames.indices.map { nativeOutputShape(handle, it) }

    /** Backend the session really runs on (MNN silently falls back to CPU). */
    val activeBackend: Int = nativeBackend(handle)

    val inputSize: Int get() = inputShape.fold(1) { a, b -> a * b }

    @Synchronized
    fun run(input: FloatArray): List<FloatArray> {
        check(handle != 0L) { "MnnNet closed: $name" }
        return nativeRun(handle, input).toList()
    }

    /** Convenience for single-output models. */
    fun runSingle(input: FloatArray): FloatArray = run(input)[0]

    fun output(name: String, outputs: List<FloatArray>): FloatArray? =
        outputNames.indexOfFirst { it.contains(name) }.takeIf { it >= 0 }?.let { outputs[it] }

    @Synchronized
    override fun close() {
        if (handle != 0L) {
            nativeRelease(handle)
            handle = 0L
        }
    }

    companion object {
        init {
            System.loadLibrary("axioslpr")
        }

        fun load(
            model: ByteArray,
            name: String,
            backend: MnnBackend = MnnBackend.CPU,
            threads: Int = 4,
            precision: MnnPrecision = MnnPrecision.NORMAL,
        ): MnnNet {
            val h = nativeCreate(model, backend.code, threads, precision.code)
            require(h != 0L) { "MNN failed to load $name" }
            return MnnNet(h, name)
        }

        @JvmStatic private external fun nativeCreate(model: ByteArray, forward: Int, threads: Int, precision: Int): Long
        @JvmStatic private external fun nativeRelease(h: Long)
        @JvmStatic private external fun nativeInputShape(h: Long): IntArray
        @JvmStatic private external fun nativeOutputNames(h: Long): Array<String>
        @JvmStatic private external fun nativeOutputShape(h: Long, i: Int): IntArray
        @JvmStatic private external fun nativeBackend(h: Long): Int
        @JvmStatic private external fun nativeRun(h: Long, input: FloatArray): Array<FloatArray>
    }
}
