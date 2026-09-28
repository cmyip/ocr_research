package com.axios.lpr.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import java.io.Closeable
import java.nio.FloatBuffer

/** ONNX Runtime session for the YOLO plate/vehicle detectors (rec_72, rec_75). */
class OrtDetector(model: ByteArray, provider: OrtProvider, threads: Int) : Closeable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    val providerLabel: String
    val size: Int
    val anchors: Int
    private val inputName: String

    init {
        var label = provider.label
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threads)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            try {
                when (provider) {
                    OrtProvider.XNNPACK -> addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
                    OrtProvider.NNAPI -> addNnapi()
                    OrtProvider.CPU -> Unit
                }
            } catch (e: Exception) {
                Log.w("AxiosOrt", "provider $provider unavailable, using CPU", e)
                label = "CPU (fallback)"
            }
        }
        session = env.createSession(model, opts)
        providerLabel = label
        inputName = session.inputNames.first()
        val inShape = (session.inputInfo.getValue(inputName).info as ai.onnxruntime.TensorInfo).shape
        size = inShape[2].toInt()
        val outShape = (session.outputInfo.values.first().info as ai.onnxruntime.TensorInfo).shape
        anchors = outShape[2].toInt()
    }

    /** x: (1,3,S,S) float. Returns the flat (1, 4+nc, N) output. */
    @Synchronized
    fun infer(x: FloatArray): FloatArray {
        OnnxTensor.createTensor(env, FloatBuffer.wrap(x), longArrayOf(1, 3, size.toLong(), size.toLong())).use { t ->
            session.run(mapOf(inputName to t)).use { r ->
                val fb = (r[0] as OnnxTensor).floatBuffer
                return FloatArray(fb.remaining()).also { fb.get(it) }
            }
        }
    }

    @Synchronized
    override fun close() = session.close()
}
