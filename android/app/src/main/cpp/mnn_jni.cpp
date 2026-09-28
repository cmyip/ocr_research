// Thin JNI bridge over the MNN C++ Interpreter API.
// One handle = one Interpreter + one Session. Callers serialise access per handle.
#include <jni.h>
#include <android/log.h>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#define LOG_TAG "AxiosMNN"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {
struct Net {
    std::shared_ptr<MNN::Interpreter> interp;
    MNN::Session* session = nullptr;
    MNN::Tensor* input = nullptr;
    std::vector<std::string> outNames;
    std::vector<MNN::Tensor*> outs;
};

Net* H(jlong h) { return reinterpret_cast<Net*>(h); }

jintArray toIntArray(JNIEnv* env, const std::vector<int>& v) {
    jintArray a = env->NewIntArray((jsize) v.size());
    env->SetIntArrayRegion(a, 0, (jsize) v.size(), v.data());
    return a;
}
}  // namespace

extern "C" {

// Model bytes come straight from the (uncompressed) APK asset; MNN copies the buffer.
JNIEXPORT jlong JNICALL
Java_com_axios_lpr_engine_MnnNet_nativeCreate(JNIEnv* env, jclass, jbyteArray jmodel, jint forward,
                                              jint threads, jint precision) {
    jbyte* bytes = env->GetByteArrayElements(jmodel, nullptr);
    const jsize size = env->GetArrayLength(jmodel);
    std::shared_ptr<MNN::Interpreter> interp(MNN::Interpreter::createFromBuffer(bytes, (size_t) size),
                                             MNN::Interpreter::destroy);
    env->ReleaseByteArrayElements(jmodel, bytes, JNI_ABORT);
    if (!interp) {
        LOGE("createFromBuffer failed");
        return 0;
    }
    MNN::ScheduleConfig cfg;
    cfg.type = static_cast<MNNForwardType>(forward);
    cfg.backupType = MNN_FORWARD_CPU;  // GPU backends fall back to CPU if unavailable
    cfg.numThread = threads > 0 ? threads : 4;
    MNN::BackendConfig bc;
    bc.precision = static_cast<MNN::BackendConfig::PrecisionMode>(precision);
    cfg.backendConfig = &bc;
    MNN::Session* session = interp->createSession(cfg);
    if (!session) {
        LOGE("createSession failed");
        return 0;
    }
    auto* n = new Net();
    n->interp = interp;
    n->session = session;
    n->input = interp->getSessionInput(session, nullptr);
    for (auto& kv : interp->getSessionOutputAll(session)) {
        n->outNames.push_back(kv.first);
        n->outs.push_back(kv.second);
    }
    return reinterpret_cast<jlong>(n);
}

JNIEXPORT void JNICALL
Java_com_axios_lpr_engine_MnnNet_nativeRelease(JNIEnv*, jclass, jlong h) {
    Net* n = H(h);
    if (!n) return;
    n->interp->releaseSession(n->session);
    delete n;
}

JNIEXPORT jintArray JNICALL
Java_com_axios_lpr_engine_MnnNet_nativeInputShape(JNIEnv* env, jclass, jlong h) {
    return toIntArray(env, H(h)->input->shape());
}

JNIEXPORT jobjectArray JNICALL
Java_com_axios_lpr_engine_MnnNet_nativeOutputNames(JNIEnv* env, jclass, jlong h) {
    Net* n = H(h);
    jobjectArray a = env->NewObjectArray((jsize) n->outNames.size(),
                                         env->FindClass("java/lang/String"), nullptr);
    for (size_t i = 0; i < n->outNames.size(); ++i) {
        env->SetObjectArrayElement(a, (jsize) i, env->NewStringUTF(n->outNames[i].c_str()));
    }
    return a;
}

JNIEXPORT jintArray JNICALL
Java_com_axios_lpr_engine_MnnNet_nativeOutputShape(JNIEnv* env, jclass, jlong h, jint i) {
    return toIntArray(env, H(h)->outs[i]->shape());
}

// Returns the forward types actually used by the session (index 0 = primary backend).
JNIEXPORT jint JNICALL
Java_com_axios_lpr_engine_MnnNet_nativeBackend(JNIEnv*, jclass, jlong h) {
    Net* n = H(h);
    int types[2] = {-1, -1};
    if (!n->interp->getSessionInfo(n->session, MNN::Interpreter::BACKENDS, types)) return -1;
    return types[0];
}

// input: NCHW float32. Returns one float[] per output, in nativeOutputNames order.
JNIEXPORT jobjectArray JNICALL
Java_com_axios_lpr_engine_MnnNet_nativeRun(JNIEnv* env, jclass, jlong h, jfloatArray jin) {
    Net* n = H(h);
    MNN::Tensor hostIn(n->input, MNN::Tensor::CAFFE);
    const jsize len = env->GetArrayLength(jin);
    if ((size_t) len != (size_t) hostIn.elementSize()) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                      ("input size mismatch: got " + std::to_string(len) + ", expected " +
                       std::to_string(hostIn.elementSize())).c_str());
        return nullptr;
    }
    env->GetFloatArrayRegion(jin, 0, len, hostIn.host<float>());
    n->input->copyFromHostTensor(&hostIn);
    if (n->interp->runSession(n->session) != MNN::NO_ERROR) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "runSession failed");
        return nullptr;
    }
    jobjectArray result = env->NewObjectArray((jsize) n->outs.size(), env->FindClass("[F"), nullptr);
    for (size_t i = 0; i < n->outs.size(); ++i) {
        MNN::Tensor hostOut(n->outs[i], MNN::Tensor::CAFFE);
        n->outs[i]->copyToHostTensor(&hostOut);
        const int count = hostOut.elementSize();
        jfloatArray fa = env->NewFloatArray(count);
        env->SetFloatArrayRegion(fa, 0, count, hostOut.host<float>());
        env->SetObjectArrayElement(result, (jsize) i, fa);
        env->DeleteLocalRef(fa);
    }
    return result;
}

}  // extern "C"
