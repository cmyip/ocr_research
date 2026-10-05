// Thin C bridge over the MNN C++ Interpreter API (same shape as android/.../mnn_jni.cpp).
// One handle = one Interpreter + one Session. Callers serialise access per handle.
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
#ifndef _WIN32
#include <unistd.h>
#endif

namespace {
struct Net {
    std::shared_ptr<MNN::Interpreter> interp;
    MNN::Session* session = nullptr;
    MNN::Tensor* input = nullptr;
    std::vector<MNN::Tensor*> outs;
};

// MNN prints a CPU-feature banner with printf when its runtime first starts. Stdout is where the
// CLI writes JSON, so the banner is sent to stderr instead.
struct StdoutToStderr {
#ifndef _WIN32
    int saved;
    StdoutToStderr() {
        fflush(stdout);
        saved = dup(STDOUT_FILENO);
        dup2(STDERR_FILENO, STDOUT_FILENO);
    }
    ~StdoutToStderr() {
        fflush(stdout);
        if (saved >= 0) {
            dup2(saved, STDOUT_FILENO);
            close(saved);
        }
    }
#endif
};

int copyShape(const std::vector<int>& shape, int* dims, int cap) {
    const int n = (int) shape.size();
    for (int i = 0; i < n && i < cap; ++i) dims[i] = shape[i];
    return n;
}
}  // namespace

extern "C" {

void* lpr_mnn_create(const char* path, int threads) {
    StdoutToStderr quiet;
    std::shared_ptr<MNN::Interpreter> interp(MNN::Interpreter::createFromFile(path), MNN::Interpreter::destroy);
    if (!interp) return nullptr;
    MNN::ScheduleConfig cfg;
    cfg.type = MNN_FORWARD_CPU;
    cfg.numThread = threads > 0 ? threads : 4;
    MNN::Session* session = interp->createSession(cfg);
    if (!session) return nullptr;
    auto* n = new Net();
    n->interp = interp;
    n->session = session;
    n->input = interp->getSessionInput(session, nullptr);
    for (auto& kv : interp->getSessionOutputAll(session)) n->outs.push_back(kv.second);
    return n;
}

void lpr_mnn_destroy(void* h) {
    Net* n = static_cast<Net*>(h);
    if (!n) return;
    n->interp->releaseSession(n->session);
    delete n;
}

int lpr_mnn_input_shape(void* h, int* dims, int cap) {
    return copyShape(static_cast<Net*>(h)->input->shape(), dims, cap);
}

int lpr_mnn_output_count(void* h) { return (int) static_cast<Net*>(h)->outs.size(); }

int lpr_mnn_output_shape(void* h, int index, int* dims, int cap) {
    return copyShape(static_cast<Net*>(h)->outs[index]->shape(), dims, cap);
}

// input: NCHW float32. Returns 0, -1 on a size mismatch, -2 if the session failed.
int lpr_mnn_run(void* h, const float* input, size_t len) {
    Net* n = static_cast<Net*>(h);
    MNN::Tensor hostIn(n->input, MNN::Tensor::CAFFE);
    if (len != (size_t) hostIn.elementSize()) return -1;
    std::memcpy(hostIn.host<float>(), input, len * sizeof(float));
    n->input->copyFromHostTensor(&hostIn);
    return n->interp->runSession(n->session) == MNN::NO_ERROR ? 0 : -2;
}

// Copies output `index` of the last run; returns the element count, or -1 if `cap` is too small.
int lpr_mnn_output(void* h, int index, float* out, size_t cap) {
    Net* n = static_cast<Net*>(h);
    MNN::Tensor hostOut(n->outs[index], MNN::Tensor::CAFFE);
    n->outs[index]->copyToHostTensor(&hostOut);
    const size_t count = (size_t) hostOut.elementSize();
    if (count > cap) return -1;
    std::memcpy(out, hostOut.host<float>(), count * sizeof(float));
    return (int) count;
}

}  // extern "C"
