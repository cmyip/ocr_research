#!/usr/bin/env bash
# Builds lpr-api with the detector on Intel OpenVINO, on Ubuntu 26.04 x86_64.
#
# The ort crate ships no prebuilt ONNX Runtime with the OpenVINO provider, so this script:
#   1. installs the build tools with apt (sudo),
#   2. downloads Intel's OpenVINO runtime archive and checks its SHA-256,
#   3. builds ONNX Runtime from source against it (the long step: 20-60 minutes),
#   4. builds lpr-api with the `openvino` cargo feature, linked to that ONNX Runtime,
#   5. collects the binary and its shared libraries in dist/lpr-api-openvino/,
#   6. times the samples on both backends, if the weights are next to the crate.
#
# Usage:   scripts/build-openvino-ubuntu.sh
# Result:  dist/lpr-api-openvino/lpr-api   (a wrapper that sets the library path)
#          dist/lpr-api-openvino/lpr-api serve --weights ../weights --detector-backend openvino
#
# Re-running skips whatever is already done. Everything it downloads or builds lives in
# .openvino-build/ (about 6 GB); delete that directory to start again.
#
# Settings (environment variables):
#   ORT_VERSION    ONNX Runtime tag. It must match the ort crate in Cargo.toml (1.28.x).
#   OPENVINO_URL   OpenVINO archive (.tgz). ONNX Runtime 1.28 needs OpenVINO 2026.0 or newer.
#   WORK           where downloads and builds go (default: <crate>/.openvino-build)
#   JOBS           parallel compile jobs (default: all cores; lower it if the build runs out of memory)
#   SKIP_APT=1     do not install packages (when they are already there, or without sudo)
set -euo pipefail

ORT_VERSION="${ORT_VERSION:-v1.28.3}"
OPENVINO_URL="${OPENVINO_URL:-https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.3.1/linux/openvino_toolkit_ubuntu26_2026.3.1.22476.56d9685302d_x86_64.tgz}"

CRATE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${WORK:-$CRATE/.openvino-build}"
JOBS="${JOBS:-$(nproc)}"
DIST="$CRATE/dist/lpr-api-openvino"

step() { printf '\n==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- checks
[ "$(uname -s)" = "Linux" ] || die "this script is for Linux (Ubuntu 26.04)"
[ "$(uname -m)" = "x86_64" ] || die "the default OpenVINO archive is x86_64 only; this machine is $(uname -m)"
if [ -r /etc/os-release ]; then
    . /etc/os-release
    if [ "${ID:-}" != "ubuntu" ] || [ "${VERSION_ID:-}" != "26.04" ]; then
        echo "warning: written for Ubuntu 26.04, this is ${PRETTY_NAME:-unknown}. Set OPENVINO_URL to the archive for this release if the build fails."
    fi
fi
if ! grep -qw avx2 /proc/cpuinfo; then
    echo "warning: this CPU has no AVX2; OpenVINO will run, but expect little gain."
fi
mkdir -p "$WORK"

# ---------------------------------------------------------------- 1. packages
if [ "${SKIP_APT:-0}" != "1" ]; then
    step "Installing build tools (apt)"
    SUDO=""
    [ "$(id -u)" -eq 0 ] || SUDO="sudo"
    $SUDO apt-get update
    # libssl-dev and pkg-config: the ort crate's build script compiles a TLS client even when it
    # downloads nothing. ffmpeg is what lpr-api uses to decode RTSP streams at run time.
    $SUDO apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git curl ca-certificates python3 \
        pkg-config libssl-dev ffmpeg
    if ! command -v cargo >/dev/null 2>&1; then
        $SUDO apt-get install -y --no-install-recommends rustup
        rustup default stable
    fi
fi

for tool in cmake ninja git curl python3 cargo; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool is not installed (run without SKIP_APT=1, or install it)"
done
# ONNX Runtime needs CMake 3.28+, the ort crate Rust 1.88+.
cmake_version="$(cmake --version | awk 'NR==1 {print $3}')"
rust_version="$(rustc --version | awk '{print $2}')"
at_least() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)" = "$1" ]; }
at_least 3.28 "$cmake_version" || die "CMake $cmake_version is too old; ONNX Runtime needs 3.28 or newer"
at_least 1.88 "$rust_version" || die "Rust $rust_version is too old; needs 1.88 or newer (try: rustup update stable)"

# ---------------------------------------------------------------- 2. OpenVINO
OPENVINO_DIR="$WORK/openvino"
if [ ! -f "$OPENVINO_DIR/setupvars.sh" ]; then
    step "Downloading OpenVINO: $(basename "$OPENVINO_URL")"
    archive="$WORK/$(basename "$OPENVINO_URL")"
    curl -fL --retry 3 -o "$archive" "$OPENVINO_URL"
    # Intel publishes the checksum next to the archive; it guards against a damaged download.
    expected="$(curl -fsSL --retry 3 "$OPENVINO_URL.sha256" | awk '{print $1}')"
    actual="$(sha256sum "$archive" | awk '{print $1}')"
    [ -n "$expected" ] && [ "$expected" = "$actual" ] || die "checksum mismatch for $archive (expected ${expected:-nothing}, got $actual)"
    rm -rf "$OPENVINO_DIR.tmp" && mkdir -p "$OPENVINO_DIR.tmp"
    tar -xzf "$archive" -C "$OPENVINO_DIR.tmp" --strip-components=1
    mv "$OPENVINO_DIR.tmp" "$OPENVINO_DIR"
    rm -f "$archive"
fi
[ -f "$OPENVINO_DIR/setupvars.sh" ] || die "$OPENVINO_DIR does not look like an OpenVINO archive (no setupvars.sh)"
# setupvars.sh reads unset variables, which `set -u` would treat as errors.
set +u
# shellcheck disable=SC1091
. "$OPENVINO_DIR/setupvars.sh" >/dev/null
set -u
echo "OpenVINO at $OPENVINO_DIR"

# ---------------------------------------------------------------- 3. ONNX Runtime
ORT_SRC="$WORK/onnxruntime"
ORT_BUILD="$WORK/ort-build"
ORT_LIBS="$ORT_BUILD/Release"
if [ ! -f "$ORT_LIBS/libonnxruntime_providers_openvino.so" ]; then
    if [ ! -d "$ORT_SRC/.git" ]; then
        step "Fetching ONNX Runtime $ORT_VERSION"
        git clone --depth 1 --branch "$ORT_VERSION" https://github.com/microsoft/onnxruntime.git "$ORT_SRC"
    fi
    step "Building ONNX Runtime $ORT_VERSION with OpenVINO ($JOBS jobs; this is the long step)"
    root_flag=()
    [ "$(id -u)" -ne 0 ] || root_flag=(--allow_running_as_root)
    # --compile_no_warning_as_error: a newer GCC than ONNX Runtime was tested with raises new
    # warnings, and the build otherwise treats them as errors.
    "$ORT_SRC/build.sh" --config Release --build_dir "$ORT_BUILD" \
        --use_openvino CPU --build_shared_lib \
        --parallel "$JOBS" --cmake_generator Ninja \
        --skip_tests --skip_submodule_sync --compile_no_warning_as_error \
        "${root_flag[@]}"
fi
for lib in libonnxruntime.so libonnxruntime_providers_openvino.so libonnxruntime_providers_shared.so; do
    [ -e "$ORT_LIBS/$lib" ] || die "the ONNX Runtime build did not produce $lib in $ORT_LIBS"
done
echo "ONNX Runtime libraries in $ORT_LIBS"

# ---------------------------------------------------------------- 4. lpr-api
step "Building lpr-api with the openvino feature"
# ORT_LIB_PATH points the ort crate at this ONNX Runtime instead of its download; the OpenVINO
# provider is a separate shared library, so ONNX Runtime has to be linked dynamically.
(cd "$CRATE" && ORT_LIB_PATH="$ORT_LIBS" ORT_PREFER_DYNAMIC_LINK=1 cargo build --release --features openvino)

# ---------------------------------------------------------------- 5. bundle
step "Collecting the binary and libraries in $DIST"
rm -rf "$DIST" && mkdir -p "$DIST/bin" "$DIST/lib"
cp "$CRATE/target/release/lpr-api" "$DIST/bin/"
cp -a "$ORT_LIBS"/libonnxruntime*.so* "$DIST/lib/"
# OpenVINO finds its CPU plugin and frontends next to libopenvino.so, so the directory is kept whole.
cp -a "$OPENVINO_DIR"/runtime/lib/intel64/. "$DIST/lib/"
if [ -d "$OPENVINO_DIR/runtime/3rdparty/tbb/lib" ]; then
    cp -a "$OPENVINO_DIR"/runtime/3rdparty/tbb/lib/*.so* "$DIST/lib/"
fi
cat > "$DIST/lpr-api" <<'WRAPPER'
#!/usr/bin/env bash
# Runs lpr-api with the ONNX Runtime and OpenVINO libraries bundled beside it.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$here/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$here/bin/lpr-api" "$@"
WRAPPER
chmod +x "$DIST/lpr-api"
"$DIST/lpr-api" --version

# ---------------------------------------------------------------- 6. compare
WEIGHTS="${LPR_WEIGHTS:-$CRATE/../weights}"
SAMPLES="$CRATE/../sample_data"
if [ -f "$WEIGHTS/rec_72.bin" ] && [ -d "$SAMPLES" ]; then
    for backend in cpu openvino; do
        step "Samples with --detector-backend $backend"
        "$DIST/lpr-api" bench --weights "$WEIGHTS" --images "$SAMPLES" --runs 30 --warmup 5 --detector-backend "$backend" 2>/dev/null
    done
    echo
    echo "Compare the p50 column and det_infer in the stage medians. Both runs should read 3/3."
else
    echo "Weights not found at $WEIGHTS, so nothing was timed. Try:"
    echo "  $DIST/lpr-api bench --weights /path/to/weights --images /path/to/images --detector-backend openvino"
fi

step "Done"
echo "Run it with:  $DIST/lpr-api serve --weights $WEIGHTS --detector-backend openvino"
