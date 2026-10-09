//! Safe wrapper over the C shim in `native/mnn_shim.cpp`: one MNN interpreter + session per `Net`.
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::Path;

use anyhow::{bail, Context, Result};

extern "C" {
    fn lpr_mnn_create(path: *const c_char, threads: c_int) -> *mut c_void;
    fn lpr_mnn_destroy(h: *mut c_void);
    fn lpr_mnn_input_shape(h: *mut c_void, dims: *mut c_int, cap: c_int) -> c_int;
    fn lpr_mnn_output_count(h: *mut c_void) -> c_int;
    fn lpr_mnn_output_name(h: *mut c_void, index: c_int, buf: *mut c_char, cap: c_int) -> c_int;
    fn lpr_mnn_output_shape(h: *mut c_void, index: c_int, dims: *mut c_int, cap: c_int) -> c_int;
    fn lpr_mnn_run(h: *mut c_void, input: *const f32, len: usize) -> c_int;
    fn lpr_mnn_output(h: *mut c_void, index: c_int, out: *mut f32, cap: usize) -> c_int;
}

pub struct Net {
    handle: *mut c_void,
    name: String,
    pub input_shape: Vec<usize>,
    /// Shape of the first output.
    pub output_shape: Vec<usize>,
    /// Every output's name and shape, in the order `run_all` returns them.
    outputs: Vec<(String, Vec<usize>)>,
}

// A session is not safe to share, but it can move between threads; `run` takes `&mut self`.
unsafe impl Send for Net {}

impl Net {
    pub fn load(path: &Path, threads: usize) -> Result<Self> {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if !path.is_file() {
            bail!("model file {} not found", path.display());
        }
        let c_path = CString::new(path.to_string_lossy().as_bytes()).context("model path")?;
        let handle = unsafe { lpr_mnn_create(c_path.as_ptr(), threads as c_int) };
        if handle.is_null() {
            bail!("MNN could not load {}", path.display());
        }
        let mut net = Net { handle, name, input_shape: Vec::new(), output_shape: Vec::new(), outputs: Vec::new() };
        let count = unsafe { lpr_mnn_output_count(handle) };
        if count < 1 {
            bail!("{} has no outputs", net.name);
        }
        net.input_shape = shape(|dims, cap| unsafe { lpr_mnn_input_shape(handle, dims, cap) });
        for i in 0..count {
            let mut buf = [0 as c_char; 128];
            unsafe { lpr_mnn_output_name(handle, i, buf.as_mut_ptr(), buf.len() as c_int) };
            let output = unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned();
            net.outputs.push((output, shape(|dims, cap| unsafe { lpr_mnn_output_shape(handle, i, dims, cap) })));
        }
        net.output_shape = net.outputs[0].1.clone();
        Ok(net)
    }

    /// Index of the output whose name contains `part` (rec_15 has `predictions_mmr` and `predictions_pose`).
    pub fn output_index(&self, part: &str) -> Option<usize> {
        self.outputs.iter().position(|(name, _)| name.contains(part))
    }

    /// Runs the model on an NCHW float32 input and returns its first output, flattened.
    pub fn run(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        self.forward(input)?;
        self.output(0)
    }

    /// Runs the model and returns every output, flattened, in `output_index` order.
    pub fn run_all(&mut self, input: &[f32]) -> Result<Vec<Vec<f32>>> {
        self.forward(input)?;
        (0..self.outputs.len()).map(|i| self.output(i)).collect()
    }

    fn forward(&mut self, input: &[f32]) -> Result<()> {
        match unsafe { lpr_mnn_run(self.handle, input.as_ptr(), input.len()) } {
            0 => Ok(()),
            -1 => bail!("{}: input has {} values, model wants {:?}", self.name, input.len(), self.input_shape),
            _ => bail!("{}: MNN session failed", self.name),
        }
    }

    fn output(&mut self, index: usize) -> Result<Vec<f32>> {
        let shape = &self.outputs[index].1;
        let mut out = vec![0f32; shape.iter().product()];
        let n = unsafe { lpr_mnn_output(self.handle, index as c_int, out.as_mut_ptr(), out.len()) };
        if n < 0 {
            bail!("{}: output {index} is larger than its declared shape {shape:?}", self.name);
        }
        out.truncate(n as usize);
        Ok(out)
    }
}

impl Drop for Net {
    fn drop(&mut self) {
        unsafe { lpr_mnn_destroy(self.handle) }
    }
}

fn shape(read: impl Fn(*mut c_int, c_int) -> c_int) -> Vec<usize> {
    let mut dims = [0 as c_int; 8];
    let n = read(dims.as_mut_ptr(), dims.len() as c_int).clamp(0, dims.len() as c_int) as usize;
    dims[..n].iter().map(|&d| d.max(0) as usize).collect()
}
