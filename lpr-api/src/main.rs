//! lpr-api: the extracted ANPR models behind an HTTP API that speaks the camera provider's event
//! payload, plus a harness for comparing it with other plate-reading APIs. See README.md.
mod bench;
mod channels;
mod image;
mod mnn;
mod payload;
mod pipeline;
mod plate_format;
mod server;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde_json::json;

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the HTTP API (POST /v1/read, POST /v1/trigger).
    Serve(server::ServeArgs),
    /// Read plates from image files and print one JSON line per image.
    Read {
        #[command(flatten)]
        pipeline: pipeline::PipelineConfig,
        #[arg(required = true)]
        images: Vec<PathBuf>,
    },
    /// Compare speed and accuracy with other APIs over a set of images.
    Bench(bench::BenchArgs),
}

fn read(cfg: &pipeline::PipelineConfig, images: &[PathBuf]) -> Result<()> {
    let mut pipeline = pipeline::Pipeline::load(cfg)?;
    for path in images {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let t = Instant::now();
        let img = image::RgbImage::decode(&bytes).with_context(|| path.display().to_string())?;
        let decode_ms = t.elapsed().as_secs_f64() * 1e3;
        let result = pipeline.process(&img)?;
        println!(
            "{}",
            json!({
                "file": path.display().to_string(),
                "plate": result.plates.first().map(|p| p.plate.as_str()),
                "plates": result.plates,
                "decode_ms": decode_ms,
                "timing": result.timings,
            })
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Read { pipeline, images } => read(&pipeline, &images),
        Command::Serve(args) => runtime()?.block_on(server::serve(args)),
        Command::Bench(args) => runtime()?.block_on(bench::run(args)),
    }
}

fn runtime() -> Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_multi_thread().enable_all().build()?)
}
