//! Speed and accuracy of this model against other plate-reading APIs over one image set.
//!
//! Providers: `local` (the pipeline in this process), any HTTP API described in a providers file
//! (see providers.example.toml), and "recorded" reads - extra columns of the labels CSV, for an
//! API that cannot be called on demand, such as the reads a camera already made.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use base64::Engine;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::image::RgbImage;
use crate::pipeline::{Pipeline, PipelineConfig};

#[derive(Clone, Debug, clap::Args)]
pub struct BenchArgs {
    #[command(flatten)]
    pub pipeline: PipelineConfig,
    /// Directory of .jpg/.jpeg/.png images.
    #[arg(long)]
    pub images: PathBuf,
    /// CSV with a header: `file,plate` plus, optionally, one column per recorded API's read.
    /// Without it the true plate is the file name up to the first `_` or `.` (BRL4104_2.jpg).
    #[arg(long)]
    pub labels: Option<PathBuf>,
    /// The file names are not plates and there are no labels: report agreement between providers only.
    #[arg(long)]
    pub no_truth: bool,
    /// TOML file describing the HTTP APIs to call.
    #[arg(long)]
    pub providers: Option<PathBuf>,
    /// Run only these providers (comma separated names; `local` is the in-process model).
    #[arg(long, value_delimiter = ',')]
    pub only: Vec<String>,
    /// Leave the in-process model out.
    #[arg(long)]
    pub no_local: bool,
    /// Timed calls per image.
    #[arg(long, default_value_t = 3)]
    pub runs: usize,
    /// Untimed calls per provider before measuring (model warm-up, connection setup).
    #[arg(long, default_value_t = 2)]
    pub warmup: usize,
    /// Write one row per image and provider.
    #[arg(long)]
    pub csv: Option<PathBuf>,
    /// Write the summary as JSON.
    #[arg(long)]
    pub json: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct ProvidersFile {
    #[serde(default)]
    provider: Vec<HttpProvider>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HttpProvider {
    name: String,
    url: String,
    /// How the image is sent: "raw" body, "multipart" file field, or "json" with a base64 string.
    #[serde(default = "default_request")]
    request: String,
    /// The multipart field or JSON key holding the image.
    #[serde(default = "default_field")]
    field: String,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    /// Extra multipart fields or JSON keys.
    #[serde(default)]
    form: BTreeMap<String, String>,
    /// JSON pointer to the plate in the reply, e.g. "/results/0/plate".
    plate: String,
    confidence: Option<String>,
    /// JSON pointer to the time the API says it spent, in milliseconds.
    server_ms: Option<String>,
    #[serde(default = "default_timeout")]
    timeout_s: u64,
}

fn default_request() -> String {
    "raw".into()
}
fn default_field() -> String {
    "image".into()
}
fn default_timeout() -> u64 {
    30
}

struct Sample {
    file: String,
    bytes: Vec<u8>,
    truth: Option<String>,
    recorded: BTreeMap<String, String>,
}

#[derive(Clone, Default)]
struct Call {
    plate: Option<String>,
    confidence: Option<f64>,
    latency_ms: f64,
    server_ms: Option<f64>,
    stages: Option<[f64; 6]>,
    error: Option<String>,
}

const STAGES: [&str; 6] = ["decode", "det_prep", "det_infer", "rectify", "region", "ocr"];

/// Per image: the read (from the first timed call) and every latency sample.
struct ImageOutcome {
    plate: Option<String>,
    confidence: Option<f64>,
    error: Option<String>,
    latencies: Vec<f64>,
}

struct ProviderRun {
    name: String,
    detail: String,
    outcomes: Vec<ImageOutcome>,
    server_ms: Vec<f64>,
    stages: Vec<[f64; 6]>,
}

/// Plates compare without case, spaces or punctuation.
pub fn normalise(plate: &str) -> String {
    plate.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_uppercase).collect()
}

pub fn levenshtein(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut diag = row[0];
        row[0] = i;
        for j in 1..=b.len() {
            let next = (diag + usize::from(a[i - 1] != b[j - 1])).min(row[j] + 1).min(row[j - 1] + 1);
            diag = row[j];
            row[j] = next;
        }
    }
    row[b.len()]
}

/// Nearest-rank percentile of an ascending slice.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// `${NAME}` is replaced from the environment, so tokens stay out of the providers file.
fn expand_env(text: &str) -> Result<String> {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find("${") {
        let end = rest[start..].find('}').with_context(|| format!("unclosed ${{ in {text:?}"))? + start;
        let name = &rest[start + 2..end];
        out.push_str(&rest[..start]);
        out.push_str(&std::env::var(name).with_context(|| format!("environment variable {name} is not set"))?);
        rest = &rest[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

fn load_samples(args: &BenchArgs) -> Result<(Vec<Sample>, Vec<String>)> {
    let mut labels: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut columns: Vec<String> = Vec::new();
    if let Some(path) = &args.labels {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let header: Vec<String> = lines.next().context("labels file is empty")?.split(',').map(|c| c.trim().trim_start_matches('\u{feff}').to_string()).collect();
        if header.len() < 2 || !header[0].eq_ignore_ascii_case("file") || !header[1].eq_ignore_ascii_case("plate") {
            bail!("{} must start with the header file,plate", path.display());
        }
        columns = header[2..].to_vec();
        for line in lines {
            let cells: Vec<String> = line.split(',').map(|c| c.trim().trim_matches('"').to_string()).collect();
            labels.insert(cells[0].clone(), cells[1..].to_vec());
        }
    }

    let mut files: Vec<PathBuf> = std::fs::read_dir(&args.images)
        .with_context(|| format!("reading {}", args.images.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()).is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png")))
        .collect();
    files.sort();

    let mut samples = Vec::new();
    for path in files {
        let file = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        let row = labels.get(&file);
        if args.labels.is_some() && row.is_none() {
            continue; // with a labels file, only labelled images are benchmarked
        }
        let truth = if args.no_truth {
            None
        } else if let Some(row) = row {
            row.first().map(|p| normalise(p)).filter(|p| !p.is_empty())
        } else {
            Some(normalise(file.split(['_', '.']).next().unwrap_or_default()))
        };
        let recorded = columns.iter().enumerate().filter_map(|(i, c)| Some((c.clone(), row?.get(i + 1)?.clone()))).collect();
        samples.push(Sample { bytes: std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?, file, truth, recorded });
    }
    if samples.is_empty() {
        bail!("no images to benchmark in {}", args.images.display());
    }
    Ok((samples, columns))
}

fn call_local(pipeline: &mut Pipeline, bytes: &[u8]) -> Call {
    let t = Instant::now();
    let outcome = RgbImage::decode(bytes).and_then(|img| {
        let decode_ms = t.elapsed().as_secs_f64() * 1e3;
        pipeline.process(&img).map(|r| (decode_ms, r))
    });
    let latency_ms = t.elapsed().as_secs_f64() * 1e3;
    match outcome {
        Ok((decode_ms, r)) => {
            let tm = r.timings;
            Call {
                plate: r.plates.first().map(|p| p.plate.clone()),
                confidence: r.plates.first().map(|p| p.confidence as f64),
                latency_ms,
                server_ms: None,
                stages: Some([decode_ms, tm.det_prep_ms, tm.det_infer_ms, tm.rectify_ms, tm.region_ms, tm.ocr_ms]),
                error: None,
            }
        }
        Err(e) => Call { latency_ms, error: Some(format!("{e:#}")), ..Call::default() },
    }
}

fn mime(file: &str) -> &'static str {
    if file.to_ascii_lowercase().ends_with(".png") { "image/png" } else { "image/jpeg" }
}

async fn call_http(client: &reqwest::Client, p: &HttpProvider, sample: &Sample) -> Call {
    let mut req = client.post(&p.url).timeout(Duration::from_secs(p.timeout_s));
    for (k, v) in &p.headers {
        req = req.header(k, v);
    }
    req = match p.request.as_str() {
        "multipart" => {
            let part = reqwest::multipart::Part::bytes(sample.bytes.clone()).file_name(sample.file.clone()).mime_str(mime(&sample.file)).expect("static mime");
            let mut form = reqwest::multipart::Form::new().part(p.field.clone(), part);
            for (k, v) in &p.form {
                form = form.text(k.clone(), v.clone());
            }
            req.multipart(form)
        }
        "json" => {
            let mut body = serde_json::Map::new();
            body.insert(p.field.clone(), json!(base64::engine::general_purpose::STANDARD.encode(&sample.bytes)));
            for (k, v) in &p.form {
                body.insert(k.clone(), json!(v));
            }
            req.json(&body)
        }
        _ => req.header("content-type", mime(&sample.file)).body(sample.bytes.clone()),
    };

    let t = Instant::now();
    let reply = async {
        let resp = req.send().await?;
        let status = resp.status();
        let text = resp.text().await?;
        Ok::<_, reqwest::Error>((status, text))
    }
    .await;
    let latency_ms = t.elapsed().as_secs_f64() * 1e3;
    let (status, text) = match reply {
        Ok(r) => r,
        Err(e) => return Call { latency_ms, error: Some(e.to_string()), ..Call::default() },
    };
    if !status.is_success() {
        let detail: String = text.chars().take(200).collect();
        return Call { latency_ms, error: Some(format!("HTTP {}: {detail}", status.as_u16())), ..Call::default() };
    }
    let Ok(v) = serde_json::from_str::<Value>(&text) else {
        return Call { latency_ms, error: Some("reply is not JSON".into()), ..Call::default() };
    };
    let number = |pointer: &Option<String>| {
        let found = v.pointer(pointer.as_deref()?)?;
        found.as_f64().or_else(|| found.as_str()?.trim().parse().ok())
    };
    Call {
        // A missing or null plate is the API saying it read nothing, not a failure.
        plate: v.pointer(&p.plate).and_then(Value::as_str).map(str::to_string).filter(|s| !s.is_empty()),
        confidence: number(&p.confidence),
        latency_ms,
        server_ms: number(&p.server_ms),
        stages: None,
        error: None,
    }
}

pub async fn run(args: BenchArgs) -> Result<()> {
    let (samples, recorded_columns) = load_samples(&args)?;
    let wanted = |name: &str| args.only.is_empty() || args.only.iter().any(|o| o.eq_ignore_ascii_case(name));

    let mut http: Vec<HttpProvider> = Vec::new();
    if let Some(path) = &args.providers {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let file: ProvidersFile = toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        for mut p in file.provider.into_iter().filter(|p| wanted(&p.name)) {
            if !matches!(p.request.as_str(), "raw" | "multipart" | "json") {
                bail!("provider {}: request must be raw, multipart or json", p.name);
            }
            let ctx = format!("provider {}", p.name);
            p.url = expand_env(&p.url).context(ctx.clone())?;
            for v in p.headers.values_mut().chain(p.form.values_mut()) {
                *v = expand_env(v).context(ctx.clone())?;
            }
            http.push(p);
        }
    }

    let has_truth = samples.iter().any(|s| s.truth.is_some());
    let total_calls = args.runs.max(1);
    eprintln!("{} images, {} timed call(s) each, {} warm-up call(s) per provider", samples.len(), total_calls, args.warmup);
    let mut runs: Vec<ProviderRun> = Vec::new();

    if !args.no_local && wanted("local") {
        let t = Instant::now();
        let mut pipeline = Pipeline::load(&args.pipeline)?;
        eprintln!("local: models loaded in {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
        for i in 0..args.warmup {
            call_local(&mut pipeline, &samples[i % samples.len()].bytes);
        }
        let mut run = ProviderRun { name: "local".into(), detail: "in process".into(), outcomes: Vec::new(), server_ms: Vec::new(), stages: Vec::new() };
        for s in &samples {
            let calls: Vec<Call> = (0..total_calls).map(|_| call_local(&mut pipeline, &s.bytes)).collect();
            run.stages.extend(calls.iter().filter_map(|c| c.stages));
            run.outcomes.push(outcome(calls));
        }
        runs.push(run);
    }

    let client = reqwest::Client::new();
    for p in &http {
        eprintln!("{}: calling {}", p.name, p.url);
        for i in 0..args.warmup {
            call_http(&client, p, &samples[i % samples.len()]).await;
        }
        let mut run = ProviderRun { name: p.name.clone(), detail: p.url.clone(), outcomes: Vec::new(), server_ms: Vec::new(), stages: Vec::new() };
        for s in &samples {
            let mut calls = Vec::with_capacity(total_calls);
            for _ in 0..total_calls {
                calls.push(call_http(&client, p, s).await);
            }
            run.server_ms.extend(calls.iter().filter_map(|c| c.server_ms));
            run.outcomes.push(outcome(calls));
        }
        runs.push(run);
    }

    for column in recorded_columns.iter().filter(|c| wanted(c)) {
        let outcomes = samples
            .iter()
            .map(|s| ImageOutcome { plate: s.recorded.get(column).cloned().filter(|p| !p.is_empty()), confidence: None, error: None, latencies: Vec::new() })
            .collect();
        runs.push(ProviderRun { name: column.clone(), detail: "recorded in the labels file".into(), outcomes, server_ms: Vec::new(), stages: Vec::new() });
    }
    if runs.is_empty() {
        bail!("no providers to run");
    }

    let summary = report(&samples, &runs, has_truth);
    if let Some(path) = &args.csv {
        write_csv(path, &samples, &runs)?;
        eprintln!("per-image rows written to {}", path.display());
    }
    if let Some(path) = &args.json {
        std::fs::write(path, serde_json::to_string_pretty(&summary)?)?;
        eprintln!("summary written to {}", path.display());
    }
    Ok(())
}

fn outcome(calls: Vec<Call>) -> ImageOutcome {
    let first = calls.first().cloned().unwrap_or_default();
    ImageOutcome {
        plate: first.plate,
        confidence: first.confidence,
        // Failed calls are counted as errors, not timed: a timeout would skew the latency.
        latencies: calls.iter().filter(|c| c.error.is_none()).map(|c| c.latency_ms).collect(),
        error: calls.into_iter().find_map(|c| c.error),
    }
}

fn fmt_ms(v: f64) -> String {
    if v.is_nan() { "-".into() } else { format!("{v:.1}") }
}

fn report(samples: &[Sample], runs: &[ProviderRun], has_truth: bool) -> Value {
    let n = samples.len();
    let reference = &runs[0];
    let mut rows = Vec::new();
    let mut summary = Vec::new();

    for run in runs {
        let read = run.outcomes.iter().filter(|o| o.plate.is_some()).count();
        let errors = run.outcomes.iter().filter(|o| o.error.is_some()).count();
        let (mut exact, mut labelled, mut edits, mut truth_chars, mut agree) = (0usize, 0usize, 0usize, 0usize, 0usize);
        for ((s, o), r) in samples.iter().zip(&run.outcomes).zip(&reference.outcomes) {
            let got = o.plate.as_deref().map(normalise).unwrap_or_default();
            if let Some(truth) = &s.truth {
                labelled += 1;
                exact += usize::from(&got == truth);
                edits += levenshtein(&got, truth).min(truth.chars().count().max(got.chars().count()));
                truth_chars += truth.chars().count();
            }
            agree += usize::from(got == r.plate.as_deref().map(normalise).unwrap_or_default());
        }
        let mut lat: Vec<f64> = run.outcomes.iter().flat_map(|o| o.latencies.iter().copied()).collect();
        lat.sort_by(f64::total_cmp);
        let mean = if lat.is_empty() { f64::NAN } else { lat.iter().sum::<f64>() / lat.len() as f64 };
        let pct = |part: usize, whole: usize| if whole == 0 { f64::NAN } else { 100.0 * part as f64 / whole as f64 };
        let exact_pct = pct(exact, labelled);
        let char_pct = if truth_chars == 0 { f64::NAN } else { (100.0 * (1.0 - edits as f64 / truth_chars as f64)).max(0.0) };
        let agree_pct = pct(agree, n);
        let show_pct = |v: f64| if v.is_nan() { "-".to_string() } else { format!("{v:.1}%") };

        let mut row = vec![run.name.clone(), n.to_string(), read.to_string(), errors.to_string()];
        if has_truth {
            row.push(format!("{exact}/{labelled} {}", show_pct(exact_pct)));
            row.push(show_pct(char_pct));
        }
        row.push(show_pct(agree_pct));
        row.extend([50.0, 90.0, 99.0].iter().map(|&p| fmt_ms(percentile(&lat, p))));
        row.push(fmt_ms(mean));
        rows.push(row);

        let nan_null = |v: f64| if v.is_nan() { Value::Null } else { json!((v * 100.0).round() / 100.0) };
        let mut server = run.server_ms.clone();
        server.sort_by(f64::total_cmp);
        summary.push(json!({
            "provider": run.name, "detail": run.detail, "images": n, "read": read, "errors": errors,
            "labelled": labelled, "exact": exact, "exact_pct": nan_null(exact_pct), "char_accuracy_pct": nan_null(char_pct),
            "agree_with_first_pct": nan_null(agree_pct),
            "latency_ms": { "samples": lat.len(), "p50": nan_null(percentile(&lat, 50.0)), "p90": nan_null(percentile(&lat, 90.0)),
                            "p99": nan_null(percentile(&lat, 99.0)), "mean": nan_null(mean),
                            "min": nan_null(lat.first().copied().unwrap_or(f64::NAN)), "max": nan_null(lat.last().copied().unwrap_or(f64::NAN)) },
            "server_reported_ms_p50": nan_null(percentile(&server, 50.0)),
        }));
    }

    let mut header = vec!["provider", "images", "read", "errors"];
    if has_truth {
        header.extend(["exact", "char acc"]);
    }
    let agree_title = format!("agree w/ {}", reference.name);
    header.push(&agree_title);
    header.extend(["p50 ms", "p90 ms", "p99 ms", "mean ms"]);
    println!();
    print_table(&header, &rows);

    for run in runs {
        if !run.stages.is_empty() {
            let medians: Vec<String> = (0..STAGES.len())
                .map(|i| {
                    let mut v: Vec<f64> = run.stages.iter().map(|s| s[i]).collect();
                    v.sort_by(f64::total_cmp);
                    format!("{} {:.1}", STAGES[i], percentile(&v, 50.0))
                })
                .collect();
            println!("\n{} stage medians (ms): {}", run.name, medians.join(", "));
        }
        if !run.server_ms.is_empty() {
            let mut v = run.server_ms.clone();
            v.sort_by(f64::total_cmp);
            println!("\n{}: the API reports {:.1} ms median for its own processing; the rest of its latency is transfer", run.name, percentile(&v, 50.0));
        }
        if let Some(e) = run.outcomes.iter().find_map(|o| o.error.as_deref()) {
            println!("\n{}: first error: {e}", run.name);
        }
    }

    // Where the providers differ from the truth, or from each other when there is none.
    let mut differences = Vec::new();
    for (i, s) in samples.iter().enumerate() {
        let reads: Vec<String> = runs.iter().map(|r| r.outcomes[i].plate.as_deref().map(normalise).unwrap_or_default()).collect();
        let off = match &s.truth {
            Some(truth) => reads.iter().any(|r| r != truth),
            None => reads.iter().any(|r| r != &reads[0]),
        };
        if off {
            let shown: Vec<String> = runs.iter().zip(&reads).map(|(r, read)| format!("{}={}", r.name, if read.is_empty() { "(none)" } else { read })).collect();
            differences.push(format!("  {}  {}{}", s.file, s.truth.as_ref().map(|t| format!("truth={t}  ")).unwrap_or_default(), shown.join("  ")));
        }
    }
    if !differences.is_empty() {
        println!("\n{} image(s) where a provider is wrong or the providers disagree:", differences.len());
        differences.iter().take(40).for_each(|d| println!("{d}"));
        if differences.len() > 40 {
            println!("  ... {} more (see --csv)", differences.len() - 40);
        }
    }
    json!({ "images": n, "providers": summary })
}

fn print_table(header: &[&str], rows: &[Vec<String>]) {
    let widths: Vec<usize> = (0..header.len()).map(|c| rows.iter().map(|r| r[c].chars().count()).chain([header[c].chars().count()]).max().unwrap_or(0)).collect();
    let line = |cells: Vec<&str>| {
        let padded: Vec<String> = cells.iter().enumerate().map(|(c, v)| if c == 0 { format!("{v:<w$}", w = widths[c]) } else { format!("{v:>w$}", w = widths[c]) }).collect();
        println!("{}", padded.join("  "));
    };
    line(header.to_vec());
    let rules: Vec<String> = widths.iter().map(|&w| "-".repeat(w)).collect();
    line(rules.iter().map(String::as_str).collect());
    for r in rows {
        line(r.iter().map(String::as_str).collect());
    }
}

fn write_csv(path: &Path, samples: &[Sample], runs: &[ProviderRun]) -> Result<()> {
    let mut out = String::from("file,truth,provider,plate,correct,confidence,latency_ms_median,error\n");
    for run in runs {
        for (s, o) in samples.iter().zip(&run.outcomes) {
            let got = o.plate.as_deref().map(normalise).unwrap_or_default();
            let mut lat = o.latencies.clone();
            lat.sort_by(f64::total_cmp);
            let correct = s.truth.as_ref().map(|t| (t == &got).to_string()).unwrap_or_default();
            let median = if lat.is_empty() { String::new() } else { format!("{:.2}", percentile(&lat, 50.0)) };
            let error = o.error.as_deref().unwrap_or_default().replace([',', '\n', '\r'], " ");
            out.push_str(&format!(
                "{},{},{},{},{},{},{},{}\n",
                s.file, s.truth.as_deref().unwrap_or_default(), run.name, got, correct,
                o.confidence.map(|c| format!("{c:.4}")).unwrap_or_default(), median, error
            ));
        }
    }
    std::fs::write(path, out).with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plates_compare_without_case_or_separators() {
        assert_eq!(normalise("brl 4104"), "BRL4104");
        assert_eq!(normalise("W-XY.1234"), "WXY1234");
    }

    #[test]
    fn edit_distance() {
        assert_eq!(levenshtein("BRL4104", "BRL4104"), 0);
        assert_eq!(levenshtein("BRL4004", "BRL4104"), 1);
        assert_eq!(levenshtein("BRL404", "BRL4104"), 1);
        assert_eq!(levenshtein("", "BRL"), 3);
    }

    #[test]
    fn percentiles_use_nearest_rank() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(percentile(&v, 50.0), 5.0);
        assert_eq!(percentile(&v, 90.0), 9.0);
        assert_eq!(percentile(&v, 99.0), 10.0);
        assert!(percentile(&[], 50.0).is_nan());
    }

    #[test]
    fn environment_variables_expand_and_missing_ones_fail() {
        std::env::set_var("LPR_TEST_TOKEN", "abc");
        assert_eq!(expand_env("Token ${LPR_TEST_TOKEN}!").unwrap(), "Token abc!");
        assert!(expand_env("${LPR_TEST_SURELY_UNSET}").is_err());
        assert_eq!(expand_env("plain").unwrap(), "plain");
    }
}
