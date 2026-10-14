use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

use crate::config::{ExportArgs, ExportFormat, LoraConfig, MergeArgs, TaskType, TrainingArgs};

#[derive(Parser, Debug)]
#[command(
    name = "trainaya",
    author = "Trainaya Team",
    version,
    about = "High-performance Rust CLI for fine-tuning, merging, and exporting models for Ollaya",
    long_about = "Trainaya is a modular, high-speed CLI written in Rust for local fine-tuning (LoRA), \
                  lossless weight merging, and zero-overhead export (SafeTensors, GGUF, ONNX) \
                  targeting the Ollaya local inference runtime."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Fine-tune a base encoder/decoder model on a local classification or NLI dataset
    Train(TrainOpts),

    /// Permanently merge LoRA adapter weights with the base model into clean SafeTensors
    Merge(MergeOpts),

    /// Export and package the model for ultra-fast serving in the Ollaya runtime
    Export(ExportOpts),

    /// Register a fine-tuned or merged model directly into the local Ollaya model store
    Register(RegisterOpts),

    /// Inspect system hardware, SIMD acceleration, and runtime capabilities
    Doctor,

    /// Benchmark models on the 13 public decision benchmarks against the Ollaya runtime
    Bench(BenchOpts),

    /// Curate, auto-label, and synthesize training datasets using teacher LLMs or local CLIs
    Curate(CurateOpts),
}

#[derive(Args, Debug)]
pub struct TrainOpts {
    /// Path to Trainfile configuration (e.g. 'Trainfile', '.Trainfile')
    #[arg(short = 'f', long = "file")]
    pub file: Option<PathBuf>,

    /// Hugging Face model repository identifier or local directory
    #[arg(long, default_value = "answerdotai/ModernBERT-base")]
    pub base: String,

    /// Path to dataset file (JSONL or CSV)
    #[arg(short, long)]
    pub data: Option<PathBuf>,

    /// Task type: 'classification' or 'nli'
    #[arg(long, default_value = "classification")]
    pub task: TaskType,

    /// Number of training epochs
    #[arg(long)]
    pub epochs: Option<usize>,

    /// Training batch size
    #[arg(long)]
    pub batch_size: Option<usize>,

    /// Gradient accumulation steps
    #[arg(long)]
    pub grad_accum: Option<usize>,

    /// Learning rate
    #[arg(long)]
    pub lr: Option<f64>,

    /// Warmup ratio for LR schedule (default: 0.1)
    #[arg(long)]
    pub warmup_ratio: Option<f32>,

    /// Maximum gradient L2 norm for clipping (default: 1.0)
    #[arg(long)]
    pub max_grad_norm: Option<f32>,

    /// Learning rate scheduler: 'cosine', 'linear', or 'constant'
    #[arg(long)]
    pub lr_scheduler: Option<String>,

    /// Enable 4-bit QLoRA quantisation
    #[arg(long, default_value_t = false)]
    pub qlora: bool,

    /// Enable mixed-precision BF16 training on supported GPUs
    #[arg(long, default_value_t = false)]
    pub bf16: bool,

    /// Number of top transformer blocks to inject LoRA into for GGUF models (0 = all)
    #[arg(long)]
    pub lora_layers: Option<usize>,

    /// Directory where the trained LoRA adapter and tokenizer will be saved
    #[arg(short, long)]
    pub output_dir: Option<PathBuf>,

    /// Maximum token sequence length
    #[arg(long)]
    pub max_length: Option<usize>,

    /// LoRA rank (r)
    #[arg(long)]
    pub lora_r: Option<usize>,

    /// LoRA alpha scaling factor
    #[arg(long)]
    pub lora_alpha: Option<f32>,

    /// LoRA dropout probability
    #[arg(long)]
    pub lora_dropout: Option<f32>,

    /// Specific module names to apply LoRA to (comma-separated, or default projections)
    #[arg(long, value_delimiter = ',')]
    pub target_modules: Option<Vec<String>>,

    /// Optional path to dedicated evaluation dataset
    #[arg(long)]
    pub eval_data: Option<PathBuf>,

    /// Fraction of training data to use for evaluation if --eval-data is not provided
    #[arg(long)]
    pub eval_split: Option<f32>,

    /// Weight decay for AdamW optimizer
    #[arg(long)]
    pub weight_decay: Option<f32>,

    /// Compute device: 'auto', 'cpu', 'cuda', 'metal', 'rocm', or 'vulkan'
    #[arg(long)]
    pub device: Option<String>,

    /// Random seed for data shuffling
    #[arg(long)]
    pub seed: Option<u64>,

    /// Optional model family identifier override (e.g. 'winnow-v1', 'kev-pointer-v1', 'nli-pairs-v1')
    #[arg(long)]
    pub family: Option<String>,

    /// Force merging adapter weights directly into base model
    #[arg(long, conflicts_with = "no_merge")]
    pub merge: bool,

    /// Force keeping adapter unmerged as a standalone sidecar (e.g. adapter.gguf or adapter safetensors)
    #[arg(long, conflicts_with = "merge")]
    pub no_merge: bool,

    /// Quantization format when merging model (e.g. 'Q8_0', 'Q4_K_M', 'int8')
    #[arg(long)]
    pub quant: Option<String>,

    /// Auto-register model in Ollaya under this tag/name (alias: -t, --tag, --register)
    #[arg(short = 't', long = "tag", alias = "register")]
    pub tag: Option<String>,

    /// Custom Ollaya models directory (defaults to system service if writable, or ~/.ollaya/models)
    #[arg(long)]
    pub models_dir: Option<PathBuf>,
}

impl TrainOpts {
    pub fn resolve_training_args(self) -> anyhow::Result<(TrainingArgs, Option<String>, Option<PathBuf>)> {
        use crate::trainfile::Trainfile;
        use std::path::Path;

        // 1. Try explicit file or locate default Trainfile
        let trainfile_path = if let Some(ref f) = self.file {
            Some(f.clone())
        } else if self.data.is_none() {
            Trainfile::find_default(Path::new("."))
        } else {
            None
        };

        if let Some(tf_path) = trainfile_path {
            println!("  Loading training recipe from: {}", tf_path.display());
            let tf = Trainfile::read(&tf_path)?;
            let mut args = tf.to_training_args();
            let mut register_name = tf.register;

            // Apply CLI overrides if user specified them
            if let Some(data) = self.data {
                args.data_path = data;
            }
            if self.base != "answerdotai/ModernBERT-base" {
                args.base_model = self.base;
            }
            if let Some(family) = self.family {
                args.family = Some(family);
            }
            if let Some(epochs) = self.epochs {
                args.epochs = epochs;
            }
            if let Some(bs) = self.batch_size {
                args.batch_size = bs;
            }
            if let Some(ga) = self.grad_accum {
                args.grad_accum_steps = ga.max(1);
            }
            if let Some(lr) = self.lr {
                args.lr = lr;
            }
            if let Some(wr) = self.warmup_ratio {
                args.warmup_ratio = wr;
            }
            if let Some(mgn) = self.max_grad_norm {
                args.max_grad_norm = mgn;
            }
            if let Some(sched) = self.lr_scheduler {
                args.lr_scheduler = sched;
            }
            if self.bf16 {
                args.bf16 = true;
            }
            if let Some(ll) = self.lora_layers {
                args.lora_layers = ll;
            }
            if let Some(dev) = self.device {
                args.device = dev;
            }
            if let Some(max_len) = self.max_length {
                args.max_length = max_len;
            }
            if let Some(r) = self.lora_r {
                args.lora.r = r;
            }
            if let Some(alpha) = self.lora_alpha {
                args.lora.alpha = alpha;
            }
            if let Some(dropout) = self.lora_dropout {
                args.lora.dropout = dropout;
            }
            if let Some(modules) = self.target_modules {
                args.lora.target_modules = modules;
            }
            if let Some(out_dir) = self.output_dir {
                args.output_dir = out_dir;
            }
            if self.merge {
                args.merge_mode = crate::config::MergeMode::Merge;
            } else if self.no_merge {
                args.merge_mode = crate::config::MergeMode::NoMerge;
            }
            if let Some(q) = self.quant {
                args.quant = Some(q);
            }
            if let Some(t) = self.tag {
                register_name = Some(t);
            }

            let mut models_dir = tf.models_dir;
            if let Some(md) = self.models_dir {
                models_dir = Some(md);
            }

            if let Some(ref reg) = register_name {
                let target_dir = crate::registry::get_ollaya_models_dir(models_dir.as_deref())?;
                let resolved_tag = crate::registry::resolve_dynamic_tag(reg, &target_dir);
                let tag_safe = resolved_tag.replace(':', "_");
                let out_str = args.output_dir.to_string_lossy();
                if out_str.contains("{tag}") {
                    args.output_dir = PathBuf::from(out_str.replace("{tag}", &tag_safe));
                }
                register_name = Some(resolved_tag);
            }

            return Ok((args, register_name, models_dir));
        }

        // 2. Otherwise require --data
        let data_path = match self.data {
            Some(d) => d,
            None => {
                anyhow::bail!(
                    "No dataset or Trainfile found.\n\
                     Please either pass a Trainfile with '-f Trainfile' (or create a .Trainfile in the directory),\n\
                     or pass a dataset file directly with '--data <PATH>'."
                );
            }
        };

        let mut lora = LoraConfig {
            r: self.lora_r.unwrap_or(16),
            alpha: self.lora_alpha.unwrap_or(32.0),
            dropout: self.lora_dropout.unwrap_or(0.05),
            ..Default::default()
        };

        if let Some(modules) = self.target_modules {
            if !modules.is_empty() {
                lora.target_modules = modules;
            }
        }

        let merge_mode = if self.merge {
            crate::config::MergeMode::Merge
        } else if self.no_merge {
            crate::config::MergeMode::NoMerge
        } else {
            crate::config::MergeMode::Auto
        };

        let args = TrainingArgs {
            base_model: self.base,
            family: self.family,
            data_path,
            eval_path: self.eval_data,
            eval_split: self.eval_split.unwrap_or(0.1),
            output_dir: self.output_dir.unwrap_or_else(|| PathBuf::from("./output_adapter")),
            task: self.task,
            epochs: self.epochs.unwrap_or(3),
            batch_size: self.batch_size.unwrap_or(8),
            grad_accum_steps: self.grad_accum.unwrap_or(1).max(1),
            lr: self.lr.unwrap_or(5e-5),
            warmup_ratio: self.warmup_ratio.unwrap_or(0.1),
            max_grad_norm: self.max_grad_norm.unwrap_or(1.0),
            lr_scheduler: self.lr_scheduler.unwrap_or_else(|| "cosine".to_string()),
            max_length: self.max_length.unwrap_or(128),
            weight_decay: self.weight_decay.unwrap_or(0.01),
            seed: self.seed.unwrap_or(42),
            qlora: self.qlora,
            bf16: self.bf16,
            lora_layers: self.lora_layers.unwrap_or(4),
            device: self.device.unwrap_or_else(|| "auto".to_string()),
            lora,
            merge_mode,
            quant: self.quant,
        };

        let mut register_name = self.tag;
        let mut final_args = args;
        if let Some(ref reg) = register_name {
            let target_dir = crate::registry::get_ollaya_models_dir(self.models_dir.as_deref())?;
            let resolved_tag = crate::registry::resolve_dynamic_tag(reg, &target_dir);
            let tag_safe = resolved_tag.replace(':', "_");
            let out_str = final_args.output_dir.to_string_lossy();
            if out_str.contains("{tag}") {
                final_args.output_dir = PathBuf::from(out_str.replace("{tag}", &tag_safe));
            }
            register_name = Some(resolved_tag);
        }

        Ok((final_args, register_name, self.models_dir))
    }
}

impl From<TrainOpts> for TrainingArgs {
    fn from(opts: TrainOpts) -> Self {
        opts.resolve_training_args()
            .expect("Failed to resolve training args")
            .0
    }
}

#[derive(Args, Debug)]
pub struct MergeOpts {
    /// Hugging Face model repository identifier or local directory
    #[arg(long)]
    pub base: String,

    /// Path to LoRA adapter directory containing 'adapter_model.safetensors'
    #[arg(long)]
    pub lora: PathBuf,

    /// Directory where the full merged model will be saved
    #[arg(short, long, default_value = "./output_merged")]
    pub output_dir: PathBuf,

    /// Compute device: 'cpu' (recommended for merge) or 'cuda'
    #[arg(long, default_value = "cpu")]
    pub device: String,
}

impl From<MergeOpts> for MergeArgs {
    fn from(opts: MergeOpts) -> Self {
        Self {
            base_model: opts.base,
            lora_path: opts.lora,
            output_dir: opts.output_dir,
            device: opts.device,
        }
    }
}

#[derive(Args, Debug)]
pub struct ExportOpts {
    /// Path to merged model directory containing 'model.safetensors'
    #[arg(short, long)]
    pub model: PathBuf,

    /// Target export format: 'safetensors', 'gguf', or 'onnx'
    #[arg(short, long, default_value = "safetensors")]
    pub format: ExportFormat,

    /// Optional quantization format (e.g. 'int8', 'q8_0', 'q4_k_m')
    #[arg(short, long)]
    pub quantize: Option<String>,

    /// Destination output directory
    #[arg(short, long, default_value = "./output_export")]
    pub output_dir: PathBuf,
}

impl From<ExportOpts> for ExportArgs {
    fn from(opts: ExportOpts) -> Self {
        Self {
            model_path: opts.model,
            format: opts.format,
            quantize: opts.quantize,
            output_dir: opts.output_dir,
        }
    }
}

#[derive(Args, Debug)]
pub struct RegisterOpts {
    /// Path to model file (model.safetensors or model.gguf) or directory
    #[arg(short, long)]
    pub model: PathBuf,

    /// Target Ollaya model name (e.g. 'laya:custom', 'winnow:custom', 'my-classifier')
    #[arg(short, long)]
    pub name: String,

    /// Base template to inherit layers from (e.g. 'laya:en', 'winnow:e4b')
    #[arg(long)]
    pub template: Option<String>,

    /// Path to custom tokenizer.json (if registering SafeTensors)
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,

    /// Custom path to Ollaya models store directory (defaults to ~/.ollaya/models)
    #[arg(long)]
    pub models_dir: Option<PathBuf>,
}

impl From<RegisterOpts> for crate::registry::RegisterArgs {
    fn from(opts: RegisterOpts) -> Self {
        Self {
            model_path: opts.model,
            name: opts.name,
            base_template: opts.template,
            tokenizer_path: opts.tokenizer,
            models_dir: opts.models_dir,
        }
    }
}

#[derive(Args, Debug, Clone)]
pub struct BenchOpts {
    /// Path to Benchfile configuration (e.g. 'Benchfile', '.Benchfile')
    #[arg(short = 'f', long = "file")]
    pub file: Option<PathBuf>,

    /// Model name or tag registered in Ollaya to benchmark (e.g. 'winnow:custom', 'winnow:v1')
    #[arg(short, long)]
    pub model: Option<String>,

    /// Optional baseline model name for side-by-side A/B comparison (e.g. 'winnow:e4b')
    #[arg(short, long)]
    pub baseline: Option<String>,

    /// Path to benchmark data directory (containing <subset>/all.jsonl) or a single .jsonl file
    #[arg(short, long)]
    pub data: Option<PathBuf>,

    /// Subsets to evaluate (comma-separated, e.g. 'boolq,multinli,paws'). Defaults to all 13 subsets.
    #[arg(long, value_delimiter = ',')]
    pub subsets: Option<Vec<String>>,

    /// Limit the number of records to evaluate per subset (useful for fast smoke-testing)
    #[arg(long)]
    pub limit: Option<usize>,

    /// Ollaya server URL hosting the /v1/systemone endpoint
    #[arg(long)]
    pub url: Option<String>,

    /// Output JSON report file path
    #[arg(short, long)]
    pub out: Option<PathBuf>,

    /// Minimum macro accuracy threshold required (e.g. 0.75)
    #[arg(long)]
    pub min_accuracy: Option<f64>,
}

impl BenchOpts {
    pub fn resolve_bench_args(
        self,
    ) -> anyhow::Result<(
        String,
        Option<String>,
        Option<PathBuf>,
        Option<Vec<String>>,
        Option<usize>,
        String,
        Option<PathBuf>,
        Option<f64>,
    )> {
        let mut bf = if let Some(path) = &self.file {
            crate::benchfile::Benchfile::from_file(path)?
        } else if std::path::Path::new("Benchfile").exists() {
            crate::benchfile::Benchfile::from_file(std::path::Path::new("Benchfile"))?
        } else if std::path::Path::new(".Benchfile").exists() {
            crate::benchfile::Benchfile::from_file(std::path::Path::new(".Benchfile"))?
        } else {
            crate::benchfile::Benchfile::default()
        };

        // Command-line overrides take precedence
        if let Some(m) = self.model {
            bf.model = m;
        }
        if let Some(b) = self.baseline {
            bf.baseline = Some(b);
        }
        if let Some(d) = self.data {
            bf.data = Some(d);
        }
        if let Some(s) = self.subsets {
            bf.subsets = Some(s);
        }
        if let Some(l) = self.limit {
            bf.limit = Some(l);
        }
        if let Some(u) = self.url {
            bf.url = u;
        }
        if let Some(o) = self.out {
            bf.output = Some(o);
        }
        if let Some(a) = self.min_accuracy {
            bf.min_accuracy = Some(a);
        }

        if bf.model.is_empty() {
            anyhow::bail!(
                "Target model is required for benchmark. Provide --model <MODEL> or define 'MODEL <name>' in Benchfile."
            );
        }

        bf.resolve_placeholders(None);

        Ok((
            bf.model,
            bf.baseline,
            bf.data,
            bf.subsets,
            bf.limit,
            bf.url,
            bf.output,
            bf.min_accuracy,
        ))
    }
}

#[derive(Args, Debug, Clone)]
pub struct CurateOpts {
    /// Path to Curatefile configuration (e.g. 'Curatefile', '.Curatefile')
    #[arg(short = 'f', long = "file")]
    pub file: Option<PathBuf>,

    /// Teacher engine: 'agy' (default), 'claude', 'ollama:<model>'
    #[arg(long)]
    pub teacher: Option<String>,

    /// Curation mode: 'label' (default), 'synth', 'augment'
    #[arg(short, long)]
    pub mode: Option<String>,

    /// Target dataset format: 'decision' (default), 'classification', 'nli'
    #[arg(long)]
    pub format: Option<String>,

    /// Path to raw input data file (JSONL or plain text)
    #[arg(short, long)]
    pub input: Option<PathBuf>,

    /// Path to policy criteria or question schema JSON
    #[arg(short, long)]
    pub schema: Option<PathBuf>,

    /// Path to write curated dataset
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Number of samples to generate in synth mode
    #[arg(long)]
    pub count: Option<usize>,

    /// Batch size for teacher queries
    #[arg(long)]
    pub batch_size: Option<usize>,

    /// Minimum confidence threshold (0.0 to 1.0)
    #[arg(long)]
    pub min_confidence: Option<f64>,
}

impl CurateOpts {
    pub fn resolve_curate_args(self) -> anyhow::Result<crate::curatefile::Curatefile> {
        let mut cf = if let Some(path) = &self.file {
            crate::curatefile::Curatefile::from_file(path)?
        } else if std::path::Path::new("Curatefile").exists() {
            crate::curatefile::Curatefile::from_file(std::path::Path::new("Curatefile"))?
        } else if std::path::Path::new(".Curatefile").exists() {
            crate::curatefile::Curatefile::from_file(std::path::Path::new(".Curatefile"))?
        } else {
            crate::curatefile::Curatefile::default()
        };

        if let Some(t) = self.teacher {
            cf.teacher = t;
        }
        if let Some(m) = self.mode {
            cf.mode = match m.to_lowercase().as_str() {
                "label" | "annotate" => crate::curatefile::CurateMode::Label,
                "synth" | "synthetic" => crate::curatefile::CurateMode::Synth,
                "augment" => crate::curatefile::CurateMode::Augment,
                other => anyhow::bail!("Unknown mode '{}'. Expected: label, synth, augment", other),
            };
        }
        if let Some(f) = self.format {
            cf.format = match f.to_lowercase().as_str() {
                "decision" => crate::curatefile::TargetDatasetFormat::Decision,
                "classification" => crate::curatefile::TargetDatasetFormat::Classification,
                "nli" => crate::curatefile::TargetDatasetFormat::Nli,
                other => anyhow::bail!("Unknown format '{}'. Expected: decision, classification, nli", other),
            };
        }
        if let Some(i) = self.input {
            cf.input = Some(i);
        }
        if let Some(s) = self.schema {
            cf.schema = Some(s);
        }
        if let Some(o) = self.output {
            cf.output = o;
        }
        if let Some(c) = self.count {
            cf.count = c;
        }
        if let Some(b) = self.batch_size {
            cf.batch_size = b;
        }
        if let Some(mc) = self.min_confidence {
            cf.min_confidence = mc;
        }

        Ok(cf)
    }
}

