use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

use crate::config::{AdaptArgs, ExportArgs, ExportFormat, LoraConfig, MergeArgs, RuntimeTarget, TaskType, TrainingArgs};

#[derive(Parser, Debug)]
#[command(
    name = "trainaya",
    author = "Trainaya Team",
    version,
    about = "High-performance Rust CLI orchestrating uv + PyTorch for training, adapting, and exporting Ollaya & Ollama decision models",
    long_about = "Trainaya is a modular CLI written in Rust backed by an embedded uv + PyTorch engine for \
                  end-to-end decision model fine-tuning, generic multi-architecture adaptation (e.g. Shieldstral, Ministral, Gemma -> Ollaya/Ollama), \
                  temperature calibration, and zero-overhead export."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Fine-tune an encoder or decoder decision model end-to-end via uv + PyTorch
    Train(TrainOpts),

    /// Adapt a non-decision model (e.g. Shieldstral, Ministral, Gemma, Qwen, ModernBERT) into a decision model
    Adapt(AdaptOpts),

    /// Permanently merge LoRA adapter weights with the base model into clean SafeTensors or GGUF
    Merge(MergeOpts),

    /// Export and package the model for ultra-fast serving in the Ollaya runtime
    Export(ExportOpts),

    /// Register a fine-tuned, adapted, or merged model directly into local Ollaya and/or Ollama model stores
    Register(RegisterOpts),

    /// Inspect system hardware, uv + PyTorch engine status, and runtime capabilities
    Doctor,

    /// Benchmark models on the 13 public decision benchmarks against the Ollaya runtime
    Bench(BenchOpts),

    /// Curate, auto-label, and synthesize training datasets using teacher LLMs or local CLIs
    Curate(CurateOpts),

    /// Check for updates and upgrade Trainaya to the latest GitHub release
    #[command(alias = "self-update", alias = "upgrade")]
    Update(UpdateOpts),
}

#[derive(Args, Debug)]
pub struct AdaptOpts {
    /// Source model path (.gguf file, local directory, or HuggingFace repo ID like 'gemma-3-4b' / 'ministral')
    #[arg(long, alias = "model")]
    pub from: String,

    /// Target model tag to register (e.g. 'gemma3:v1', 'ministral:v1')
    #[arg(short = 't', long = "target", alias = "tag")]
    pub target: String,

    /// Target layout: 'auto', 'llm-logits-v1', or 'kev-pointer-v1'
    #[arg(long, default_value = "auto")]
    pub layout: String,

    /// Adaptation mode: 'full' (biorthogonal + Fisher ARD + enriched score head), 'head-only', or 'chained-lora'
    #[arg(long, default_value = "full")]
    pub mode: String,

    /// Optional decision dataset (.jsonl) for Decision Alignment fine-tuning & temperature calibration
    #[arg(short, long)]
    pub data: Option<PathBuf>,

    /// Output directory for adapted bundle (decision.json, calibration.json, weights)
    #[arg(short, long)]
    pub output_dir: Option<PathBuf>,

    /// Number of alignment epochs if --data is provided
    #[arg(long, default_value_t = 5)]
    pub epochs: usize,

    /// Learning rate for alignment fine-tuning
    #[arg(long, default_value_t = 2e-4)]
    pub lr: f64,

    /// Batch size for alignment fine-tuning
    #[arg(long, default_value_t = 4)]
    pub batch_size: usize,

    /// Compute device: 'auto', 'cpu', 'cuda', 'rocm', or 'metal'
    #[arg(long, default_value = "auto")]
    pub device: String,

    /// Target runtime store(s) for registration: 'auto', 'ollaya', 'ollama', or 'both'
    #[arg(long, default_value = "auto")]
    pub runtime: RuntimeTarget,

    /// Custom Ollaya models store directory
    #[arg(long)]
    pub models_dir: Option<PathBuf>,
}

impl AdaptOpts {
    pub fn into_adapt_args(self) -> (AdaptArgs, Option<PathBuf>, RuntimeTarget) {
        let out_dir = self.output_dir.unwrap_or_else(|| {
            let safe = self.target.replace(':', "_");
            PathBuf::from(format!("./output_adapted_{}", safe))
        });
        (
            AdaptArgs {
                from: self.from,
                target: self.target,
                layout: self.layout,
                mode: self.mode,
                data_path: self.data,
                output_dir: out_dir,
                epochs: self.epochs,
                lr: self.lr,
                batch_size: self.batch_size,
                device: self.device,
            },
            self.models_dir,
            self.runtime,
        )
    }
}

#[derive(Args, Debug)]
pub struct TrainOpts {
    /// Path to Trainfile configuration (e.g. 'Trainfile', '.Trainfile')
    #[arg(short = 'f', long = "file")]
    pub file: Option<PathBuf>,

    /// Base model (Ollaya/Ollama tag, GGUF path, or HuggingFace Causal LLM repo)
    #[arg(long, default_value = "winnow:e4b")]
    pub base: String,

    /// Path to dataset file (JSONL or CSV)
    #[arg(short, long)]
    pub data: Option<PathBuf>,

    /// Task type: 'classification', 'nli', or 'decision'
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

    /// Number of top backbone layers to unfreeze during PyTorch end-to-end training
    #[arg(long)]
    pub unfreeze_layers: Option<usize>,

    /// Directory where the trained model/adapter and tokenizer will be saved
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

    /// Optional model family identifier override (e.g. 'llm-logits-v1', 'winnow-v1', 'kev-pointer-v1')
    #[arg(long)]
    pub family: Option<String>,

    /// Force merging adapter weights directly into base model
    #[arg(long, conflicts_with = "no_merge")]
    pub merge: bool,

    /// Force keeping adapter unmerged as a standalone sidecar
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
    pub fn resolve_training_args(
        self,
    ) -> anyhow::Result<(TrainingArgs, Option<String>, Option<PathBuf>)> {
        use crate::trainfile::Trainfile;
        use std::path::Path;

        let trainfile_path = if let Some(ref f) = self.file {
            Some(f.clone())
        } else if self.data.is_none() && self.base == "winnow:e4b" {
            Trainfile::find_default(Path::new("."))
        } else {
            None
        };

        if let Some(tf_path) = trainfile_path {
            println!("  Loading training recipe from: {}", tf_path.display());
            let tf = Trainfile::read(&tf_path)?;
            let mut args = tf.to_training_args();
            let mut register_name = tf.register;

            if let Some(data) = self.data {
                args.data_path = data;
            }
            if self.base != "winnow:e4b" {
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
            if let Some(ul) = self.unfreeze_layers {
                args.unfreeze_layers = ul;
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

        let data_path = self.data.unwrap_or_else(|| PathBuf::from("auto"));

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

        let default_out_dir = {
            let bp = PathBuf::from(&self.base);
            if bp.is_dir() && bp.join("model.gguf").exists() {
                bp
            } else {
                let cand = PathBuf::from("output").join(self.base.replace(':', "_"));
                if cand.is_dir() && cand.join("model.gguf").exists() {
                    cand
                } else {
                    PathBuf::from("./output_adapter")
                }
            }
        };

        let args = TrainingArgs {
            base_model: self.base,
            family: self.family,
            data_path,
            eval_path: self.eval_data,
            eval_split: self.eval_split.unwrap_or(0.1),
            output_dir: self.output_dir.unwrap_or(default_out_dir),
            task: self.task,
            epochs: self.epochs.unwrap_or(12),
            batch_size: self.batch_size.unwrap_or(8),
            grad_accum_steps: self.grad_accum.unwrap_or(1).max(1),
            lr: self.lr.unwrap_or(1.5e-4),
            warmup_ratio: self.warmup_ratio.unwrap_or(0.1),
            max_grad_norm: self.max_grad_norm.unwrap_or(1.0),
            lr_scheduler: self.lr_scheduler.unwrap_or_else(|| "cosine".to_string()),
            max_length: self.max_length.unwrap_or(128),
            weight_decay: self.weight_decay.unwrap_or(0.01),
            seed: self.seed.unwrap_or(42),
            qlora: self.qlora,
            bf16: self.bf16,
            lora_layers: self.lora_layers.unwrap_or(4),
            unfreeze_layers: self.unfreeze_layers.unwrap_or(0),
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
    #[arg(long)]
    pub base: String,

    #[arg(long)]
    pub lora: PathBuf,

    #[arg(short, long, default_value = "./output_merged")]
    pub output_dir: PathBuf,

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
    #[arg(short, long)]
    pub model: PathBuf,

    #[arg(short, long, default_value = "safetensors")]
    pub format: ExportFormat,

    #[arg(short, long)]
    pub quantize: Option<String>,

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
    #[arg(short, long, alias = "model-path", alias = "path")]
    pub model: PathBuf,

    #[arg(short, long, alias = "tag", alias = "target")]
    pub name: String,

    #[arg(long)]
    pub template: Option<String>,

    #[arg(long)]
    pub tokenizer: Option<PathBuf>,

    /// Target runtime store(s) for registration: 'auto', 'ollaya', 'ollama', or 'both'
    #[arg(long, default_value = "auto")]
    pub runtime: RuntimeTarget,

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
            runtime: opts.runtime,
        }
    }
}

#[derive(Args, Debug, Clone)]
pub struct BenchOpts {
    #[arg(short = 'f', long = "file")]
    pub file: Option<PathBuf>,

    #[arg(short, long)]
    pub model: Option<String>,

    #[arg(short, long)]
    pub baseline: Option<String>,

    #[arg(short, long)]
    pub data: Option<PathBuf>,

    #[arg(long, value_delimiter = ',')]
    pub subsets: Option<Vec<String>>,

    #[arg(long)]
    pub limit: Option<usize>,

    #[arg(long)]
    pub url: Option<String>,

    #[arg(short, long)]
    pub out: Option<PathBuf>,

    #[arg(long)]
    pub min_accuracy: Option<f64>,

    #[arg(long)]
    pub export_failures: Option<PathBuf>,

    /// Bypass the SHA-256 model binary result cache and force re-evaluating all probes
    #[arg(long)]
    pub no_cache: bool,
}

impl BenchOpts {
    pub fn resolve_benchfile(self) -> anyhow::Result<crate::benchfile::Benchfile> {
        let mut bf = if let Some(path) = &self.file {
            crate::benchfile::Benchfile::from_file(path)?
        } else if self.model.is_none() && std::path::Path::new("Benchfile").exists() {
            crate::benchfile::Benchfile::from_file(std::path::Path::new("Benchfile"))?
        } else if self.model.is_none() && std::path::Path::new(".Benchfile").exists() {
            crate::benchfile::Benchfile::from_file(std::path::Path::new(".Benchfile"))?
        } else {
            crate::benchfile::Benchfile::default()
        };

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
        if let Some(ef) = self.export_failures {
            bf.export_failures = Some(ef);
        }
        if self.no_cache {
            bf.no_cache = true;
        }

        if bf.model.is_empty() {
            anyhow::bail!(
                "Target model is required for benchmark. Provide --model <MODEL> or define 'MODEL <name>' in Benchfile."
            );
        }

        bf.resolve_placeholders(None);

        if bf.baseline.as_deref() == Some(bf.model.as_str()) {
            bf.baseline = None;
        }

        Ok(bf)
    }
}

#[derive(Args, Debug, Clone)]
pub struct CurateOpts {
    /// Path to Curatefile configuration (e.g. 'Curatefile', '.Curatefile')
    #[arg(short = 'f', long = "file")]
    pub file: Option<PathBuf>,

    /// Free-text description of your final Use-Case, domain rules, and decision objective
    #[arg(short = 'u', long = "use-case", alias = "description")]
    pub use_case: Option<String>,

    /// Teacher agent to distill ('agy', 'claude', 'codex', 'gemini', 'ollama:<model>', 'ollaya:<model>')
    #[arg(long)]
    pub teacher: Option<String>,

    /// Curation workflow mode: 'synth' (generate from Use-Case), 'reformat' / 'label' (reformat & distill dataset), or 'augment'
    #[arg(short, long)]
    pub mode: Option<String>,

    /// Target dataset format: 'decision' (default), 'classification', or 'nli'
    #[arg(long)]
    pub format: Option<String>,

    /// Decision primitive kind when synthesizing schema from Use-Case: 'choice' (default), 'noul', or 'score'
    #[arg(long = "kind", alias = "decision-type")]
    pub kind: Option<String>,

    /// Path to an existing raw dataset (.jsonl, .csv, .json, .txt) to reformat & distill
    #[arg(short, long)]
    pub input: Option<PathBuf>,

    /// Optional path to an explicit decision schema (.json); if omitted, Teacher synthesizes it from --use-case
    #[arg(short, long)]
    pub schema: Option<PathBuf>,

    /// Output path for the curated training dataset (.jsonl)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Number of samples to synthesize in 'synth' mode
    #[arg(long)]
    pub count: Option<usize>,

    /// Batch size for teacher distillation queries
    #[arg(long)]
    pub batch_size: Option<usize>,

    /// Minimum teacher confidence threshold (0.0..1.0) before quarantining ambiguous samples
    #[arg(long)]
    pub min_confidence: Option<f64>,

    /// Launch the interactive Use-Case & Teacher Distillation wizard
    #[arg(long = "interactive")]
    pub interactive: bool,

    /// Run strictly non-interactively using Curatefile or CLI flags
    #[arg(long = "non-interactive", conflicts_with = "interactive")]
    pub non_interactive: bool,
}

impl CurateOpts {
    pub fn resolve_curate_args(self) -> anyhow::Result<crate::curatefile::Curatefile> {
        let no_cli_workflow_flags = self.file.is_none()
            && self.use_case.is_none()
            && self.mode.is_none()
            && self.input.is_none()
            && self.schema.is_none();

        let mut cf = if let Some(path) = &self.file {
            crate::curatefile::Curatefile::from_file(path)?
        } else if std::path::Path::new("Curatefile").exists() {
            crate::curatefile::Curatefile::from_file(std::path::Path::new("Curatefile"))?
        } else if std::path::Path::new(".Curatefile").exists() {
            crate::curatefile::Curatefile::from_file(std::path::Path::new(".Curatefile"))?
        } else {
            crate::curatefile::Curatefile::default()
        };

        if let Some(uc) = self.use_case {
            cf.use_case = Some(uc);
        }
        if let Some(t) = self.teacher {
            cf.teacher = t;
        }
        if let Some(m) = self.mode {
            cf.mode = match m.to_lowercase().as_str() {
                "reformat" | "distill" | "label" | "annotate" => {
                    crate::curatefile::CurateMode::Label
                }
                "synth" | "synthetic" | "generate" => crate::curatefile::CurateMode::Synth,
                "augment" | "contrastive" => crate::curatefile::CurateMode::Augment,
                other => anyhow::bail!(
                    "Unknown mode '{}'. Expected: synth, reformat, label, augment",
                    other
                ),
            };
        } else if self.input.is_some() && cf.mode == crate::curatefile::CurateMode::Synth {
            // If user passed --input without --mode, default to reformat/label mode
            cf.mode = crate::curatefile::CurateMode::Label;
        }
        if let Some(f) = self.format {
            let lower = f.to_lowercase();
            cf.format = match lower.as_str() {
                "decision" | "choice" | "noul" | "score" => {
                    crate::curatefile::TargetDatasetFormat::Decision
                }
                "classification" | "text" => crate::curatefile::TargetDatasetFormat::Classification,
                "nli" | "pair" | "pairs" => crate::curatefile::TargetDatasetFormat::Nli,
                other => anyhow::bail!(
                    "Unknown format '{}'. Expected: decision, classification, nli",
                    other
                ),
            };
            if matches!(lower.as_str(), "choice" | "noul" | "score") {
                cf.kind = lower;
            }
        }
        if let Some(k) = self.kind {
            let lower = k.to_lowercase();
            match lower.as_str() {
                "choice" | "noul" | "score" => cf.kind = lower,
                other => anyhow::bail!("Unknown kind '{}'. Expected: choice, noul, score", other),
            }
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

        if self.interactive {
            cf.interactive = true;
        } else if self.non_interactive {
            cf.interactive = false;
        } else if no_cli_workflow_flags {
            // Default to interactive wizard when `trainaya curate` is run directly without CLI flags
            cf.interactive = true;
        }

        Ok(cf)
    }
}

#[derive(Args, Debug, Clone)]
pub struct UpdateOpts {
    #[arg(short, long)]
    pub check: bool,

    #[arg(short, long)]
    pub force: bool,

    #[arg(long)]
    pub version: Option<String>,
}
