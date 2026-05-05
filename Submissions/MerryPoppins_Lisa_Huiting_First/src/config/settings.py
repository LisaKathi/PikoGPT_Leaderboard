from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class EmailConfig(BaseModel):
    enabled: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""                  # sender Gmail address
    to_addresses: List[str] = []         # recipient addresses
    cooldown_minutes: int = 5            # minimum gap between warnings to avoid spam
    periodic_status_hours: float = 4.0  # send a status email every N hours (0 = disabled)
    disk_min_gb: float = 5.0            # warn when free disk space drops below this (GB)
    # password is read from env var PIKOGPT_SMTP_PASSWORD — never put it in the TOML


class RunConfig(BaseModel):
    name: str = "default_run"
    artifacts_root: Path = Path("logs")
    resume: bool = True
    deterministic: bool = False


class DistributedConfig(BaseModel):
    enabled: bool = False
    backend: str = "nccl"


# This File uses Pydantic to give default values, while experiment_01 gives us experiments

class DataConfig(BaseModel):
    data_dir: Path = Path("data")
    processed_dir: Path = Path("data/processed")
    log_dir: Path = Path("logs/preprocessing")
    train_split: float = 0.9
    num_proc: int = 4
    seed: int = 42
    subset_size: int = 0                        # 0 = use full dataset
    raw_data_path: Optional[Path] = None        # if set, load from disk instead of downloading
    test_data_path: Optional[Path] = None       # if set, deduplicate against this test split
    # Filter toggles (individually disable specific filter steps)
    skip_language_filter: bool = False
    skip_repetition_filter: bool = False
    skip_quality_filter: bool = False

    # Quality filter thresholds
    min_words: int = 100
    max_words: int = 10_000
    max_non_ascii: float = 0.3                  # max fraction of non-ASCII characters
    min_line_uniqueness: float = 0.7            # min unique_lines / total_lines
    min_sentence_uniqueness: float = 0.8        # min unique_sentences / total_sentences

    # Token cap (0 = no cap). Truncates train tokens to this limit after preprocessing.
    max_train_tokens: int = 0


class ModelConfig(BaseModel):
    vocab_size: int = 50304       # GPT-2 vocab (50257) rounded up to nearest multiple of 64
    n_layers: int = 6
    n_heads: int = 6
    n_embd: int = 384
    context_len: int = 1024
    dropout: float = 0.0
    bias: bool = False
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    norm_eps: float = 1e-5
    positional_embedding: Literal["learned", "rope"] = "learned"
    rope_theta: float = 10_000.0
    rope_fraction: float = 1.0
    mlp_type: Literal["gelu", "swiglu"] = "gelu"
    mlp_hidden_mult: float = 4.0
    qk_norm: bool = False
    block_style: Literal["sequential", "parallel"] = "sequential"


class TrainingConfig(BaseModel):
    seed: int = 0

    # Optimization
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Schedule
    max_iters: int = 200_000
    warmup_steps: int = 150
    lr_schedule: Literal["cosine", "wsd"] = "cosine"
    wsd_stable_frac: float = 0.85

    # Batching
    batch_size: int = 32
    gradient_accumulation_steps: int = 4

    # Precision
    dtype: str = "bfloat16"

    # Device
    device: str = "auto"

    # Evaluation & Logging
    eval_step_interval: int = 500
    eval_batches: int = 20
    log_interval: int = 1

    # Checkpointing
    max_checkpoints: int = 10


class InferenceConfig(BaseModel):
    checkpoint: Optional[Path] = None
    prompt: str = ""
    max_tokens: int = 100
    temperature: float = 1.0
    seed: int = 0
    device: str = "auto"
    leaderboard: bool = False


class PostTrainingConfig(BaseModel):
    base_checkpoint: Optional[Path] = None
    resume_checkpoint: Optional[Path] = None  # if set, resume SFT from this checkpoint

    # Dataset mode: "alpaca" = original single-dataset, "mixed" = multi-task benchmark mix
    dataset_mode: str = "mixed"
    dataset: str = "tatsu-lab/alpaca"           # used only in "alpaca" mode

    # Mixed-mode dataset sample caps (0 = use all available, -1 = disable dataset)
    alpaca_samples: int = 10_000
    hellaswag_samples: int = 8_000
    winogrande_samples: int = 8_000
    openbookqa_samples: int = 0                 # small (~4.9k), use all
    lambada_samples: int = -1       # disabled: all LAMBADA HF variants are eval-only (test split only)
    arc_easy_samples: int = 0                   # small (~2.2k), use all
    arc_challenge_samples: int = 0              # small (~1.1k), use all
    piqa_samples: int = -1          # disabled: ybisk/piqa uses a loading script unsupported in datasets>=3.x
    boolq_samples: int = 5_000
    commonsenseqa_samples: int = 5_000
    replay_data_path: str = ""                  # path to pretraining .bin or .txt for replay
    replay_samples: int = 2_000                 # 0 = disabled

    max_seq_len: int = 512
    # Lowered LR vs original 1e-5 — reduces catastrophic forgetting of pretrained knowledge
    learning_rate: float = 5e-6
    min_lr: float = 5e-7
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    # 3k steps was the sweet spot — beyond that Alpaca-only training hurt benchmarks
    max_iters: int = 3_000
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    dtype: str = "bfloat16"
    device: str = "auto"
    eval_interval: int = 200
    checkpoint_interval: int = 0   # 0 = save checkpoint at every eval (old behaviour)
    eval_samples: int = 200
    log_interval: int = 10
    seed: int = 42
    max_checkpoints: int = 3
    checkpoint_dir: Path = Path("checkpoints/post")
    log_dir: Path = Path("logs/post")


class BenchmarkConfig(BaseModel):
    checkpoint: Optional[Path] = None
    device: str = "auto"
    log_dir: Path = Path("logs/benchmark")
    benchmarks: List[str] = ["lambada", "hellaswag", "winogrande", "openbookqa"]
    max_samples: Optional[int] = None


class EvaluationConfig(BaseModel):
    checkpoint: Optional[Path] = None
    batch_size: int = 8
    device: str = "auto"
    log_dir: Path = Path("logs/evaluation")


class ExperimentConfig(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)
    preprocessing: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    post_training: PostTrainingConfig = Field(default_factory=PostTrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    notifications: EmailConfig = Field(default_factory=EmailConfig)


def load_config(path: Path) -> ExperimentConfig:
    import sys

    if sys.version_info >= (3, 11):
        import tomllib
        loader = tomllib.loads
    else:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
            loader = tomllib.loads
        except ImportError as e:
            raise ImportError("Install 'tomli' for Python < 3.11: pip install tomli") from e

    raw = loader(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(raw)
