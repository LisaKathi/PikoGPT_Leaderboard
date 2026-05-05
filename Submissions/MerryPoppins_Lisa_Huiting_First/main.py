import argparse
import logging
import sys
from pathlib import Path

import src.stages.inference  # noqa: F401

from src.stages.base import StageRegistry
from src.config.settings import ExperimentConfig, load_config
from src.utils.logging_config import setup_logging


logger = logging.getLogger(__name__)


def _load_experiment_config(config_path: str | None) -> ExperimentConfig:
    if config_path is not None:
        return load_config(Path(config_path))
    return ExperimentConfig()


def run_inference(args: argparse.Namespace) -> None:
    from src.config.settings import InferenceConfig

    cfg = _load_experiment_config(args.config)

    inference_cfg = cfg.inference.model_copy(
        update={
            k: v
            for k, v in {
                "checkpoint": Path(args.checkpoint) if args.checkpoint else None,
                "prompt": args.prompt or cfg.inference.prompt,
                "max_tokens": args.max_tokens if args.max_tokens is not None else cfg.inference.max_tokens,
                "temperature": args.temperature,
                "seed": args.seed,
                "device": args.device,
                "leaderboard": args.leaderboard,
            }.items()
            if v is not None
        }
    )

    stage_cls = StageRegistry.get("inference")
    stage = stage_cls(config=inference_cfg)
    stage.run()


_STAGE_RUNNERS = {
    "inference": run_inference,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pikogpt",
        description="PikoGPT — inference",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=StageRegistry.all_names(),
        metavar="STAGE",
        help=f"Pipeline stage to run. Choices: {StageRegistry.all_names()}",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to a TOML experiment config (optional)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        metavar="CKPT.pt",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        metavar="TEXT",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        dest="max_tokens",
        metavar="N",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--device",
        default="auto",
    )
    parser.add_argument(
        "--leaderboard",
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        metavar="LEVEL",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(
        level=args.log_level,
        log_file=Path(args.log_file) if args.log_file else None,
        disable_logging=args.leaderboard,
    )

    runner = _STAGE_RUNNERS[args.stage]
    try:
        runner(args)
    except NotImplementedError as exc:
        if args.leaderboard:
            sys.exit(1)
        logger.error("Stage '%s' is not yet implemented: %s", args.stage, exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
