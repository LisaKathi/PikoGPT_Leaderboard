from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


# -----------------------
# Data structures
# -----------------------
@dataclass
class EvalResult:
    total: int = 0
    correct: int = 0
    invalid: int = 0  # example couldn't be evaluated (e.g., subprocess failed / couldn't parse)


# -----------------------
# IO helpers
# -----------------------
def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def default_results_path(output_dir: Path, bench: str, checkpoint: Path, data_path: Path) -> Path:
    ckpt_name = checkpoint.stem
    split = data_path.stem
    bench_dir = output_dir / bench
    bench_dir.mkdir(parents=True, exist_ok=True)
    return bench_dir / f"{bench}__{ckpt_name}__{split}.json"


def sanitize_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


def overview_results_path(output_dir: Path, submission: str, checkpoint: Path) -> Path:
    sub_name = sanitize_name(submission or "submission")
    ckpt_name = sanitize_name(checkpoint.stem)
    return output_dir / f"{sub_name}__{ckpt_name}__overview.json"


def resolve_under_submission(submission_dir: Path, maybe_relative: Path) -> Path:
    if maybe_relative.is_absolute():
        return maybe_relative.resolve()
    return (submission_dir / maybe_relative).resolve()


# -----------------------
# CLI inference backend
# -----------------------
def run_inference_cli(
    *,
    python_exe: str,
    main_path: Path,
    checkpoint: Path,
    run_cwd: Path,
    prompt: str,
    max_tokens: int,
    temperature: float,
    device: str,
    seed: int,
    timeout_s: int,
) -> str:
    """
    Calls the student's CLI inference stage and returns STDOUT (expected: ONLY generated continuation text).
    IMPORTANT: Student main.py/inference must not print banners to stdout when verbose is false.
    """
    cmd = [
        python_exe,
        str(main_path),
        "--stage",
        "inference",
        "--checkpoint",
        str(checkpoint),
        "--prompt",
        prompt,
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        str(temperature),
        "--device",
        device,
        "--leaderboard",
        "--seed",
        str(seed),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(run_cwd),
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Inference timed out after {timeout_s}s. cmd={cmd!r}") from e

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        msg = f"Inference failed (exit={proc.returncode})."
        if stderr:
            msg += f"\n--- STDERR ---\n{stderr}"
        if stdout:
            msg += f"\n--- STDOUT ---\n{stdout}"
        msg += f"\n--- CMD ---\n{cmd!r}"
        raise RuntimeError(msg)

    return proc.stdout or ""


# -----------------------
# Parsing helpers (generation-based)
# -----------------------
def parse_mc_letter(gen: str, allowed_letters: set[str]) -> Optional[str]:
    """
    For multiple choice: generate 2-3 tokens, strip leading whitespace, take first char.
    """
    s = gen.lstrip()
    if not s:
        return None
    ch = s[0].upper()
    return ch if ch in allowed_letters else None


def normalize_lambada(s: str) -> str:
    s = s.strip()
    s = s.strip(" \t\r\n\"'“”‘’.,;:!?()[]{}")
    return s.lower()


def parse_lambada_word(gen: str) -> str:
    """
    For LAMBADA: generate ~5 tokens, strip leading whitespace, take first word, strip punctuation.
    """
    s = gen.lstrip()
    if not s:
        return ""
    word = re.split(r"\s+", s, maxsplit=1)[0]
    return normalize_lambada(word)


# -----------------------
# Benchmark evaluators (CLI inference only)
# -----------------------
def eval_mc_benchmark_cli(
    *,
    bench_name: str,
    allowed_letters: set[str],
    data_path: Path,
    checkpoint: Path,
    python_exe: str,
    main_path: Path,
    run_cwd: Path,
    device: str,
    seed: int,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
    limit: Optional[int],
    verbose: bool,
    save_wrong: int,
) -> Tuple[EvalResult, list[dict]]:
    res = EvalResult()
    wrong: list[dict] = []

    for i, ex in enumerate(read_jsonl(data_path)):
        if limit is not None and i >= limit:
            break

        prompt = ex["prompt"]
        gold = str(ex["answer_letter"]).upper()

        try:
            gen = run_inference_cli(
                python_exe=python_exe,
                main_path=main_path,
                checkpoint=checkpoint,
                run_cwd=run_cwd,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                device=device,
                seed=seed,
                timeout_s=timeout_s,
            )
            pred = parse_mc_letter(gen, allowed_letters)
        except Exception as e:
            res.total += 1
            res.invalid += 1
            if verbose:
                print("------------------------------------------------------------")
                print(f"bench: {bench_name}")
                print(f"idx:   {i}")
                print(f"id:    {ex.get('id')}")
                print("ERROR:")
                print(repr(e))
                print("------------------------------------------------------------\n")
            continue

        res.total += 1

        if pred is None:
            res.invalid += 1
            ok = False
        else:
            ok = (pred == gold)
            if ok:
                res.correct += 1

        if (not ok) and len(wrong) < save_wrong:
            wrong.append(
                {
                    "id": ex.get("id"),
                    "gold": gold,
                    "pred": pred,
                    "prompt": prompt,
                    "raw_generation": gen,
                }
            )

        if verbose:
            print("------------------------------------------------------------")
            print(f"bench: {bench_name}")
            print(f"idx:   {i}")
            print(f"id:    {ex.get('id')}")
            print(f"gold:  {gold}")
            print(f"pred:  {pred}")
            print(f"raw:   {gen!r}")
            print(f"ok:    {ok}")
            print("------------------------------------------------------------\n")

    return res, wrong


def eval_lambada_cli(
    *,
    data_path: Path,
    checkpoint: Path,
    python_exe: str,
    main_path: Path,
    run_cwd: Path,
    device: str,
    seed: int,
    max_gen_tokens: int,
    temperature: float,
    timeout_s: int,
    limit: Optional[int],
    verbose: bool,
    save_wrong: int,
) -> Tuple[EvalResult, list[dict]]:
    res = EvalResult()
    wrong: list[dict] = []

    for i, ex in enumerate(read_jsonl(data_path)):
        if limit is not None and i >= limit:
            break

        prompt = ex["prompt"]
        gold = ex["answer_text"]
        gold_n = normalize_lambada(gold)

        try:
            gen = run_inference_cli(
                python_exe=python_exe,
                main_path=main_path,
                checkpoint=checkpoint,
                run_cwd=run_cwd,
                prompt=prompt,
                max_tokens=max_gen_tokens,
                temperature=temperature,
                device=device,
                seed=seed,
                timeout_s=timeout_s,
            )
            pred_n = parse_lambada_word(gen)
        except Exception as e:
            res.total += 1
            res.invalid += 1
            if verbose:
                print("------------------------------------------------------------")
                print("bench: lambada")
                print(f"idx:   {i}")
                print(f"id:    {ex.get('id')}")
                print("ERROR:")
                print(repr(e))
                print("------------------------------------------------------------\n")
            continue

        res.total += 1
        ok = (pred_n == gold_n)
        if ok:
            res.correct += 1
        else:
            if len(wrong) < save_wrong:
                wrong.append(
                    {
                        "id": ex.get("id"),
                        "gold": gold,
                        "pred": pred_n,
                        "prompt": prompt,
                        "raw_generation": gen,
                    }
                )

        if verbose:
            print("------------------------------------------------------------")
            print("bench: lambada")
            print(f"idx:   {i}")
            print(f"id:    {ex.get('id')}")
            print(f"gold:  {gold} (norm={gold_n})")
            print(f"pred:  {pred_n}")
            print(f"raw:   {gen!r}")
            print(f"ok:    {ok}")
            print("------------------------------------------------------------\n")

    return res, wrong


# -----------------------
# Dataset paths + chance thresholds
# -----------------------
DEFAULT_DATA = {
    "hellaswag": Path("leaderboard/benchmarks/hellaswag/cleaned/validation.jsonl"),
    "winogrande": Path("leaderboard/benchmarks/winogrande/cleaned/validation.jsonl"),
    "openbookqa": Path("leaderboard/benchmarks/openbookqa/cleaned/validation.jsonl"),
    "lambada": Path("leaderboard/benchmarks/lambada/cleaned/test.jsonl"),
}

RANDOM_CHANCE = {
    "hellaswag": 25.0,
    "winogrande": 50.0,
    "openbookqa": 25.0,
    # lambada omitted
}


# -----------------------
# Main
# -----------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="PikoGPT Leaderboard Benchmark Runner (CLI inference only).")

    parser.add_argument(
        "--submission",
        type=str,
        required=True,
        help="Submission folder name under --submissions-dir (e.g. Group1-test).",
    )
    parser.add_argument(
        "--submissions-dir",
        type=Path,
        default=Path("Submissions"),
        help="Root directory that contains submission folders (default: Submissions).",
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to run student main.py")
    parser.add_argument(
        "--main-path",
        type=Path,
        default=Path("main.py"),
        help="Path to main.py relative to the selected submission folder (or absolute path).",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device passed to student CLI (auto/cpu/cuda/mps)")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint path relative to the selected submission folder (or absolute path).",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for inference calls (default: 0.0 for deterministic).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed passed to student inference in leaderboard mode (default: 0).",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=60,
        help="Timeout per inference call in seconds (default: 60).",
    )
    parser.add_argument(
        "--mc-max-tokens",
        type=int,
        default=3,
        help="Max tokens to generate for multiple-choice tasks (default: 3).",
    )
    parser.add_argument(
        "--lambada-max-tokens",
        type=int,
        default=5,
        help="Max tokens to generate for LAMBADA next-word (default: 5).",
    )

    PUBLIC_BENCHES = ["hellaswag", "winogrande", "openbookqa", "lambada"]

    parser.add_argument(
        "--bench",
        nargs="+",
        choices=PUBLIC_BENCHES,
        default=PUBLIC_BENCHES,
        help="Benchmarks to run (default: all public benchmarks).",
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to preprocessed benchmark JSONL (only valid when running one benchmark)",
    )

    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N examples (per benchmark)")
    parser.add_argument("--verbose", action="store_true", help="Print per-example results")
    parser.add_argument("--save-wrong", type=int, default=20, help="How many wrong examples to include")

    args = parser.parse_args()

    submission_dir = (args.submissions_dir / args.submission).resolve()
    if not submission_dir.is_dir():
        raise FileNotFoundError(f"Submission directory not found: {submission_dir}")

    main_path = resolve_under_submission(submission_dir, args.main_path)
    checkpoint = resolve_under_submission(submission_dir, args.checkpoint)

    if not main_path.exists():
        raise FileNotFoundError(f"main.py not found at: {main_path}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    checkpoint_name = sanitize_name(checkpoint.stem)
    output_dir = Path("Results") / sanitize_name(args.submission) / checkpoint_name

    if args.data and len(args.bench) != 1:
        raise ValueError("--data can only be used when running exactly one benchmark.")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print("PikoGPT - Leaderboard Benchmark Runner (CLI inference only)")
    print("============================================================")
    print(f"Submission:  {args.submission}")
    print(f"Sub dir:     {submission_dir}")
    print(f"Main path:   {main_path}")
    print(f"Python:      {args.python}")
    print(f"Checkpoint:  {checkpoint}")
    print(f"Benchmarks:  {', '.join(args.bench)}")
    if args.limit:
        print(f"Limit:       {args.limit}")
    print(f"Output dir:  {output_dir}")
    print(f"Device:      {args.device}")
    print(f"Temp:        {args.temperature}")
    print(f"Seed:        {args.seed}")
    print(f"Timeout (s):  {args.timeout_s}")
    print(f"MC tokens:   {args.mc_max_tokens}")
    print(f"LAMBADA tok: {args.lambada_max_tokens}")
    print("============================================================\n")

    overall_exit_ok = True

    overview: dict = {
        "submission": args.submission,
        "submission_dir": str(submission_dir),
        "checkpoint": str(checkpoint),
        "limit": args.limit,
        "output_dir": str(output_dir),
        "main_path": str(main_path),
        "python": args.python,
        "device": args.device,
        "temperature": args.temperature,
        "seed": args.seed,
        "timeout_s": args.timeout_s,
        "mc_max_tokens": args.mc_max_tokens,
        "lambada_max_tokens": args.lambada_max_tokens,
        "benchmarks": [],
    }

    for bench in args.bench:
        data_path = args.data if (args.data and len(args.bench) == 1) else DEFAULT_DATA[bench]
        if not data_path.is_file():
            raise FileNotFoundError(f"Preprocessed benchmark file not found for {bench}: {data_path}")

        print(f"--- Running: {bench} ---")
        print(f"Data: {data_path}\n")

        if bench == "hellaswag":
            res, details = eval_mc_benchmark_cli(
                bench_name=bench,
                allowed_letters={"A", "B", "C", "D"},
                data_path=data_path,
                checkpoint=checkpoint,
                python_exe=args.python,
                main_path=main_path,
                run_cwd=submission_dir,
                device=args.device,
                seed=args.seed,
                max_tokens=args.mc_max_tokens,
                temperature=args.temperature,
                timeout_s=args.timeout_s,
                limit=args.limit,
                verbose=args.verbose,
                save_wrong=args.save_wrong,
            )
        elif bench == "winogrande":
            res, details = eval_mc_benchmark_cli(
                bench_name=bench,
                allowed_letters={"A", "B"},
                data_path=data_path,
                checkpoint=checkpoint,
                python_exe=args.python,
                main_path=main_path,
                run_cwd=submission_dir,
                device=args.device,
                seed=args.seed,
                max_tokens=args.mc_max_tokens,
                temperature=args.temperature,
                timeout_s=args.timeout_s,
                limit=args.limit,
                verbose=args.verbose,
                save_wrong=args.save_wrong,
            )
        elif bench == "openbookqa":
            res, details = eval_mc_benchmark_cli(
                bench_name=bench,
                allowed_letters={"A", "B", "C", "D"},
                data_path=data_path,
                checkpoint=checkpoint,
                python_exe=args.python,
                main_path=main_path,
                run_cwd=submission_dir,
                device=args.device,
                seed=args.seed,
                max_tokens=args.mc_max_tokens,
                temperature=args.temperature,
                timeout_s=args.timeout_s,
                limit=args.limit,
                verbose=args.verbose,
                save_wrong=args.save_wrong,
            )
        elif bench == "lambada":
            res, details = eval_lambada_cli(
                data_path=data_path,
                checkpoint=checkpoint,
                python_exe=args.python,
                main_path=main_path,
                run_cwd=submission_dir,
                device=args.device,
                seed=args.seed,
                max_gen_tokens=args.lambada_max_tokens,
                temperature=args.temperature,
                timeout_s=args.timeout_s,
                limit=args.limit,
                verbose=args.verbose,
                save_wrong=args.save_wrong,
            )
        else:
            raise ValueError(f"Unknown benchmark: {bench}")

        # metrics
        acc = (res.correct / res.total * 100.0) if res.total else 0.0

        # print summary
        print("============================================================")
        print("SUMMARY")
        print("============================================================")
        print(f"{bench}: {res.correct}/{res.total} ({acc:.2f}%)")
        print(f"invalid: {res.invalid}/{res.total}")
        print("============================================================\n")

        # decide save path
        save_path = default_results_path(output_dir, bench, checkpoint, data_path)

        payload: dict = {
            "benchmark": bench,
            "submission": args.submission,
            "checkpoint": str(checkpoint),
            "data": str(data_path),
            "limit": args.limit,
            "total": res.total,
            "correct": res.correct,
            "invalid": res.invalid,
            "accuracy_pct": acc,
            "wrong_examples": details,
        }
        save_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved results to: {save_path}\n")

        # overview item
        ov_item: dict = {
            "benchmark": bench,
            "data": str(data_path),
            "total": res.total,
            "correct": res.correct,
            "invalid": res.invalid,
            "accuracy_pct": acc,
        }
        overview["benchmarks"].append(ov_item)

        # exit condition only for public MC tasks (optional)
        chance = RANDOM_CHANCE.get(bench)
        if chance is not None and acc <= chance:
            overall_exit_ok = False

    # write overview at END
    ov_path = overview_results_path(output_dir, args.submission, checkpoint)
    ov_path.write_text(json.dumps(overview, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved overview to: {ov_path}\n")

    raise SystemExit(0 if overall_exit_ok else 1)


if __name__ == "__main__":
    main()
