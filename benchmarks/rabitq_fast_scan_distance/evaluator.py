"""
Evaluator for the RaBitQ fast-scan distance evolution benchmark.

Fitness = QPS reported by bench_rabitq_fast_scan_distance, or 0.0 if:
  - the benchmark build fails
  - the benchmark exits non-zero
  - the CHECKSUM does not match the reference run

The reference checksum is computed lazily on the first call using the
unmodified rabitq_fast_scan.rs already on disk in the sibling scratch project.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

BENCHMARK_DIR = Path(__file__).resolve().parent
SKYDISCOVER_ROOT = BENCHMARK_DIR.parents[1]
SCRATCH_ROOT = SKYDISCOVER_ROOT.parent / "scratch"
TARGET_SRC = SCRATCH_ROOT / "src/data_handling/rabitq_fast_scan.rs"
SOURCE_TEMPLATE = BENCHMARK_DIR / "source_template.rs"
DRIVER_MANIFEST = BENCHMARK_DIR / "bench_driver/Cargo.toml"
TARGET_DIR = BENCHMARK_DIR / "target"
BENCH_BIN_NAME = "bench_rabitq_fast_scan_distance"
BENCH_BIN = TARGET_DIR / "release" / BENCH_BIN_NAME

_lock = threading.Lock()
_reference_checksum: Optional[str] = None
_reference_qps: Optional[float] = None


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    rustflags = env.get("RUSTFLAGS", "").strip()
    native_flag = "-C target-cpu=native"
    env["RUSTFLAGS"] = f"{rustflags} {native_flag}".strip() if rustflags else native_flag
    env["CARGO_TARGET_DIR"] = str(TARGET_DIR)
    return env


def _run_bench(timeout: int = 180) -> dict[str, str]:
    result = subprocess.run(
        [str(BENCH_BIN)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=BENCHMARK_DIR,
    )
    if result.returncode != 0:
        return {"_error": (result.stderr + "\n" + result.stdout).strip()}

    parsed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            parsed[key.strip()] = value.strip()
    return parsed


def _build(timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "cargo",
            "build",
            "--offline",
            "--locked",
            "--release",
            "--manifest-path",
            str(DRIVER_MANIFEST),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=BENCHMARK_DIR,
        env=_build_env(),
    )


def _load_marked_block(program_text: str, start_marker: str, end_marker: str) -> str:
    if start_marker in program_text and end_marker in program_text:
        start = program_text.index(start_marker) + len(start_marker)
        end = program_text.index(end_marker, start)
        return program_text[start:end].strip("\n")
    raise ValueError(f"Missing required markers: {start_marker} / {end_marker}")


def _load_evolve_blocks(program_path: str) -> dict[str, str]:
    program_text = Path(program_path).read_text()
    return {
        "__EVOLVE_AVX2_BLOCK__": _load_marked_block(
            program_text,
            "// EVOLVE-AVX2-BLOCK-START",
            "// EVOLVE-AVX2-BLOCK-END",
        ),
        "__EVOLVE_ORACLE_BLOCK__": _load_marked_block(
            program_text,
            "// EVOLVE-ORACLE-BLOCK-START",
            "// EVOLVE-ORACLE-BLOCK-END",
        ),
    }


def _render_candidate_source(program_path: str) -> str:
    template = SOURCE_TEMPLATE.read_text()
    for placeholder, block in _load_evolve_blocks(program_path).items():
        template = template.replace(placeholder, block)
    return template


def _ensure_reference_checksum() -> None:
    global _reference_checksum, _reference_qps

    build = _build()
    if build.returncode != 0:
        raise RuntimeError(
            "Could not build reference benchmark:\n" + build.stderr[-3000:]
        )

    parsed = _run_bench()
    if "_error" in parsed:
        raise RuntimeError("Reference benchmark run failed:\n" + parsed["_error"][-3000:])

    _reference_checksum = parsed.get("CHECKSUM")
    _reference_qps = float(parsed.get("QPS", 0.0))
    if not _reference_checksum:
        raise RuntimeError(f"No CHECKSUM in reference output: {parsed}")


def evaluate(program_path: str) -> dict:
    global _reference_checksum

    with _lock:
        if _reference_checksum is None:
            try:
                _ensure_reference_checksum()
            except Exception as exc:
                return {
                    "combined_score": 0.0,
                    "artifacts": {"setup_error": str(exc)},
                }

        backup_path = TARGET_SRC.with_suffix(TARGET_SRC.suffix + ".skydiscover_bak")
        shutil.copy2(TARGET_SRC, backup_path)

        try:
            TARGET_SRC.write_text(_render_candidate_source(program_path))

            build = _build()
            if build.returncode != 0:
                return {
                    "combined_score": 0.0,
                    "artifacts": {"compiler_error": build.stderr[-3000:]},
                }

            parsed = _run_bench()
            if "_error" in parsed:
                return {
                    "combined_score": 0.0,
                    "artifacts": {"run_error": parsed["_error"][-3000:]},
                }

            checksum = parsed.get("CHECKSUM", "")
            qps = float(parsed.get("QPS", 0.0))

            if checksum != _reference_checksum:
                return {
                    "combined_score": 0.0,
                    "artifacts": {
                        "checksum_mismatch": (
                            f"expected={_reference_checksum} got={checksum} — "
                            "the optimized fast-scan distance path changed output bits. "
                            "Only exact semantics-preserving rewrites are accepted."
                        )
                    },
                }

            return {
                "combined_score": qps,
                "qps": qps,
                "checksum": checksum,
                "avg_query_ms": float(parsed.get("AVG_QUERY_MS", 0.0)),
                "avg_blocks_per_query": float(parsed.get("AVG_BLOCKS_PER_QUERY", 0.0)),
                "reference_qps": _reference_qps,
            }

        except subprocess.TimeoutExpired:
            return {
                "combined_score": 0.0,
                "artifacts": {"run_error": "Timed out during build or benchmark run."},
            }
        except Exception as exc:
            return {
                "combined_score": 0.0,
                "artifacts": {"run_error": str(exc)},
            }
        finally:
            shutil.copy2(backup_path, TARGET_SRC)
            os.unlink(backup_path)
