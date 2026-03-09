"""
Evaluator for the RaBitQ QPS evolution benchmark.

Fitness = QPS reported by bench_rabitq_openai_arxiv_qps, or 0.0 if:
  - cargo build fails
  - the binary exits non-zero
  - the CHECKSUM does not match the reference run (i.e. results changed)

The reference checksum is computed lazily on the first call using the
unmodified classic_graph.rs that is already on disk in the scratch project.
"""

import os
import shutil
import subprocess
import threading
from typing import Optional

SCRATCH_ROOT = "/home/btl46/scratch"
GRAPH_SRC = os.path.join(SCRATCH_ROOT, "src/graph/classic_graph.rs")
BENCH_BIN_NAME = "bench_rabitq_openai_arxiv_qps"
BENCH_BIN = os.path.join(SCRATCH_ROOT, "target/release", BENCH_BIN_NAME)

_lock = threading.Lock()          # serialises the file-surgery + build step
_reference_checksum: Optional[str] = None
_reference_qps: Optional[float] = None


def _run_bench(timeout: int = 120) -> dict:
    """Run the benchmark binary and parse its output. Returns a dict of key→value strings."""
    result = subprocess.run(
        [BENCH_BIN],
        capture_output=True,
        timeout=timeout,
        cwd=SCRATCH_ROOT,
    )
    if result.returncode != 0:
        return {"_error": result.stderr.decode(errors="replace") + result.stdout.decode(errors="replace")}
    parsed = {}
    for line in result.stdout.decode(errors="replace").splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            parsed[k.strip()] = v.strip()
    return parsed


def _build(timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["cargo", "build", "--release", f"--bin={BENCH_BIN_NAME}"],
        capture_output=True,
        timeout=timeout,
        cwd=SCRATCH_ROOT,
    )


def _ensure_reference_checksum():
    """Build + run the unmodified binary once to capture the reference checksum."""
    global _reference_checksum, _reference_qps

    build = _build()
    if build.returncode != 0:
        raise RuntimeError(
            "Could not build reference binary:\n"
            + build.stderr.decode(errors="replace")[-2000:]
        )

    parsed = _run_bench()
    if "_error" in parsed:
        raise RuntimeError("Reference bench run failed:\n" + parsed["_error"][-2000:])

    _reference_checksum = parsed.get("CHECKSUM")
    _reference_qps = float(parsed.get("QPS", 0.0))
    if not _reference_checksum:
        raise RuntimeError(f"No CHECKSUM in reference output: {parsed}")


def evaluate(program_path: str) -> dict:
    global _reference_checksum

    with _lock:
        # ── 1. Ensure we have a reference checksum ────────────────────────────
        if _reference_checksum is None:
            try:
                _ensure_reference_checksum()
            except Exception as exc:
                return {
                    "combined_score": 0.0,
                    "artifacts": {"setup_error": str(exc)},
                }

        # ── 2. Back up the original source file ───────────────────────────────
        backup_path = GRAPH_SRC + ".skydiscover_bak"
        shutil.copy2(GRAPH_SRC, backup_path)

        try:
            # ── 3. Splice in the evolved file ─────────────────────────────────
            shutil.copy2(program_path, GRAPH_SRC)

            # ── 4. Incremental release build ──────────────────────────────────
            build = _build()
            if build.returncode != 0:
                stderr_tail = build.stderr.decode(errors="replace")[-3000:]
                return {
                    "combined_score": 0.0,
                    "artifacts": {"compiler_error": stderr_tail},
                }

            # ── 5. Run the benchmark ──────────────────────────────────────────
            parsed = _run_bench()
            if "_error" in parsed:
                return {
                    "combined_score": 0.0,
                    "artifacts": {"run_error": parsed["_error"][-2000:]},
                }

            checksum = parsed.get("CHECKSUM", "")
            qps = float(parsed.get("QPS", 0.0))

            # ── 6. Validate checksum ──────────────────────────────────────────
            if checksum != _reference_checksum:
                return {
                    "combined_score": 0.0,
                    "artifacts": {
                        "checksum_mismatch": (
                            f"expected={_reference_checksum} got={checksum} — "
                            "the optimized expand_node produced different results. "
                            "Only semantics-preserving rewrites are accepted."
                        )
                    },
                }

            return {
                "combined_score": qps,
                "qps": qps,
                "checksum": checksum,
                "avg_query_ms": float(parsed.get("AVG_QUERY_MS", 0.0)),
                "avg_scanned_blocks": float(parsed.get("AVG_SCANNED_BLOCKS", 0.0)),
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
            # ── 7. Always restore the original source ─────────────────────────
            shutil.copy2(backup_path, GRAPH_SRC)
            os.unlink(backup_path)
