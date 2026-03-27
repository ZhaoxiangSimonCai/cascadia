"""Parallel wrapper for ``cascadia sequence``.

Dispatches one subprocess per mzML file using a process pool, each running
in an isolated temporary working directory to avoid temp-path collisions.

Usage
-----
    cascadia-parallel ~/data/mzml models/checkpoint.ckpt \\
        -o ~/data/embeddings -a weighted_mean -j 8

All flags not consumed by this wrapper (``-j``, ``--no_resume``) are forwarded
transparently to ``cascadia sequence``.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _get_mzml_files(input_path):
    """Return deduplicated list of .mzML files from *input_path*."""
    path = Path(input_path)
    if path.is_file() and path.suffix.lower() == ".mzml":
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.mzML")) + sorted(path.glob("*.mzml"))
        seen: set[str] = set()
        unique: list[Path] = []
        for f in files:
            low = f.name.lower()
            if low not in seen:
                seen.add(low)
                unique.append(f)
        return unique
    raise ValueError(f"Invalid input: {input_path} (must be .mzML file or directory)")


def _output_path_for(mzml_file, output_dir):
    """Return the expected output .h5 path for a given mzML file."""
    return Path(output_dir) / f"{Path(mzml_file).stem}_embeddings.h5"


def _get_pending_files(mzml_files, output_dir):
    """Return (pending, skipped_count) after filtering already-completed files."""
    pending: list[Path] = []
    skipped = 0
    for f in mzml_files:
        if _output_path_for(f, output_dir).exists():
            skipped += 1
        else:
            pending.append(f)
    return pending, skipped


def _run_one(mzml_file, model_path, output_dir, extra_args):
    """Run ``cascadia sequence`` for a single file in an isolated temp dir."""
    cascadia_bin = shutil.which("cascadia")
    if cascadia_bin is None:
        cascadia_bin = "cascadia"

    cmd = [
        cascadia_bin,
        "sequence",
        str(mzml_file),
        str(model_path),
        "-o",
        str(output_dir),
    ] + list(extra_args)

    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="cascadia_par_") as tmpdir:
        result = subprocess.run(
            cmd,
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.monotonic() - start
    return mzml_file, result.returncode, result.stdout, elapsed


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Parallel wrapper for cascadia sequence. "
            "Unrecognised arguments are forwarded to each cascadia subprocess."
        ),
    )
    parser.add_argument(
        "input_path",
        help="Directory of .mzML files (or a single .mzML file)",
    )
    parser.add_argument(
        "model",
        help="Path to trained Cascadia model checkpoint",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="cascadia_embeddings",
        help="Output directory for embeddings (default: cascadia_embeddings)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes (default: 4)",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Reprocess all files even if outputs already exist",
    )

    args, extra_args = parser.parse_known_args()

    if args.no_resume:
        extra_args.append("--no_resume")

    input_path = Path(args.input_path).resolve()
    model_path = Path(args.model).resolve()
    output_dir = Path(args.output_dir).resolve()

    mzml_files = _get_mzml_files(input_path)
    total_found = len(mzml_files)
    print(f"Found {total_found} .mzML file(s) in {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.no_resume:
        pending = mzml_files
        skipped = 0
    else:
        pending, skipped = _get_pending_files(mzml_files, output_dir)

    n_workers = min(args.workers, len(pending)) if pending else 0
    print(
        f"To process: {len(pending)} | Already done: {skipped} | Workers: {n_workers}"
    )

    if not pending:
        print("All files already processed. Use --no_resume to force reprocessing.")
        return

    if extra_args:
        print(f"Forwarding to cascadia: {' '.join(extra_args)}")

    total = len(pending)
    completed = 0
    failed = 0
    failed_files: list[Path] = []
    wall_start = time.monotonic()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _run_one, f, model_path, output_dir, extra_args
            ): f
            for f in pending
        }

        for future in as_completed(futures):
            mzml_file, returncode, output, elapsed = future.result()
            if returncode == 0:
                completed += 1
                tag = "OK"
            else:
                failed += 1
                failed_files.append(mzml_file)
                tag = "FAILED"

            print(
                f"[{completed + failed}/{total}] {tag}: {mzml_file.name} "
                f"({elapsed:.1f}s)"
            )

            if returncode != 0:
                for line in output.strip().splitlines()[-10:]:
                    print(f"  | {line}")

    wall_elapsed = time.monotonic() - wall_start
    print(f"\nFinished in {wall_elapsed:.1f}s — {completed} succeeded, {failed} failed")

    if failed_files:
        print("\nFailed files:")
        for f in failed_files:
            print(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
