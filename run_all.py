#!/usr/bin/env python3
"""Run the complete article reproducibility workflow in publication order."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
OUTPUTS_DIR = REPO_ROOT / "outputs"
IMAGES_DIR = REPO_ROOT / "images"
DATA_DIR = REPO_ROOT / "data" / "brehm"
MANIFEST_PATH = OUTPUTS_DIR / "run_manifest.json"

REQUIRED_PYTHON = "3.9.13"
REQUIRED_PACKAGE = "bnn-for-14c-calibration"
REQUIRED_PACKAGE_VERSION = "0.2.1"
DEFAULT_DPI = 600


@dataclass(frozen=True)
class Block:
    name: str
    script: str
    output_subdir: str
    default_seed: int
    figure_files: Sequence[str]


BLOCKS: Sequence[Block] = (
    Block(
        name="Figures 1–3",
        script="reproduce_fig1_to_fig3.py",
        output_subdir="outputs_fig1_to_fig3",
        default_seed=2022,
        figure_files=("Fig1.png", "Fig2.png", "Fig3.png"),
    ),
    Block(
        name="Figures 4–7",
        script="reproduce_fig4_to_fig7.py",
        output_subdir="outputs_fig4_to_fig7",
        default_seed=1234,
        figure_files=(
            "Fig4.png",
            "Fig5.png",
            "Fig6a.png",
            "Fig6b.png",
            "Fig7a.png",
            "Fig7b.png",
        ),
    ),
    Block(
        name="Figures 8–10 + Table 3",
        script="reproduce_fig8_to_fig10_table3.py",
        output_subdir="outputs_fig8_to_fig10_table3",
        default_seed=1234,
        figure_files=(
            "Fig8a.png",
            "Fig8b.png",
            "Fig9a.png",
            "Fig9b.png",
            "Fig10.png",
        ),
    ),
    Block(
        name="Figure 11",
        script="reproduce_fig11.py",
        output_subdir="outputs_fig11",
        default_seed=1234,
        figure_files=("Fig11.png",),
    ),
    Block(
        name="Figures 12–16 + Tables 4–5",
        script="reproduce_fig12_to_fig16_tables4_5.py",
        output_subdir="outputs_fig12_to_fig16_tables4_5",
        default_seed=1234,
        figure_files=(
            "Fig12.png",
            "Fig13.png",
            "Fig14.png",
            "Fig15.png",
            "Fig16.png",
        ),
    ),
)

FIGURE_GROUPS: Dict[str, Sequence[str]] = {
    "Figure 1": ("Fig1.png",),
    "Figure 2": ("Fig2.png",),
    "Figure 3": ("Fig3.png",),
    "Figure 4": ("Fig4.png",),
    "Figure 5": ("Fig5.png",),
    "Figure 6": ("Fig6a.png", "Fig6b.png"),
    "Figure 7": ("Fig7a.png", "Fig7b.png"),
    "Figure 8": ("Fig8a.png", "Fig8b.png"),
    "Figure 9": ("Fig9a.png", "Fig9b.png"),
    "Figure 10": ("Fig10.png",),
    "Figure 11": ("Fig11.png",),
    "Figure 12": ("Fig12.png",),
    "Figure 13": ("Fig13.png",),
    "Figure 14": ("Fig14.png",),
    "Figure 15": ("Fig15.png",),
    "Figure 16": ("Fig16.png",),
}

FIGURE_SOURCE_SUBDIR: Dict[str, str] = {}
for block in BLOCKS:
    for figure_file in block.figure_files:
        FIGURE_SOURCE_SUBDIR[figure_file] = block.output_subdir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional global seed override passed to every block. If omitted, "
            "each script keeps its validated default seed."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"PNG resolution in dots per inch (default: {DEFAULT_DPI}).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively as they are generated.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Run a structural smoke test without importing scientific "
            "dependencies or executing experiments."
        ),
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help=(
            "Do not clear managed images/output directories before the run. "
            "Fresh runs clean them by default to avoid stale artifacts."
        ),
    )
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version() -> str:
    try:
        return metadata.version(REQUIRED_PACKAGE)
    except metadata.PackageNotFoundError:
        return "not-installed"


def write_manifest(manifest: dict) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=False),
        encoding="utf-8",
    )


def clean_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def structural_check() -> List[str]:
    problems: List[str] = []
    for block in BLOCKS:
        script_path = SCRIPTS_DIR / block.script
        if not script_path.is_file():
            problems.append(f"Missing script: {relative(script_path)}")
    required_dirs = (OUTPUTS_DIR, IMAGES_DIR, DATA_DIR)
    for directory in required_dirs:
        if not directory.is_dir():
            problems.append(f"Missing directory: {relative(directory)}")
    requirements = REPO_ROOT / "requirements.txt"
    if not requirements.is_file():
        problems.append("Missing requirements.txt")
    elif requirements.read_text(encoding="utf-8").strip() != (
        f"{REQUIRED_PACKAGE}=={REQUIRED_PACKAGE_VERSION}"
    ):
        problems.append("requirements.txt does not contain the expected pinned package only")
    return problems


def preflight_environment() -> None:
    if sys.version.split()[0] != REQUIRED_PYTHON:
        raise RuntimeError(
            f"Python {REQUIRED_PYTHON} is required for the reference environment; "
            f"current interpreter is {sys.version.split()[0]}."
        )

    installed = package_version()
    if installed != REQUIRED_PACKAGE_VERSION:
        raise RuntimeError(
            f"{REQUIRED_PACKAGE}=={REQUIRED_PACKAGE_VERSION} is required; "
            f"installed version is {installed}."
        )

    missing = [
        DATA_DIR / "donnees_traites_fig_a.csv",
        DATA_DIR / "donnees_traites_fig_b.csv",
    ]
    missing = [path for path in missing if not path.is_file()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            f"Missing required Brehm input file(s): {names}. "
            "Place them in data/brehm/ before running the workflow."
        )


def collect_generated_files(output_dir: Path) -> List[dict]:
    records: List[dict] = []
    if not output_dir.exists():
        return records
    for path in sorted(p for p in output_dir.rglob("*") if p.is_file()):
        records.append(
            {
                "path": relative(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def sync_block_figures(block: Block) -> List[dict]:
    records: List[dict] = []
    source_dir = OUTPUTS_DIR / block.output_subdir
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for file_name in block.figure_files:
        source = source_dir / file_name
        if not source.is_file():
            raise FileNotFoundError(
                f"Expected figure was not generated: {relative(source)}"
            )
        destination = IMAGES_DIR / file_name
        shutil.copy2(source, destination)
        source_hash = sha256_file(source)
        destination_hash = sha256_file(destination)
        if source_hash != destination_hash:
            raise RuntimeError(
                f"SHA-256 mismatch after synchronizing {relative(destination)}"
            )
        records.append(
            {
                "image": relative(destination),
                "source": relative(source),
                "sha256": destination_hash,
                "sha256_verified_equal": True,
            }
        )
    return records


def build_final_figure_manifest() -> Dict[str, List[dict]]:
    final: Dict[str, List[dict]] = {}
    for figure_name, file_names in FIGURE_GROUPS.items():
        entries: List[dict] = []
        for file_name in file_names:
            image = IMAGES_DIR / file_name
            source = OUTPUTS_DIR / FIGURE_SOURCE_SUBDIR[file_name] / file_name
            if image.is_file() and source.is_file():
                image_hash = sha256_file(image)
                source_hash = sha256_file(source)
                entries.append(
                    {
                        "image": relative(image),
                        "source": relative(source),
                        "sha256": image_hash,
                        "sha256_verified_equal": image_hash == source_hash,
                    }
                )
        final[figure_name] = entries
    return final


def run_block(block: Block, args: argparse.Namespace) -> tuple:
    output_dir = OUTPUTS_DIR / block.output_subdir
    script_path = SCRIPTS_DIR / block.script

    command = [
        sys.executable,
        "-u",
        str(script_path),
        "--output-dir",
        str(output_dir),
        "--dpi",
        str(args.dpi),
    ]
    effective_seed = block.default_seed if args.seed is None else args.seed
    command.extend(["--seed", str(effective_seed)])
    if args.show:
        command.append("--show")

    public_command = [
        "python",
        f"scripts/{block.script}",
        "--output-dir",
        f"outputs/{block.output_subdir}",
        "--dpi",
        str(args.dpi),
        "--seed",
        str(effective_seed),
    ]
    if args.show:
        public_command.append("--show")

    print("\n" + "=" * 72, flush=True)
    print(f"START: {block.name}", flush=True)
    print(f"Script: scripts/{block.script}", flush=True)
    print(f"Seed: {effective_seed}", flush=True)
    print("=" * 72, flush=True)

    started = time.perf_counter()
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    elapsed = time.perf_counter() - started

    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, public_command)

    synced = sync_block_figures(block)
    print(f"SUCCESS: {block.name} ({elapsed:.2f} s)", flush=True)
    return elapsed, effective_seed, public_command, synced


def main() -> int:
    args = parse_args()

    problems = structural_check()
    if args.check_only:
        if problems:
            print("Structural smoke test: FAILED", flush=True)
            for problem in problems:
                print(f"- {problem}", flush=True)
            return 1
        print("Structural smoke test: OK", flush=True)
        print("Scientific calculations were not executed.", flush=True)
        return 0

    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr, flush=True)
        return 1

    try:
        preflight_environment()
    except Exception as exc:
        print(f"PRE-FLIGHT ERROR: {exc}", file=sys.stderr, flush=True)
        return 1

    if not args.no_clean:
        clean_directory(IMAGES_DIR)
        for block in BLOCKS:
            clean_directory(OUTPUTS_DIR / block.output_subdir)

    run_started_wall = now_utc()
    run_started = time.perf_counter()
    per_block_seed = {
        block.name: (block.default_seed if args.seed is None else args.seed)
        for block in BLOCKS
    }

    manifest = {
        "schema_version": 1,
        "run_status": "running",
        "started_at_utc": run_started_wall,
        "finished_at_utc": None,
        "total_elapsed_seconds": None,
        "python": {
            "version": sys.version.split()[0],
            "implementation": sys.implementation.name,
        },
        "bnn_for_14c_calibration_version": package_version(),
        "seed_configuration": {
            "global_override": args.seed,
            "policy": (
                "global override applied to all blocks"
                if args.seed is not None
                else "validated per-script defaults"
            ),
            "per_block": per_block_seed,
        },
        "dpi": args.dpi,
        "script_order": [block.script for block in BLOCKS],
        "blocks": [],
        "final_figures": {},
    }
    write_manifest(manifest)

    overall_failure: Optional[BaseException] = None

    for index, block in enumerate(BLOCKS, start=1):
        block_record = {
            "order": index,
            "name": block.name,
            "script": f"scripts/{block.script}",
            "script_sha256": sha256_file(SCRIPTS_DIR / block.script),
            "output_dir": f"outputs/{block.output_subdir}",
            "effective_seed": per_block_seed[block.name],
            "status": "running",
            "started_at_utc": now_utc(),
            "finished_at_utc": None,
            "elapsed_seconds": None,
            "command": None,
            "generated_files": [],
            "synced_figures": [],
            "error": None,
        }
        manifest["blocks"].append(block_record)
        write_manifest(manifest)

        try:
            elapsed, effective_seed, public_command, synced = run_block(block, args)
            block_record["effective_seed"] = effective_seed
            block_record["command"] = public_command
            block_record["status"] = "success"
            block_record["elapsed_seconds"] = elapsed
            block_record["finished_at_utc"] = now_utc()
            block_record["generated_files"] = collect_generated_files(
                OUTPUTS_DIR / block.output_subdir
            )
            block_record["synced_figures"] = synced
            manifest["final_figures"] = build_final_figure_manifest()
            write_manifest(manifest)
        except BaseException as exc:
            elapsed = time.perf_counter() - run_started
            block_record["status"] = "error"
            block_record["finished_at_utc"] = now_utc()
            block_record["generated_files"] = collect_generated_files(
                OUTPUTS_DIR / block.output_subdir
            )
            block_record["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            manifest["run_status"] = "error"
            manifest["finished_at_utc"] = now_utc()
            manifest["total_elapsed_seconds"] = elapsed
            manifest["final_figures"] = build_final_figure_manifest()
            write_manifest(manifest)
            print(f"ERROR: {block.name}: {exc}", file=sys.stderr, flush=True)
            overall_failure = exc
            break

    if overall_failure is None:
        total_elapsed = time.perf_counter() - run_started
        manifest["run_status"] = "success"
        manifest["finished_at_utc"] = now_utc()
        manifest["total_elapsed_seconds"] = total_elapsed
        manifest["final_figures"] = build_final_figure_manifest()
        write_manifest(manifest)
        print("\n" + "=" * 72, flush=True)
        print("ALL BLOCKS COMPLETED SUCCESSFULLY", flush=True)
        print(f"Total duration: {total_elapsed:.2f} s", flush=True)
        print("Final manuscript figures: images/", flush=True)
        print("Global audit manifest: outputs/run_manifest.json", flush=True)
        print("=" * 72, flush=True)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
