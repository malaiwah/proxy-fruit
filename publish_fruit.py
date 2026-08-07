#!/usr/bin/env python3
"""Atomically publish an authenticated Fruit pilot serving artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from huggingface_hub import CommitOperationAdd

DEFAULT_REPO = "malaiwah/GLM-5.2-SIQ-Fruit-pilot"
DEFAULT_BASE = Path("/mnt/vault/llm/fruit-pilot")
SOURCE = Path(__file__).resolve().parent
TOOLS = (
    "checkpoint_contract.py",
    "export_fruit.py",
    "fruit_serve_test.py",
    "fruit_serve_mtp.py",
    "parity_test.py",
)
LOGS = ("train.log", "serve-r25.log", "serve-r28.log")
MANIFEST_LINE = re.compile(r"([0-9a-f]{64})  (.+)")
MANIFEST_EXCLUSIONS = {".gitattributes", "MANIFEST.sha256", "README.md"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _required_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def validate_export_manifest(export: Path) -> dict[str, str]:
    """Validate manifest syntax, hashes, safe paths, and export inventory closure."""
    export = export.expanduser().resolve()
    if not export.is_dir():
        raise NotADirectoryError(f"pilot export is not a directory: {export}")

    manifest_path = _required_file(export / "MANIFEST.sha256", "export manifest")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        match = MANIFEST_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"malformed manifest line {line_number}: {line!r}")
        digest, name = match.groups()
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe manifest path on line {line_number}: {name!r}")
        if name in entries:
            raise ValueError(f"duplicate manifest path on line {line_number}: {name}")
        entries[name] = digest

    inventory = {
        path.relative_to(export).as_posix()
        for path in export.rglob("*")
        if path.is_file()
        and path.relative_to(export).as_posix() not in MANIFEST_EXCLUSIONS
    }
    declared = set(entries)
    if declared != inventory:
        missing = sorted(inventory - declared)
        absent = sorted(declared - inventory)
        raise ValueError(
            "manifest inventory does not close: "
            f"unmanifested={missing}, missing={absent}"
        )

    mismatches = {}
    for name, digest in entries.items():
        actual = _sha256(export / name)
        if actual != digest:
            mismatches[name] = {"declared": digest, "actual": actual}
    if mismatches:
        raise ValueError(f"manifest hash mismatch: {mismatches}")
    return entries


def build_commit_operations(
    export: Path,
    checkpoint: Path,
    run_assets: Path,
    source: Path = SOURCE,
) -> tuple[list[CommitOperationAdd], dict[str, object]]:
    """Build one complete, validated commit without mutating the Hub."""
    from huggingface_hub import CommitOperationAdd

    export = export.expanduser().resolve()
    checkpoint = _required_file(checkpoint, "pilot checkpoint")
    run_assets = run_assets.expanduser().resolve()
    source = source.expanduser().resolve()
    manifest = validate_export_manifest(export)

    files: dict[str, Path] = {
        path.relative_to(export).as_posix(): path
        for path in export.rglob("*")
        if path.is_file()
    }
    files["training/fruit_pilot.pt"] = checkpoint
    for name in TOOLS:
        files[f"tools/{name}"] = _required_file(source / name, f"tool {name}")
    for name in LOGS:
        files[f"logs/{name}"] = _required_file(run_assets / name, f"log {name}")

    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
        for name, path in sorted(files.items())
    ]
    report: dict[str, object] = {
        "export": str(export),
        "manifest_entries": len(manifest),
        "operations": len(operations),
        "checkpoint_sha256": _sha256(checkpoint),
        "files": {
            name: {"size": path.stat().st_size}
            for name, path in sorted(files.items())
        },
    }
    return operations, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--export",
        type=Path,
        default=Path(
            os.environ.get(
                "FRUIT_PILOT_EXPORT",
                DEFAULT_BASE / "pilot-repair" / "export",
            )
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            os.environ.get(
                "FRUIT_PILOT_CHECKPOINT",
                DEFAULT_BASE / "pilot-repair" / "training" / "fruit_pilot.pt",
            )
        ),
    )
    parser.add_argument(
        "--run-assets",
        type=Path,
        default=Path(os.environ.get("FRUIT_RUN_ASSETS", "~/fruit-pilot")),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the exact commit plan without publishing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    operations, report = build_commit_operations(
        args.export,
        args.checkpoint,
        args.run_assets,
    )
    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    from huggingface_hub import HfApi, get_token

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError(
            "Hugging Face authentication is required unless --dry-run is used"
        )
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=False)
    parent = api.model_info(args.repo).sha
    commit = api.create_commit(
        repo_id=args.repo,
        repo_type="model",
        operations=operations,
        commit_message="pilot: atomically publish authenticated corrected artifact",
        parent_commit=parent,
    )
    print(
        json.dumps(
            {
                "repo": args.repo,
                "parent": parent,
                "commit": commit.oid,
                "url": commit.commit_url,
                **report,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
