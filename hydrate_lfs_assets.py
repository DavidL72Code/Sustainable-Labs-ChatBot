#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import urllib.parse
import urllib.request
from pathlib import Path


LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as file_handle:
            return file_handle.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX
    except OSError:
        return False


def build_resolve_url(repo_type: str, repo_id: str, relative_path: str, revision: str) -> str:
    quoted_path = urllib.parse.quote(relative_path.replace("\\", "/"))
    return f"https://huggingface.co/{repo_type}/{repo_id}/resolve/{revision}/{quoted_path}"


def hydrate_file(path: Path, url: str, timeout_seconds: int) -> None:
    temporary_path = path.with_suffix(path.suffix + ".download")
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response, temporary_path.open("wb") as output_file:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output_file.write(chunk)
    temporary_path.replace(path)


def iter_target_files(base_dir: Path, targets: list[str]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        target_path = (base_dir / target).resolve()
        if not target_path.exists():
            continue
        if target_path.is_file():
            files.append(target_path)
            continue
        files.extend(sorted(path for path in target_path.rglob("*") if path.is_file()))
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replace Git LFS pointer files with resolved Hugging Face assets.")
    parser.add_argument("targets", nargs="+", help="Files or directories to scan relative to --base-dir.")
    parser.add_argument("--base-dir", default=".", help="Working tree root containing the target paths.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, for example owner/name.")
    parser.add_argument("--repo-type", default="spaces", choices=["spaces", "datasets", "models"])
    parser.add_argument("--revision", default="main", help="Git revision to resolve.")
    parser.add_argument("--timeout", type=int, default=300, help="Per-file download timeout in seconds.")
    parser.add_argument("--check", action="store_true", help="Only report pointer files without downloading them.")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    base_dir = Path(arguments.base_dir).resolve()
    target_files = iter_target_files(base_dir, arguments.targets)
    pointer_files: list[Path] = [path for path in target_files if is_lfs_pointer(path)]

    if not pointer_files:
        print("No Git LFS pointer files found.")
        return 0

    print(f"Found {len(pointer_files)} Git LFS pointer file(s).")
    for pointer_file in pointer_files:
        relative_path = pointer_file.relative_to(base_dir).as_posix()
        print(relative_path)
        if arguments.check:
            continue
        resolve_url = build_resolve_url(arguments.repo_type, arguments.repo_id, relative_path, arguments.revision)
        print(f"Hydrating {relative_path} from {resolve_url}", flush=True)
        try:
            hydrate_file(pointer_file, resolve_url, arguments.timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to hydrate {relative_path}: {exc}", file=sys.stderr)
            return 1

    if arguments.check:
        return 0

    print("Hydration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
