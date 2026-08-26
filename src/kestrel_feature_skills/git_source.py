"""Constrained git-backed source inspection and checkout."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .errors import GitSourceError, SkillNotFoundError
from .format import validate_skill_folder, validate_skill_name

_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
MAX_GIT_TRANSFER_BYTES = 32 * 1024 * 1024
_TRANSFER_POLL_SECONDS = 0.05


def validate_remote_url(url: object) -> str:
    if not isinstance(url, str) or len(url) > 2048:
        raise GitSourceError("git source URL must be a bounded HTTPS URL")
    parsed = urlsplit(url)
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise GitSourceError(
            "git source must use HTTPS without credentials, query parameters, or fragments"
        )
    return url


def validate_ref(ref: object) -> str:
    if not isinstance(ref, str) or not _REF_RE.fullmatch(ref) or ".." in ref:
        raise GitSourceError("git ref contains unsupported characters")
    return ref


def _tree_exceeds(root: Path, limit: int) -> bool:
    """Return as soon as checkout data below ``root`` crosses ``limit``."""

    if not root.exists():
        return False
    total = 0
    try:
        for current, directories, files in os.walk(root, followlinks=False):
            for name in (*directories, *files):
                try:
                    total += (Path(current) / name).lstat().st_size
                except FileNotFoundError:
                    continue
                if total > limit:
                    return True
    except OSError as exc:
        raise GitSourceError("could not measure bounded git checkout data") from exc
    return False


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (AttributeError, ProcessLookupError, PermissionError):
        process.kill()


def _git_failure_detail(stdout: str | None, stderr: str | None) -> str:
    return (
        (stderr or stdout or "git command failed").strip().splitlines()
        or ["git command failed"]
    )[-1][:500]


def _run_git(
    argv: list[str],
    *,
    timeout: int = 120,
    size_limit_root: Path | None = None,
    max_bytes: int | None = None,
) -> str:
    if (size_limit_root is None) != (max_bytes is None) or (
        max_bytes is not None and max_bytes < 1
    ):
        raise ValueError("git size limiting requires a root and a positive byte bound")
    command = ["git", *argv]
    environment = os.environ.copy()
    environment["GIT_LFS_SKIP_SMUDGE"] = "1"
    environment["GIT_TERMINAL_PROMPT"] = "0"
    if size_limit_root is not None and max_bytes is not None:
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                env=environment,
            )
        except FileNotFoundError as exc:
            raise GitSourceError("git executable is unavailable") from exc
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_process_group(process)
                process.communicate()
                raise GitSourceError("git source operation timed out")
            try:
                stdout, stderr = process.communicate(
                    timeout=min(_TRANSFER_POLL_SECONDS, remaining)
                )
            except subprocess.TimeoutExpired:
                if _tree_exceeds(size_limit_root, max_bytes):
                    _kill_process_group(process)
                    process.communicate()
                    raise GitSourceError(
                        f"git source exceeded the {max_bytes}-byte transfer limit"
                    )
                continue
            break
        if _tree_exceeds(size_limit_root, max_bytes):
            raise GitSourceError(
                f"git source exceeded the {max_bytes}-byte transfer limit"
            )
        if process.returncode:
            raise GitSourceError(
                f"git source operation failed: {_git_failure_detail(stdout, stderr)}"
            )
        return stdout.strip()
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
        )
    except FileNotFoundError as exc:
        raise GitSourceError("git executable is unavailable") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitSourceError("git source operation timed out") from exc
    except subprocess.CalledProcessError as exc:
        detail = _git_failure_detail(exc.stdout, exc.stderr)
        raise GitSourceError(f"git source operation failed: {detail}") from exc
    return completed.stdout.strip()


@dataclass(frozen=True, slots=True)
class GitCheckout:
    root: Path
    skill_folder: Path
    revision: str
    remote_url: str
    ref: str


class GitSkillSource:
    """Clone one immutable view of a remote repository into a caller-owned directory."""

    def checkout(
        self, *, url: str, ref: str, skill_name: str, target: Path
    ) -> GitCheckout:
        url = validate_remote_url(url)
        ref = validate_ref(ref)
        skill_name = validate_skill_name(skill_name)
        if target.exists():
            raise GitSourceError("git checkout target already exists")
        clone = [
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            "--no-checkout",
        ]
        if ref != "HEAD":
            clone.extend(["--branch", ref, "--single-branch"])
        clone.extend(["--", url, str(target)])
        _run_git(
            clone,
            size_limit_root=target,
            max_bytes=MAX_GIT_TRANSFER_BYTES,
        )
        sparse_paths = [f"/{skill_name}/", f"/skills/{skill_name}/"]
        _run_git(
            [
                "-C",
                str(target),
                "sparse-checkout",
                "set",
                "--no-cone",
                "--",
                *sparse_paths,
            ],
            size_limit_root=target,
            max_bytes=MAX_GIT_TRANSFER_BYTES,
        )
        _run_git(
            ["-C", str(target), "checkout", "--detach", "HEAD"],
            size_limit_root=target,
            max_bytes=MAX_GIT_TRANSFER_BYTES,
        )
        revision = _run_git(["-C", str(target), "rev-parse", "HEAD"])
        if not _COMMIT_RE.fullmatch(revision):
            raise GitSourceError("git checkout returned an invalid commit identity")
        candidates = (target / "skills" / skill_name, target / skill_name)
        folder = next(
            (candidate for candidate in candidates if candidate.is_dir()), None
        )
        if folder is None:
            raise SkillNotFoundError(
                f"remote source contains no {skill_name!r} folder at /skills or repository root"
            )
        validate_skill_folder(folder, source_root=folder.parent)
        return GitCheckout(
            root=target,
            skill_folder=folder,
            revision=revision,
            remote_url=url,
            ref=ref,
        )

    def remote_revision(self, *, url: str, ref: str) -> str:
        url = validate_remote_url(url)
        ref = validate_ref(ref)
        output = _run_git(["ls-remote", "--exit-code", "--", url, ref], timeout=60)
        first = output.splitlines()[0].split()[0] if output else ""
        if not _COMMIT_RE.fullmatch(first):
            raise GitSourceError(f"remote ref {ref!r} did not resolve to one commit")
        return first

    def has_changed(self, *, url: str, ref: str, installed_revision: str) -> bool:
        if not _COMMIT_RE.fullmatch(installed_revision):
            raise GitSourceError("installed revision is not a full commit hash")
        return self.remote_revision(url=url, ref=ref) != installed_revision


__all__ = [
    "MAX_GIT_TRANSFER_BYTES",
    "GitCheckout",
    "GitSkillSource",
    "validate_ref",
    "validate_remote_url",
]
