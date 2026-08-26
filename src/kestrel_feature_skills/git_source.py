"""Constrained git-backed source inspection and checkout."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .errors import GitSourceError, SkillNotFoundError
from .format import validate_skill_folder, validate_skill_name

_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


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


def _run_git(argv: list[str], *, timeout: int = 120) -> str:
    try:
        completed = subprocess.run(
            ["git", *argv],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise GitSourceError("git executable is unavailable") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitSourceError("git source operation timed out") from exc
    except subprocess.CalledProcessError as exc:
        detail = (
            (exc.stderr or exc.stdout or "git command failed").strip().splitlines()[-1]
        )
        raise GitSourceError(f"git source operation failed: {detail[:500]}") from exc
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
        _run_git(
            [
                "clone",
                "--depth",
                "1",
                "--branch",
                ref,
                "--single-branch",
                "--",
                url,
                str(target),
            ]
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
    "GitCheckout",
    "GitSkillSource",
    "validate_ref",
    "validate_remote_url",
]
