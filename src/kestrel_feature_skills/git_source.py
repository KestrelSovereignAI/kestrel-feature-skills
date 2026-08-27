"""Constrained git-backed source inspection and checkout."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlsplit

from .errors import GitInputError, GitSourceError, SkillNotFoundError
from .format import validate_skill_folder, validate_skill_name

_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")
_OBJECT_ID_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_URL_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
MAX_GIT_TRANSFER_BYTES = 32 * 1024 * 1024
MAX_GIT_TRANSFER_ENTRIES = 4096
MAX_GIT_OUTPUT_BYTES = 1024 * 1024
_TRANSFER_POLL_SECONDS = 0.05
_GIT_CONFIG_PREFIX = (
    "-c",
    "protocol.allow=never",
    "-c",
    "protocol.https.allow=always",
    "-c",
    "protocol.file.allow=never",
    "-c",
    "http.followRedirects=false",
)


def validate_remote_url(url: object) -> str:
    if not isinstance(url, str):
        raise GitInputError("git source URL must be a bounded HTTPS URL")
    try:
        encoded = url.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise GitInputError("git source URL must be valid UTF-8 text") from exc
    if len(encoded) > 2048 or _URL_CONTROL_RE.search(url):
        raise GitInputError(
            "git source URL must be bounded text without control characters"
        )
    try:
        parsed = urlsplit(url)
        # urllib deliberately defers malformed/out-of-range port checks until
        # this property is read. Validate it before Git turns caller input into
        # an upstream/tooling failure.
        _ = parsed.port
    except ValueError as exc:
        raise GitInputError("git source URL is malformed") from exc
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise GitInputError(
            "git source must use HTTPS without credentials, query parameters, or fragments"
        )
    return url


def validate_ref(ref: object) -> str:
    if not isinstance(ref, str):
        raise GitInputError("git ref contains unsupported characters")
    components = ref.split("/")
    if (
        not _REF_RE.fullmatch(ref)
        or ".." in ref
        or "//" in ref
        or ref.endswith(("/", "."))
        or any(
            not component
            or component.startswith(".")
            or component.endswith(".lock")
            for component in components
        )
    ):
        raise GitInputError("git ref contains unsupported characters")
    return ref


def is_full_object_id(value: object) -> bool:
    """Return whether ``value`` is a full SHA-1 or SHA-256 Git object ID."""

    return isinstance(value, str) and _OBJECT_ID_RE.fullmatch(value) is not None


def _tree_limit_exceeded(root: Path, *, max_bytes: int, max_entries: int) -> str | None:
    """Return the first checkout resource limit crossed below ``root``."""

    try:
        root.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise GitSourceError("could not measure bounded git checkout data") from exc
    total_bytes = 0
    total_entries = 0
    pending = [root]
    try:
        while pending:
            current = pending.pop()
            try:
                iterator = os.scandir(current)
            except FileNotFoundError:
                continue
            with iterator:
                for entry in iterator:
                    total_entries += 1
                    if total_entries > max_entries:
                        return "entries"
                    try:
                        metadata = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    total_bytes += metadata.st_size
                    if total_bytes > max_bytes:
                        return "bytes"
                    try:
                        is_directory = entry.is_dir(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    if is_directory:
                        pending.append(Path(entry.path))
    except OSError as exc:
        raise GitSourceError("could not measure bounded git checkout data") from exc
    return None


def _tree_limit_error(root: Path, *, max_bytes: int, max_entries: int) -> str | None:
    exceeded = _tree_limit_exceeded(root, max_bytes=max_bytes, max_entries=max_entries)
    if exceeded == "bytes":
        return f"git source exceeded the {max_bytes}-byte transfer limit"
    if exceeded == "entries":
        return f"git source exceeded the transfer entry limit of {max_entries}"
    return None


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (AttributeError, ProcessLookupError, PermissionError):
        process.kill()


def _git_failure_detail(stdout: str | None, stderr: str | None) -> str:
    return (
        (stderr or stdout or "git command failed").strip().splitlines()
        or ["git command failed"]
    )[-1][:500]


def _git_environment() -> dict[str, str]:
    """Return a noninteractive Git environment isolated from host Git config."""

    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "GCM_INTERACTIVE": "Never",
        }
    )
    return environment


def _bounded_output_size(*handles: BinaryIO) -> int:
    try:
        return sum(os.fstat(handle.fileno()).st_size for handle in handles)
    except OSError as exc:
        raise GitSourceError("could not measure bounded git command output") from exc


def _read_git_output(handle: BinaryIO) -> str:
    handle.seek(0)
    payload = handle.read(MAX_GIT_OUTPUT_BYTES + 1)
    return payload.decode("utf-8", errors="replace")


def _validate_sparse_tree_listing(listing: str) -> int | None:
    """Reject a sparse tree whose logical checkout cannot fit the hard bounds."""

    total_bytes = 0
    total_entries = 0
    object_id_length: int | None = None
    for raw_entry in listing.split("\x00"):
        if not raw_entry:
            continue
        try:
            metadata, path = raw_entry.split("\t", 1)
            mode, object_type, object_id, size_text = metadata.split()
        except ValueError as exc:
            raise GitSourceError(
                "git source returned malformed sparse tree metadata"
            ) from exc
        if (
            not re.fullmatch(r"[0-7]{6}", mode)
            or not is_full_object_id(object_id)
            or not path
        ):
            raise GitSourceError("git source returned malformed sparse tree metadata")
        if object_id_length is None:
            object_id_length = len(object_id)
        elif len(object_id) != object_id_length:
            raise GitSourceError(
                "git source returned mixed object formats in sparse tree metadata"
            )
        total_entries += 1
        if total_entries > MAX_GIT_TRANSFER_ENTRIES:
            raise GitSourceError(
                f"git source exceeded the transfer entry limit of {MAX_GIT_TRANSFER_ENTRIES}"
            )
        if object_type == "blob":
            try:
                size = int(size_text)
            except ValueError as exc:
                raise GitSourceError(
                    "git source returned malformed sparse blob metadata"
                ) from exc
            if size < 0:
                raise GitSourceError(
                    "git source returned malformed sparse blob metadata"
                )
            total_bytes += size
            if total_bytes > MAX_GIT_TRANSFER_BYTES:
                raise GitSourceError(
                    f"git source exceeded the {MAX_GIT_TRANSFER_BYTES}-byte transfer limit"
                )
        elif object_type == "commit":
            raise GitSourceError("git source skill folders cannot contain submodules")
        elif object_type != "tree" or size_text != "-":
            raise GitSourceError("git source returned malformed sparse tree metadata")
    return object_id_length


def _run_git(
    argv: list[str],
    *,
    timeout: int = 120,
    size_limit_root: Path | None = None,
    max_bytes: int | None = None,
    max_entries: int | None = None,
    cancel_event: threading.Event | None = None,
) -> str:
    limits = (max_bytes, max_entries)
    if size_limit_root is None:
        if any(limit is not None for limit in limits):
            raise ValueError("git limiting requires a root and positive bounds")
    elif any(
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        for limit in limits
    ):
        raise ValueError("git limiting requires a root and positive bounds")
    command = ["git", *_GIT_CONFIG_PREFIX, *argv]
    environment = _git_environment()
    if cancel_event is not None and cancel_event.is_set():
        raise GitSourceError("git source operation was cancelled")
    with (
        tempfile.TemporaryFile() as stdout_file,
        tempfile.TemporaryFile() as stderr_file,
    ):
        try:
            process = subprocess.Popen(
                command,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
                env=environment,
            )
        except FileNotFoundError as exc:
            raise GitSourceError("git executable is unavailable") from exc
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                if cancel_event is not None and cancel_event.is_set():
                    raise GitSourceError("git source operation was cancelled")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise GitSourceError("git source operation timed out")
                if (
                    _bounded_output_size(stdout_file, stderr_file)
                    > MAX_GIT_OUTPUT_BYTES
                ):
                    raise GitSourceError(
                        f"git source exceeded the {MAX_GIT_OUTPUT_BYTES}-byte output limit"
                    )
                limit_error = (
                    _tree_limit_error(
                        size_limit_root,
                        max_bytes=max_bytes,
                        max_entries=max_entries,
                    )
                    if size_limit_root is not None
                    and max_bytes is not None
                    and max_entries is not None
                    else None
                )
                if limit_error is not None:
                    raise GitSourceError(limit_error)
                time.sleep(min(_TRANSFER_POLL_SECONDS, max(remaining, 0)))
            if _bounded_output_size(stdout_file, stderr_file) > MAX_GIT_OUTPUT_BYTES:
                raise GitSourceError(
                    f"git source exceeded the {MAX_GIT_OUTPUT_BYTES}-byte output limit"
                )
            limit_error = (
                _tree_limit_error(
                    size_limit_root,
                    max_bytes=max_bytes,
                    max_entries=max_entries,
                )
                if size_limit_root is not None
                and max_bytes is not None
                and max_entries is not None
                else None
            )
            if limit_error is not None:
                raise GitSourceError(limit_error)
            stdout = _read_git_output(stdout_file)
            stderr = _read_git_output(stderr_file)
            if process.returncode:
                raise GitSourceError(
                    f"git source operation failed: {_git_failure_detail(stdout, stderr)}"
                )
            return stdout.strip()
        except BaseException:
            # Every failure after Popen—including a scanner or output-monitor
            # error—must end the detached process before the caller can release
            # its privacy lock or remove the checkout directory.
            if process.poll() is None:
                _kill_process_group(process)
            process.wait()
            raise


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
        self,
        *,
        url: str,
        ref: str,
        skill_name: str,
        target: Path,
        cancel_event: threading.Event | None = None,
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
            max_entries=MAX_GIT_TRANSFER_ENTRIES,
            cancel_event=cancel_event,
        )
        sparse_listing = _run_git(
            [
                "-C",
                str(target),
                "ls-tree",
                "-r",
                "-t",
                "-l",
                "-z",
                "HEAD",
                "--",
                skill_name,
                f"skills/{skill_name}",
            ],
            size_limit_root=target,
            max_bytes=MAX_GIT_TRANSFER_BYTES,
            max_entries=MAX_GIT_TRANSFER_ENTRIES,
            cancel_event=cancel_event,
        )
        listing_object_id_length = _validate_sparse_tree_listing(sparse_listing)
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
            max_entries=MAX_GIT_TRANSFER_ENTRIES,
            cancel_event=cancel_event,
        )
        _run_git(
            ["-C", str(target), "checkout", "--detach", "HEAD"],
            size_limit_root=target,
            max_bytes=MAX_GIT_TRANSFER_BYTES,
            max_entries=MAX_GIT_TRANSFER_ENTRIES,
            cancel_event=cancel_event,
        )
        revision = _run_git(
            ["-C", str(target), "rev-parse", "HEAD"],
            cancel_event=cancel_event,
        )
        if not is_full_object_id(revision) or (
            listing_object_id_length is not None
            and len(revision) != listing_object_id_length
        ):
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
        output = _run_git(
            ["ls-remote", "--exit-code", "--", url, ref, f"{ref}^{{}}"],
            timeout=60,
        )
        direct: list[str] = []
        peeled: list[str] = []
        object_id_length: int | None = None
        for line in output.splitlines():
            fields = line.split()
            if len(fields) != 2 or not is_full_object_id(fields[0]):
                raise GitSourceError(
                    f"remote ref {ref!r} returned an invalid commit identity"
                )
            if object_id_length is None:
                object_id_length = len(fields[0])
            elif len(fields[0]) != object_id_length:
                raise GitSourceError(
                    f"remote ref {ref!r} returned mixed object formats"
                )
            (peeled if fields[1].endswith("^{}") else direct).append(fields[0])
        revisions = peeled or direct
        if len(set(revisions)) != 1:
            raise GitSourceError(f"remote ref {ref!r} did not resolve to one commit")
        return revisions[0]

    def has_changed(self, *, url: str, ref: str, installed_revision: str) -> bool:
        if not is_full_object_id(installed_revision):
            raise GitSourceError("installed revision is not a full commit hash")
        return self.remote_revision(url=url, ref=ref) != installed_revision


__all__ = [
    "MAX_GIT_OUTPUT_BYTES",
    "MAX_GIT_TRANSFER_BYTES",
    "MAX_GIT_TRANSFER_ENTRIES",
    "GitCheckout",
    "GitSkillSource",
    "is_full_object_id",
    "validate_ref",
    "validate_remote_url",
]
