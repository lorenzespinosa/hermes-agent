"""Fail-closed macOS native real-profile browser supervision.

This module is deliberately separate from the legacy packaged-browser path.
Nothing here is selected unless ``browser.real_profile_macos_native`` is true.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import atexit
import contextlib
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import tempfile
import threading
import time
from typing import Mapping
import urllib.error
import urllib.request
from urllib.parse import urlparse
import uuid

from utils import is_truthy_value

try:
    import fcntl
except ImportError:  # pragma: no cover - native lane is macOS-only
    fcntl = None  # type: ignore[assignment]


STABLE_CHROME_EXECUTABLE = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
)
STABLE_CHROME_APP = "/Applications/Google Chrome.app"
STABLE_CHROME_REQUIREMENT = (
    'anchor apple generic and identifier "com.google.Chrome" and '
    'certificate leaf[subject.OU] = "EQHXZ8M8AV"'
)
_MANIFEST_NAME = ".hermes-native-manifest.json"
logger = logging.getLogger(__name__)
_NATIVE_OVERRIDE_KEYS = (
    "BROWSER_CDP_URL",
    "BU_CDP_URL",
    "BU_CDP_WS",
)
_NATIVE_INCOMPATIBLE_SELECTOR_KEYS = ("CAMOFOX_URL",)
_PROCESS_START_TIME_TOLERANCE = 0.01
_RUNTIME_LEASE_VERSION = 1
_RUNTIME_LEASE_MAX_BYTES = 16 * 1024
_MANIFEST_MAX_BYTES = 16 * 1024
_ACTIVE_PORT_MAX_BYTES = 1024
_SOURCE_STATE_MAX_BYTES = 64 * 1024 * 1024
_CDP_VERSION_MAX_BYTES = 64 * 1024
_CHROMIUM_PROCESS_NAME_MARKERS = (
    "chrome",
    "chromium",
    "brave",
    "edge",
    "vivaldi",
    "opera",
)


class NativeProfileError(RuntimeError):
    """A typed, safe-to-display native real-profile failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class NativeProfileRequest:
    source_browser: str
    source_profile: str
    expected_account: str


def native_intent(
    browser_config: Mapping[str, object],
    env: Mapping[str, str],
    *,
    system: str,
) -> tuple[bool, NativeProfileRequest | None]:
    """Select native intent first, then validate every prerequisite.

    A selected native lane never degrades to a legacy, cloud, or explicit-CDP
    path. The account value is retained only for equality checks and is never
    included in an error or diagnostic.
    """
    if not is_truthy_value(
        browser_config.get("real_profile_macos_native"), default=False
    ):
        return False, None
    if not is_truthy_value(browser_config.get("use_real_profile"), default=False):
        raise NativeProfileError(
            "native_consent_required",
            "Native real-profile browsing requires explicit real-profile consent.",
        )
    if system != "Darwin":
        raise NativeProfileError(
            "native_platform_unsupported",
            "Native real-profile browsing is available only on macOS.",
        )
    backend = str(browser_config.get("backend", "") or "").strip().lower()
    if backend != "browser-use":
        raise NativeProfileError(
            "native_backend_required",
            "Native real-profile browsing requires browser.backend: browser-use.",
        )
    if any(str(env.get(key, "") or "").strip() for key in _NATIVE_OVERRIDE_KEYS):
        raise NativeProfileError(
            "native_override_conflict",
            "Native real-profile browsing cannot run with a CDP override.",
        )
    if str(browser_config.get("cdp_url", "") or "").strip():
        raise NativeProfileError(
            "native_override_conflict",
            "Native real-profile browsing cannot run with a CDP override.",
        )
    if any(
        str(env.get(key, "") or "").strip()
        for key in _NATIVE_INCOMPATIBLE_SELECTOR_KEYS
    ):
        raise NativeProfileError(
            "native_selector_conflict",
            "Native real-profile browsing cannot run with another browser selector.",
        )
    ambient_engine = str(env.get("AGENT_BROWSER_ENGINE", "") or "").strip().lower()
    if ambient_engine not in {"", "auto", "chrome"}:
        raise NativeProfileError(
            "native_selector_conflict",
            "Native real-profile browsing cannot run with another browser selector.",
        )
    engine = str(browser_config.get("engine", "auto") or "auto").strip().lower()
    if engine == "lightpanda":
        raise NativeProfileError(
            "native_selector_conflict",
            "Native real-profile browsing cannot run with another browser selector.",
        )
    provider = str(browser_config.get("cloud_provider", "local") or "local").strip().lower()
    if provider not in {"", "local", "none"}:
        raise NativeProfileError(
            "native_local_only",
            "Native real-profile browsing is local-only; disable the cloud provider.",
        )
    source_browser = str(
        browser_config.get("real_profile_source_browser", "") or ""
    ).strip().lower()
    if source_browser != "chrome":
        raise NativeProfileError(
            "native_browser_unsupported",
            "Native real-profile browsing requires stable chrome explicitly.",
        )
    source_profile = str(
        browser_config.get("real_profile_source_profile", "") or ""
    ).strip()
    if not source_profile or source_profile in {".", ".."} or "/" in source_profile:
        raise NativeProfileError(
            "native_source_profile_missing",
            "Set an explicit Chrome source profile directory.",
        )
    expected_account = str(
        browser_config.get("real_profile_expected_account", "") or ""
    ).strip()
    if not expected_account:
        raise NativeProfileError(
            "native_expected_account_missing",
            "Set the expected account for native real-profile identity validation.",
        )
    if not is_truthy_value(browser_config.get("headed"), default=False):
        raise NativeProfileError(
            "native_headed_required",
            "Native real-profile acceptance currently requires headed mode.",
        )
    return True, NativeProfileRequest(
        source_browser=source_browser,
        source_profile=source_profile,
        expected_account=expected_account,
    )


def native_chrome_argv(executable: str, snapshot_dir: str) -> list[str]:
    """Return the frozen direct stable-Chrome launch vector."""
    return [
        executable,
        f"--user-data-dir={snapshot_dir}",
        "--profile-directory=Default",
        "--remote-debugging-address=127.0.0.1",
        "--remote-debugging-port=0",
        "--no-first-run",
        "--no-default-browser-check",
        "about:blank",
    ]


def _canonical_existing_dir(path: str, *, label: str) -> str:
    candidate = Path(path)
    if candidate.is_symlink():
        raise NativeProfileError("native_unsafe_path", f"The {label} cannot be a symlink.")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise NativeProfileError("native_missing_path", f"The {label} is unavailable.") from exc
    if not resolved.is_dir():
        raise NativeProfileError("native_unsafe_path", f"The {label} must be a directory.")
    if str(resolved) != os.path.abspath(path):
        raise NativeProfileError(
            "native_unsafe_path", f"The {label} contains a symlinked path component."
        )
    return str(resolved)


def _validated_source_identity_bytes(
    source_data_dir: str,
    source_profile: str,
    expected_account: str,
) -> bytes:
    """Validate source identity and return the exact bytes that were parsed."""
    source = _canonical_existing_dir(source_data_dir, label="source data directory")
    profile = Path(source, source_profile)
    if profile.is_symlink() or not profile.is_dir():
        raise NativeProfileError(
            "native_identity_mismatch", "The configured source profile identity is unavailable."
        )
    state_path = Path(source, "Local State")
    try:
        state_bytes = _read_private_file_exact(
            state_path,
            max_bytes=_SOURCE_STATE_MAX_BYTES,
            error_code="native_identity_mismatch",
        )
        state = json.loads(state_bytes.decode("utf-8"))
        cache = state["profile"]["info_cache"]
        selected = cache[source_profile]
    except NativeProfileError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise NativeProfileError(
            "native_identity_mismatch",
            "The configured source profile identity is ambiguous.",
        ) from exc
    if not isinstance(cache, dict) or not isinstance(selected, dict):
        raise NativeProfileError(
            "native_identity_mismatch",
            "The configured source profile identity is ambiguous.",
        )
    expected = expected_account.strip().casefold()
    fields = ("user_name", "account_name", "email")

    def account_values(record: object) -> set[str]:
        if not isinstance(record, dict):
            return set()
        return {
            str(record.get(field) or "").strip().casefold()
            for field in fields
            if str(record.get(field) or "").strip()
        }

    if expected not in account_values(selected):
        raise NativeProfileError(
            "native_identity_mismatch",
            "The configured source profile identity does not match.",
        )
    matching_profiles = [
        name for name, record in cache.items() if expected in account_values(record)
    ]
    if matching_profiles != [source_profile]:
        raise NativeProfileError(
            "native_identity_mismatch",
            "The configured source profile identity is conflicting.",
        )
    return state_bytes


def validate_source_identity(
    source_data_dir: str,
    source_profile: str,
    expected_account: str,
) -> None:
    """Require an explicit profile and exactly matching Chrome account record."""
    _validated_source_identity_bytes(
        source_data_dir, source_profile, expected_account
    )


def argv_owns_data_dir(
    argv: list[str] | tuple[str, ...],
    data_dir: str,
    *,
    implicit_default_executable: str | None = None,
) -> bool | None:
    """Return ownership proof, or ``None`` when Chromium argv is ambiguous."""
    wanted = os.path.normcase(os.path.realpath(data_dir))
    values: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--user-data-dir":
            if index + 1 >= len(argv):
                return None
            value = argv[index + 1]
            if not value or value.startswith("--"):
                return None
            values.append(value)
            index += 2
            continue
        if arg.startswith("--user-data-dir="):
            value = arg.split("=", 1)[1]
            if not value:
                return None
            values.append(value)
            index += 1
            continue
        if arg.startswith("--user-data-dir"):
            # Reject typo/concatenated variants rather than guessing how a
            # particular Chromium build will interpret them.
            return None
        index += 1

    if values:
        if len(values) != 1 or not os.path.isabs(values[0]):
            return None
        return os.path.normcase(os.path.realpath(values[0])) == wanted
    if not argv or not implicit_default_executable:
        return False
    return os.path.realpath(argv[0]) == os.path.realpath(implicit_default_executable)


def validate_codesign_output(output: str) -> None:
    """Validate the non-sensitive identity fields emitted by ``codesign -dv``."""
    required = (
        "Identifier=com.google.Chrome",
        "TeamIdentifier=EQHXZ8M8AV",
        'identifier "com.google.Chrome"',
        "EQHXZ8M8AV",
    )
    if not all(item in output for item in required):
        raise NativeProfileError(
            "native_signature_invalid", "Stable Chrome signature validation failed."
        )


def _validate_credential_stat(info: os.stat_result) -> None:
    """Validate security properties from a descriptor-bound file identity."""
    if not stat.S_ISREG(info.st_mode):
        raise NativeProfileError(
            "native_credential_type", "A credential input must be a regular file."
        )
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise NativeProfileError(
            "native_credential_owner", "A credential input has unsafe ownership."
        )
    if info.st_nlink != 1:
        raise NativeProfileError(
            "native_credential_hardlink", "A credential input cannot be hard-linked."
        )
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise NativeProfileError(
            "native_credential_permissions", "A credential input has unsafe permissions."
        )


def validate_credential_input(path: str) -> None:
    """Reject credential inputs that are links, foreign-owned, or too open."""
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise NativeProfileError(
            "native_credential_missing", "A required credential input is unavailable."
        ) from exc
    if stat.S_ISLNK(info.st_mode):
        raise NativeProfileError(
            "native_credential_symlink", "A credential input cannot be a symlink."
        )
    if os.path.normcase(os.path.realpath(path)) != os.path.normcase(
        os.path.abspath(path)
    ):
        raise NativeProfileError(
            "native_credential_symlink",
            "A credential input cannot contain a symlinked path component.",
        )
    _validate_credential_stat(info)


def _ensure_private_directory(path: Path) -> os.stat_result:
    """Create or validate one owner-only, non-symlinked supervisor directory."""
    if not os.path.lexists(path):
        try:
            os.mkdir(path, 0o700)
        except OSError as exc:
            raise NativeProfileError(
                "native_lock_unsafe",
                "The native supervisor directory could not be created safely.",
            ) from exc
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise NativeProfileError(
            "native_lock_unsafe", "The native supervisor directory is unavailable."
        ) from exc
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o700
        or (hasattr(os, "getuid") and info.st_uid != os.getuid())
        or os.path.realpath(path) != os.path.abspath(path)
    ):
        raise NativeProfileError(
            "native_lock_unsafe",
            "The native supervisor directory has unsafe identity.",
        )
    return info


def _validate_lock_file_stat(info: os.stat_result) -> None:
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_nlink != 1
        or (hasattr(os, "getuid") and info.st_uid != os.getuid())
    ):
        raise NativeProfileError(
            "native_lock_unsafe", "The native lock file has unsafe identity."
        )


class NativeProfileLock:
    """Non-blocking cross-process lock rooted outside the snapshot tree."""

    def __init__(self, supervisor_root: str):
        self.root = Path(supervisor_root)
        self._fd: int | None = None
        self._root_identity: tuple[int, int] | None = None
        self._lock_identity: tuple[int, int] | None = None

    def acquire(self) -> "NativeProfileLock":
        if fcntl is None:
            raise NativeProfileError(
                "native_lock_unavailable", "Native profile locking is unavailable."
            )
        parent_info = _ensure_private_directory(self.root.parent)
        root_info = _ensure_private_directory(self.root)
        # Rebind the parent after creating/validating the child so a rename of
        # browser-supervisor cannot silently retarget the lock root.
        confirmed_parent = os.lstat(self.root.parent)
        if (parent_info.st_dev, parent_info.st_ino) != (
            confirmed_parent.st_dev,
            confirmed_parent.st_ino,
        ):
            raise NativeProfileError(
                "native_lock_unsafe", "The native supervisor root changed identity."
            )
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        lock_path = self.root / "native.lock"
        before: os.stat_result | None = None
        if os.path.lexists(lock_path):
            try:
                before = os.lstat(lock_path)
                _validate_lock_file_stat(before)
            except OSError as exc:
                raise NativeProfileError(
                    "native_lock_unsafe", "The native lock file is unavailable."
                ) from exc
        fd: int | None = None
        try:
            fd = os.open(lock_path, flags, 0o600)
            opened = os.fstat(fd)
            after = os.lstat(lock_path)
            _validate_lock_file_stat(opened)
            _validate_lock_file_stat(after)
            if (opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino) or (
                before is not None
                and (before.st_dev, before.st_ino)
                != (opened.st_dev, opened.st_ino)
            ):
                raise NativeProfileError(
                    "native_lock_unsafe", "The native lock file changed identity."
                )
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            with contextlib.suppress(Exception):
                if fd is not None:
                    os.close(fd)
            raise NativeProfileError(
                "native_profile_busy", "The native profile is already in use."
            ) from exc
        except NativeProfileError:
            if fd is not None:
                os.close(fd)
            raise
        except OSError as exc:
            if fd is not None:
                os.close(fd)
            raise NativeProfileError(
                "native_lock_unsafe", "The native lock file could not be opened safely."
            ) from exc
        self._fd = fd
        self._root_identity = (root_info.st_dev, root_info.st_ino)
        self._lock_identity = (opened.st_dev, opened.st_ino)
        try:
            self.prove()
        except Exception:
            self.release()
            raise
        return self

    def prove(self) -> None:
        """Reprove that the held flock is still named by the frozen scope."""
        if self._fd is None or self._root_identity is None or self._lock_identity is None:
            raise NativeProfileError(
                "native_lock_unsafe", "The native lock is not held."
            )
        try:
            root_info = os.lstat(self.root)
            path_info = os.lstat(self.root / "native.lock")
            opened = os.fstat(self._fd)
            _validate_lock_file_stat(path_info)
            _validate_lock_file_stat(opened)
        except (OSError, NativeProfileError) as exc:
            if isinstance(exc, NativeProfileError):
                raise
            raise NativeProfileError(
                "native_lock_unsafe", "The native lock changed identity."
            ) from exc
        if (
            (root_info.st_dev, root_info.st_ino) != self._root_identity
            or (path_info.st_dev, path_info.st_ino) != self._lock_identity
            or (opened.st_dev, opened.st_ino) != self._lock_identity
        ):
            raise NativeProfileError(
                "native_lock_unsafe", "The native lock changed identity."
            )

    @property
    def identity(self) -> tuple[int, int]:
        self.prove()
        assert self._lock_identity is not None
        return self._lock_identity

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        self._root_identity = None
        self._lock_identity = None
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    def __enter__(self) -> "NativeProfileLock":
        return self.acquire()

    def __exit__(self, *_exc: object) -> None:
        self.release()


_NATIVE_PROFILE_FILES = (
    "Cookies",
    "Network/Cookies",
    "Login Data",
    "Login Data For Account",
    "Web Data",
    "Preferences",
)


def _open_nofollow_components(path: str, flags: int) -> int:
    """Open an absolute path without following any directory component."""
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise OSError("component-safe nofollow opens are unavailable")
    parts = Path(os.path.abspath(path)).parts
    if len(parts) < 2 or parts[0] != os.path.sep:
        raise OSError("credential path is not absolute")
    directory_fd = os.open(
        os.path.sep,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        for component in parts[1:-1]:
            next_fd = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        return os.open(
            parts[-1],
            flags | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
    finally:
        os.close(directory_fd)


def _open_validated_credential(path: str) -> int:
    """Open and validate the exact credential inode returned to the caller."""
    descriptor = _open_nofollow_components(path, os.O_RDONLY)
    try:
        _validate_credential_stat(os.fstat(descriptor))
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _read_private_file_exact(
    path: str | Path,
    *,
    max_bytes: int,
    error_code: str,
) -> bytes:
    """Read one owner-only file through a frozen descriptor and pathname identity."""
    pathname = os.path.abspath(os.fspath(path))
    descriptor: int | None = None
    try:
        before = os.lstat(pathname)
        _validate_credential_stat(before)
        if stat.S_IMODE(before.st_mode) != 0o600 or before.st_size > max_bytes:
            raise OSError("private file has unsafe mode or size")
        descriptor = _open_nofollow_components(pathname, os.O_RDONLY)
        opened = os.fstat(descriptor)
        _validate_credential_stat(opened)
        if (
            (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_size != before.st_size
            or opened.st_size > max_bytes
        ):
            raise OSError("private file changed before read")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(4096, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise OSError("private file grew during read")
        opened_after = os.fstat(descriptor)
        after = os.lstat(pathname)
        _validate_credential_stat(opened_after)
        _validate_credential_stat(after)
        if (
            total != opened.st_size
            or (opened_after.st_dev, opened_after.st_ino, opened_after.st_size)
            != (opened.st_dev, opened.st_ino, opened.st_size)
            or (after.st_dev, after.st_ino, after.st_size)
            != (opened.st_dev, opened.st_ino, opened.st_size)
            or stat.S_IMODE(after.st_mode) != 0o600
        ):
            raise OSError("private file changed during read")
        return b"".join(chunks)
    except (OSError, NativeProfileError) as exc:
        raise NativeProfileError(
            error_code, "A native supervisor file has unsafe identity."
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_active_port_lines(snapshot: Path) -> list[str]:
    try:
        return _read_private_file_exact(
            snapshot / "DevToolsActivePort",
            max_bytes=_ACTIVE_PORT_MAX_BYTES,
            error_code="native_active_port_unsafe",
        ).decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise NativeProfileError(
            "native_active_port_unsafe",
            "The native DevTools port record is invalid.",
        ) from exc


def _copy_native_file(source: str, destination: str) -> None:
    """Copy one already-validated credential file without following links."""
    Path(destination).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    source_fd: int | None = None
    try:
        before = os.lstat(source)
        _validate_credential_stat(before)
        source_fd = _open_validated_credential(source)
        opened = os.fstat(source_fd)
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise NativeProfileError(
                "native_credential_changed",
                "A credential input changed identity before it was copied.",
            )
    except NativeProfileError:
        if source_fd is not None:
            os.close(source_fd)
        raise
    except OSError as exc:
        if source_fd is not None:
            os.close(source_fd)
        raise NativeProfileError(
            "native_credential_changed",
            "A credential input changed identity before it was copied.",
        ) from exc
    try:
        _copy_native_descriptor(source_fd, destination)
        opened_after = os.fstat(source_fd)
        after = os.lstat(source)
        if (
            (opened_after.st_dev, opened_after.st_ino, opened_after.st_size)
            != (opened.st_dev, opened.st_ino, opened.st_size)
            or (after.st_dev, after.st_ino, after.st_size)
            != (opened.st_dev, opened.st_ino, opened.st_size)
        ):
            raise NativeProfileError(
                "native_credential_changed",
                "A credential input changed identity while it was copied.",
            )
        _validate_credential_stat(opened_after)
        _validate_credential_stat(after)
    except OSError as exc:
        raise NativeProfileError(
            "native_credential_changed",
            "A credential input changed identity while it was copied.",
        ) from exc
    finally:
        os.close(source_fd)


def _copy_native_descriptor(source_fd: int, destination: str) -> None:
    """Copy from one already-open, descriptor-validated credential inode."""
    source_before = os.fstat(source_fd)
    _validate_credential_stat(source_before)
    os.lseek(source_fd, 0, os.SEEK_SET)
    Path(destination).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    destination_fd = os.open(
        destination,
        flags,
        0o600,
    )
    try:
        copied = 0
        while copied < source_before.st_size:
            chunk = os.read(source_fd, min(1024 * 1024, source_before.st_size - copied))
            if not chunk:
                raise NativeProfileError(
                    "native_credential_changed",
                    "A credential input changed size while it was copied.",
                )
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise OSError("credential snapshot write was incomplete")
                view = view[written:]
            copied += len(chunk)
        if os.read(source_fd, 1):
            raise NativeProfileError(
                "native_credential_changed",
                "A credential input grew while it was copied.",
            )
        source_after = os.fstat(source_fd)
        if (
            source_after.st_dev,
            source_after.st_ino,
            source_after.st_size,
        ) != (
            source_before.st_dev,
            source_before.st_ino,
            source_before.st_size,
        ):
            raise NativeProfileError(
                "native_credential_changed",
                "A credential input changed while it was copied.",
            )
        os.fsync(destination_fd)
    finally:
        os.close(destination_fd)


def _copy_native_bytes(source: bytes, destination: str) -> None:
    """Persist already-validated credential bytes without reopening the source."""
    Path(destination).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination_fd = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        view = memoryview(source)
        while view:
            written = os.write(destination_fd, view)
            if written <= 0:
                raise OSError("credential snapshot write was incomplete")
            view = view[written:]
        os.fsync(destination_fd)
    finally:
        os.close(destination_fd)


def _fsync_tree(root: Path) -> None:
    for current, dirs, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in files:
            path = current_path / name
            with open(path, "rb") as handle:
                os.fsync(handle.fileno())
        for name in dirs:
            os.chmod(current_path / name, 0o700)
        directory_fd = os.open(current_path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def provision_native_snapshot(
    request: NativeProfileRequest,
    source_data_dir: str,
    destination: str,
    *,
    executable_fingerprint: str = "",
) -> dict[str, object]:
    """Build, validate, fsync, and atomically publish a native snapshot."""
    source = _canonical_existing_dir(source_data_dir, label="source data directory")
    destination_path = Path(destination)
    parent = destination_path.parent
    if parent.is_symlink() or destination_path.is_symlink():
        raise NativeProfileError(
            "native_snapshot_unsafe", "The native snapshot path cannot be a symlink."
        )
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(parent, 0o700)
    backup = parent / f".{destination_path.name}.previous-{uuid.uuid4().hex}"
    published = False
    local_state_bytes = _validated_source_identity_bytes(
        source, request.source_profile, request.expected_account
    )
    build: Path | None = None
    try:
        build = Path(
            tempfile.mkdtemp(prefix=f".{destination_path.name}.build-", dir=parent)
        )
        os.chmod(build, 0o700)
        default_dir = build / "Default"
        default_dir.mkdir(mode=0o700)
        _copy_native_bytes(local_state_bytes, str(build / "Local State"))
        copied = 0
        for relative in _NATIVE_PROFILE_FILES:
            source_file = Path(source, request.source_profile, relative)
            if not source_file.exists():
                continue
            validate_credential_input(str(source_file))
            _copy_native_file(str(source_file), str(default_dir / relative))
            copied += 1
        if copied == 0:
            raise NativeProfileError(
                "native_snapshot_empty", "No native profile credential inputs were available."
            )
        manifest: dict[str, object] = {
            "version": 1,
            "snapshot_uuid": uuid.uuid4().hex,
            "source_profile_hash": hashlib.sha256(
                request.source_profile.encode("utf-8")
            ).hexdigest(),
            "expected_account_hash": hashlib.sha256(
                request.expected_account.casefold().encode("utf-8")
            ).hexdigest(),
            "executable_fingerprint": executable_fingerprint,
        }
        manifest_path = build / _MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.chmod(manifest_path, 0o600)
        _fsync_tree(build)
        if _processes_owning_data_dir(
            source, implicit_default_executable=STABLE_CHROME_EXECUTABLE
        ):
            raise NativeProfileError(
                "native_source_raced",
                "Work Chrome opened while provisioning; the unpublished snapshot was discarded.",
            )
        if _processes_owning_data_dir(str(destination_path)):
            raise NativeProfileError(
                "native_snapshot_raced",
                "A browser acquired the native snapshot during provisioning; publish was refused.",
            )
        if destination_path.exists():
            os.replace(destination_path, backup)
        try:
            os.replace(build, destination_path)
            published = True
            parent_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except Exception:
            if backup.exists() and not destination_path.exists():
                os.replace(backup, destination_path)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return manifest
    except NativeProfileError:
        raise
    except Exception as exc:
        raise NativeProfileError(
            "native_snapshot_publish_failed", "Native snapshot publish failed."
        ) from exc
    finally:
        if not published and build is not None and build.exists():
            shutil.rmtree(build, ignore_errors=True)


@dataclass
class _NativeRuntime:
    process: object
    process_start_time: float
    cdp_url: str
    cdp_port: int
    snapshot_uuid: str
    executable_fingerprint: str
    lock: NativeProfileLock
    hermes_home: str = ""
    snapshot_path: str = ""
    supervisor_path: str = ""
    source_profile_hash: str = ""
    expected_account_hash: str = ""
    executable_path: str = STABLE_CHROME_EXECUTABLE
    lease_path: str = ""
    runtime_generation: str = ""
    lock_device: int = 0
    lock_inode: int = 0


@dataclass(frozen=True)
class NativeProfileClient:
    """One generation-bound client of a profile-scoped native Chrome."""

    hermes_home: str
    token: str
    cdp_url: str
    runtime_namespace: str
    inactivity_delay: float


class _PidHandle:
    """Minimal process reference reconstructed from a durable runtime lease."""

    def __init__(self, pid: int, start_time: float):
        self.pid = pid
        self._start_time = start_time

    def create_time(self) -> float:
        return self._start_time


_state_guard = threading.RLock()
_runtime_guards: dict[str, threading.RLock] = {}
_runtimes: dict[str, _NativeRuntime] = {}
_cleanup_timers: dict[str, threading.Timer] = {}
_client_leases: dict[str, set[str]] = {}


def _runtime_guard_for(hermes_home: str) -> threading.RLock:
    with _state_guard:
        return _runtime_guards.setdefault(hermes_home, threading.RLock())


def _profile_scope(home: str | Path | None = None) -> tuple[str, Path, Path]:
    from hermes_constants import hermes_home_key

    canonical_home = hermes_home_key(home)
    supervisor = Path(canonical_home) / "browser-supervisor" / "native-real-profile"
    return (
        canonical_home,
        supervisor / "snapshot",
        supervisor,
    )


def _profile_paths() -> tuple[Path, Path]:
    _home, snapshot, supervisor = _profile_scope()
    return snapshot, supervisor


def _validate_stable_chrome() -> str:
    executable = Path(STABLE_CHROME_EXECUTABLE)
    if executable.is_symlink():
        raise NativeProfileError(
            "native_executable_unsafe", "Stable Chrome executable cannot be a symlink."
        )
    try:
        resolved = str(executable.resolve(strict=True))
    except OSError as exc:
        raise NativeProfileError(
            "native_executable_missing", "Stable Chrome is not installed in /Applications."
        ) from exc
    if resolved != STABLE_CHROME_EXECUTABLE:
        raise NativeProfileError(
            "native_executable_unsafe", "Stable Chrome resolved to an unexpected path."
        )
    try:
        verify = subprocess.run(
            [
                "/usr/bin/codesign",
                "--verify",
                "--deep",
                "--strict",
                f"-R={STABLE_CHROME_REQUIREMENT}",
                STABLE_CHROME_APP,
            ],
            capture_output=True,
            timeout=20,
        )
        details = subprocess.run(
            [
                "/usr/bin/codesign", "-dv", "--verbose=4", "--requirements", "-",
                STABLE_CHROME_APP,
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise NativeProfileError(
            "native_signature_unavailable", "Stable Chrome signature validation failed."
        ) from exc
    if verify.returncode != 0 or details.returncode != 0:
        raise NativeProfileError(
            "native_signature_invalid", "Stable Chrome signature validation failed."
        )
    output = f"{details.stdout or ''}\n{details.stderr or ''}"
    validate_codesign_output(output)
    info = executable.stat()
    return hashlib.sha256(
        f"{resolved}:{info.st_ino}:{info.st_size}:{info.st_mtime_ns}:{output}".encode()
    ).hexdigest()


def _security_check_guest_requirement(pid: int, requirement: str) -> bool:
    """Evaluate a compiled requirement against the Security.framework PID guest."""
    import ctypes

    security = ctypes.CDLL(
        "/System/Library/Frameworks/Security.framework/Security"
    )
    core_foundation = ctypes.CDLL(
        "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
    )

    void_pointer = ctypes.c_void_p
    core_foundation.CFStringCreateWithCString.argtypes = (
        void_pointer,
        ctypes.c_char_p,
        ctypes.c_uint32,
    )
    core_foundation.CFStringCreateWithCString.restype = void_pointer
    core_foundation.CFNumberCreate.argtypes = (
        void_pointer,
        ctypes.c_int,
        void_pointer,
    )
    core_foundation.CFNumberCreate.restype = void_pointer
    core_foundation.CFDictionaryCreate.argtypes = (
        void_pointer,
        ctypes.POINTER(void_pointer),
        ctypes.POINTER(void_pointer),
        ctypes.c_long,
        void_pointer,
        void_pointer,
    )
    core_foundation.CFDictionaryCreate.restype = void_pointer
    core_foundation.CFRelease.argtypes = (void_pointer,)
    core_foundation.CFRelease.restype = None

    security.SecRequirementCreateWithString.argtypes = (
        void_pointer,
        ctypes.c_uint32,
        ctypes.POINTER(void_pointer),
    )
    security.SecRequirementCreateWithString.restype = ctypes.c_int32
    security.SecCodeCopyGuestWithAttributes.argtypes = (
        void_pointer,
        void_pointer,
        ctypes.c_uint32,
        ctypes.POINTER(void_pointer),
    )
    security.SecCodeCopyGuestWithAttributes.restype = ctypes.c_int32
    security.SecCodeCheckValidity.argtypes = (
        void_pointer,
        ctypes.c_uint32,
        void_pointer,
    )
    security.SecCodeCheckValidity.restype = ctypes.c_int32

    created: list[int] = []
    guest = void_pointer()
    compiled_requirement = void_pointer()
    try:
        requirement_string = core_foundation.CFStringCreateWithCString(
            None,
            requirement.encode("utf-8"),
            0x08000100,  # kCFStringEncodingUTF8
        )
        if not requirement_string:
            raise OSError("could not create requirement string")
        created.append(requirement_string)

        pid_value = ctypes.c_int(pid)
        pid_number = core_foundation.CFNumberCreate(
            None,
            9,  # kCFNumberIntType
            ctypes.byref(pid_value),
        )
        if not pid_number:
            raise OSError("could not create PID attribute")
        created.append(pid_number)

        pid_key = void_pointer.in_dll(security, "kSecGuestAttributePid").value
        if not pid_key:
            raise OSError("PID guest attribute is unavailable")
        keys = (void_pointer * 1)(pid_key)
        values = (void_pointer * 1)(pid_number)
        attributes = core_foundation.CFDictionaryCreate(
            None, keys, values, 1, None, None
        )
        if not attributes:
            raise OSError("could not create guest attributes")
        created.append(attributes)

        status = security.SecRequirementCreateWithString(
            requirement_string,
            0,
            ctypes.byref(compiled_requirement),
        )
        if status != 0 or not compiled_requirement.value:
            raise OSError("could not compile fixed code requirement")

        status = security.SecCodeCopyGuestWithAttributes(
            None,
            attributes,
            0,
            ctypes.byref(guest),
        )
        if status != 0 or not guest.value:
            return False
        return security.SecCodeCheckValidity(
            guest,
            0,
            compiled_requirement,
        ) == 0
    finally:
        if guest.value:
            core_foundation.CFRelease(guest)
        if compiled_requirement.value:
            core_foundation.CFRelease(compiled_requirement)
        for reference in reversed(created):
            core_foundation.CFRelease(reference)


def _validate_live_process_signature(pid: int) -> None:
    """Apply the fixed Google requirement to the live Security guest."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise NativeProfileError(
            "native_live_signature_invalid",
            "Stable Chrome live-process signature validation failed.",
        )
    try:
        valid = _security_check_guest_requirement(pid, STABLE_CHROME_REQUIREMENT)
    except Exception as exc:
        raise NativeProfileError(
            "native_live_signature_unavailable",
            "Stable Chrome live-process signature validation was unavailable.",
        ) from exc
    if not valid:
        raise NativeProfileError(
            "native_live_signature_invalid",
            "Stable Chrome live-process signature validation failed.",
        )


def _processes_owning_data_dir(
    data_dir: str,
    *,
    implicit_default_executable: str | None = None,
) -> list[object]:
    try:
        import psutil
    except ImportError as exc:
        raise NativeProfileError(
            "native_process_proof_unavailable", "Process ownership proof is unavailable."
        ) from exc
    owners: list[object] = []
    for process in psutil.process_iter(["pid", "name"]):
        try:
            info = process.info
            raw_name = info.get("name") if isinstance(info, Mapping) else None
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise NativeProfileError(
                    "native_process_proof_unavailable",
                    "Process ownership proof is unavailable.",
                )
            name = raw_name.casefold()
            if not any(marker in name for marker in _CHROMIUM_PROCESS_NAME_MARKERS):
                continue
            listed_pid = info.get("pid")
            identity = _process_identity(process)
            if (
                isinstance(listed_pid, bool)
                or not isinstance(listed_pid, int)
                or listed_pid <= 0
                or identity is None
                or identity[0] != listed_pid
            ):
                raise NativeProfileError(
                    "native_process_proof_unavailable",
                    "Process ownership proof is unavailable.",
                )
            argv = process.cmdline()
            if (
                not isinstance(argv, (list, tuple))
                or not argv
                or any(not isinstance(arg, str) for arg in argv)
                or not argv[0]
            ):
                raise NativeProfileError(
                    "native_process_proof_unavailable",
                    "Process ownership proof is unavailable.",
                )
            resolved_identity = _process_identity(psutil.Process(identity[0]))
            if resolved_identity != identity:
                raise NativeProfileError(
                    "native_process_proof_unavailable",
                    "Process ownership proof is unavailable.",
                )
            ownership = argv_owns_data_dir(
                argv,
                data_dir,
                implicit_default_executable=implicit_default_executable,
            )
            if ownership is not False:
                has_explicit_data_dir = any(
                    arg == "--user-data-dir"
                    or arg.startswith("--user-data-dir=")
                    for arg in argv
                )
                if (
                    implicit_default_executable
                    and not has_explicit_data_dir
                    and not _process_executable_matches(
                        process, implicit_default_executable
                    )
                ):
                    continue
                owners.append(process)
        except NativeProfileError:
            raise
        except (psutil.AccessDenied, OSError) as exc:
            raise NativeProfileError(
                "native_process_proof_unavailable",
                "Process ownership proof is unavailable.",
            ) from exc
        except psutil.NoSuchProcess as exc:
            raise NativeProfileError(
                "native_process_proof_unavailable",
                "Process ownership proof is unavailable.",
            ) from exc
    return owners


def _manifest_for(
    request: NativeProfileRequest,
    snapshot: Path,
    executable_fingerprint: str,
) -> dict[str, object] | None:
    manifest_path = snapshot / _MANIFEST_NAME
    try:
        manifest = json.loads(
            _read_private_file_exact(
                manifest_path,
                max_bytes=_MANIFEST_MAX_BYTES,
                error_code="native_manifest_unsafe",
            ).decode("utf-8")
        )
    except (NativeProfileError, OSError, ValueError, TypeError):
        return None
    expected = {
        "source_profile_hash": hashlib.sha256(request.source_profile.encode()).hexdigest(),
        "expected_account_hash": hashlib.sha256(
            request.expected_account.casefold().encode()
        ).hexdigest(),
        "executable_fingerprint": executable_fingerprint,
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"version", "snapshot_uuid", *expected}
        or isinstance(manifest.get("version"), bool)
        or manifest.get("version") != 1
        or any(manifest.get(k) != v for k, v in expected.items())
    ):
        return None
    if not isinstance(manifest.get("snapshot_uuid"), str) or not manifest["snapshot_uuid"]:
        return None
    try:
        canonical = _canonical_existing_dir(str(snapshot), label="native snapshot")
        info = os.lstat(canonical)
        if (
            canonical != str(snapshot)
            or not stat.S_ISDIR(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o700
            or (hasattr(os, "getuid") and info.st_uid != os.getuid())
        ):
            return None
        copied = 0
        for relative in _NATIVE_PROFILE_FILES:
            credential = snapshot / "Default" / relative
            if credential.exists():
                validate_credential_input(str(credential))
                copied += 1
        if copied == 0:
            return None
    except (OSError, NativeProfileError):
        return None
    return manifest


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        raise urllib.error.HTTPError(req.full_url, code, "redirect rejected", headers, fp)


def _validated_cdp_websocket(
    snapshot: Path, port: int, *, timeout: float = 1.0
) -> str | None:
    """Return the exact loopback browser websocket only after CDP identity proof."""
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        return None
    try:
        lines = _read_active_port_lines(snapshot)
        if (
            len(lines) != 2
            or int(lines[0]) != port
            or not lines[1].startswith("/devtools/browser/")
        ):
            return None
        active_websocket_path = lines[1]
    except (NativeProfileError, OSError, ValueError):
        return None
    url = f"http://127.0.0.1:{port}/json/version"
    try:
        opener = urllib.request.build_opener(_NoRedirect)
        with opener.open(url, timeout=timeout) as response:
            if response.geturl() != url:
                return None
            response_bytes = response.read(_CDP_VERSION_MAX_BYTES + 1)
            if len(response_bytes) > _CDP_VERSION_MAX_BYTES:
                return None
            payload = json.loads(response_bytes.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    websocket_url = str(payload.get("webSocketDebuggerUrl") or "")
    parsed = urlparse(websocket_url)
    if (
        parsed.scheme != "ws"
        or parsed.hostname != "127.0.0.1"
        or parsed.port != port
        or parsed.path != active_websocket_path
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        return None
    if not str(payload.get("Browser") or "").startswith("Chrome/"):
        return None
    return websocket_url


def _probe_cdp(snapshot: Path, port: int, *, timeout: float = 1.0) -> bool:
    return _validated_cdp_websocket(snapshot, port, timeout=timeout) is not None


def _process_identity(process: object) -> tuple[int, float] | None:
    try:
        pid = process.pid
        start_time = process.create_time()
    except Exception:
        return None
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or isinstance(start_time, bool)
        or not isinstance(start_time, (int, float))
        or not math.isfinite(start_time)
        or start_time < 0
    ):
        return None
    return pid, float(start_time)


def _process_start_times_match(actual: object, expected: object) -> bool:
    return (
        not isinstance(actual, bool)
        and isinstance(actual, (int, float))
        and math.isfinite(actual)
        and actual >= 0
        and not isinstance(expected, bool)
        and isinstance(expected, (int, float))
        and math.isfinite(expected)
        and expected >= 0
        and abs(actual - expected) <= _PROCESS_START_TIME_TOLERANCE
    )


def _process_identity_matches(
    process: object,
    expected_pid: object,
    expected_start_time: object,
) -> bool:
    identity = _process_identity(process)
    return (
        identity is not None
        and not isinstance(expected_pid, bool)
        and isinstance(expected_pid, int)
        and expected_pid > 0
        and identity[0] == expected_pid
        and _process_start_times_match(identity[1], expected_start_time)
    )


def _process_executable_matches(process: object, expected: str) -> bool:
    """Bind process identity to the kernel-reported executable, never argv[0]."""
    try:
        executable = process.exe()
    except Exception:
        return False
    return isinstance(executable, str) and os.path.realpath(executable) == os.path.realpath(
        expected
    )


def _parse_lsof_listener_owners(output: bytes, port: int) -> tuple[int, ...] | None:
    """Parse macOS ``lsof -F0pcfnPT`` loopback listener owner records."""
    try:
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            return None
        text = output.decode("utf-8", errors="strict")
        if not text or not text.endswith("\n"):
            return None
        lines = text[:-1].split("\n")
        if not lines or any(not line or not line.endswith("\0") for line in lines):
            return None

        owners: list[int] = []
        seen_pids: set[int] = set()
        seen_fds: set[int] = set()
        current_pid: int | None = None
        current_has_listener = False
        for line in lines:
            fields = line[:-1].split("\0")
            if not fields or any(not field for field in fields):
                return None
            if fields[0].startswith("p"):
                if current_pid is not None and not current_has_listener:
                    return None
                process_values: dict[str, str] = {}
                for field in fields:
                    tag, value = field[0], field[1:]
                    if tag not in {"p", "c"} or tag in process_values or not value:
                        return None
                    process_values[tag] = value
                if set(process_values) != {"p", "c"}:
                    return None
                if not process_values["p"].isdecimal():
                    return None
                current_pid = int(process_values["p"])
                if current_pid <= 0 or current_pid in seen_pids:
                    return None
                seen_pids.add(current_pid)
                owners.append(current_pid)
                seen_fds = set()
                current_has_listener = False
                continue
            if not fields[0].startswith("f") or current_pid is None:
                return None

            file_values: dict[str, str] = {}
            tcp_values: dict[str, str] = {}
            for field in fields:
                tag, value = field[0], field[1:]
                if tag in {"f", "P", "n"}:
                    if tag in file_values or not value:
                        return None
                    file_values[tag] = value
                    continue
                if tag != "T" or "=" not in value:
                    return None
                tcp_key, tcp_value = value.split("=", 1)
                if not tcp_key or not tcp_value or tcp_key in tcp_values:
                    return None
                tcp_values[tcp_key] = tcp_value
            if set(file_values) != {"f", "P", "n"}:
                return None
            if not file_values["f"].isdecimal():
                return None
            fd = int(file_values["f"])
            if fd in seen_fds or file_values["P"] != "TCP":
                return None
            seen_fds.add(fd)
            if file_values["n"] not in {
                f"127.0.0.1:{port}",
                f"[::1]:{port}",
            }:
                return None
            if tcp_values.get("ST") != "LISTEN":
                return None
            current_has_listener = True
        if current_pid is None or not current_has_listener or not owners:
            return None
    except (AttributeError, UnicodeDecodeError, ValueError):
        return None
    return tuple(owners)


def _query_lsof_listener_owners(port: int) -> tuple[int, ...] | None:
    completed = subprocess.run(
        [
            "/usr/sbin/lsof",
            "-nP",
            f"-iTCP:{port}",
            "-sTCP:LISTEN",
            "-F0pcfnPT",
        ],
        capture_output=True,
        text=False,
        timeout=5,
        check=False,
        env={"LC_ALL": "C", "LANG": "C"},
    )
    if not isinstance(completed.stdout, bytes) or completed.stderr != b"":
        return None
    if completed.returncode == 1 and completed.stdout == b"":
        return ()
    if completed.returncode != 0:
        return None
    return _parse_lsof_listener_owners(completed.stdout, port)


def _recorded_listener_is_absent(port: int) -> bool:
    """Prove no listener remains for a persisted ready-runtime port."""
    if port == 0:
        return True
    return _query_lsof_listener_owners(port) == ()


def _listener_is_loopback_only(
    pid: int, port: int, expected_root_start_time: object
) -> bool:
    """Prove exact-port lsof listeners belong to the recorded Chrome tree."""
    try:
        import psutil

        if (
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 0
            or isinstance(port, bool)
            or not isinstance(port, int)
            or not 1 <= port <= 65535
        ):
            return False
        root = psutil.Process(pid)
        root_identity = _process_identity(root)
        if (
            root_identity is None
            or root_identity[0] != pid
            or not _process_start_times_match(
                root_identity[1], expected_root_start_time
            )
        ):
            return False
        owner_pids = _query_lsof_listener_owners(port)
        if not owner_pids:
            return False

        owner_identities: dict[int, tuple[int, float]] = {}
        for owner_pid in owner_pids:
            owner = psutil.Process(owner_pid)
            owner_identity = _process_identity(owner)
            if owner_identity is None or owner_identity[0] != owner_pid:
                return False
            if owner_pid == pid:
                if not _process_start_times_match(
                    owner_identity[1], expected_root_start_time
                ):
                    return False
            else:
                ancestors = owner.parents()
                if not isinstance(ancestors, (list, tuple)):
                    return False
                seen_ancestor_pids: set[int] = set()
                found_root = False
                for ancestor in ancestors:
                    ancestor_identity = _process_identity(ancestor)
                    if (
                        ancestor_identity is None
                        or ancestor_identity[0] in seen_ancestor_pids
                    ):
                        return False
                    seen_ancestor_pids.add(ancestor_identity[0])
                    if ancestor_identity[0] != pid:
                        continue
                    if not _process_start_times_match(
                        ancestor_identity[1], expected_root_start_time
                    ):
                        return False
                    found_root = True
                    break
                if not found_root:
                    return False
            owner_identities[owner_pid] = owner_identity

        for owner_pid, owner_identity in owner_identities.items():
            if not _process_identity_matches(
                psutil.Process(owner_pid), owner_pid, owner_identity[1]
            ):
                return False
        confirmed_owner_pids = _query_lsof_listener_owners(port)
        if (
            confirmed_owner_pids is None
            or frozenset(confirmed_owner_pids) != frozenset(owner_pids)
        ):
            return False

        # The second lsof result is a fresh observation, so every PID in it
        # must be recreated and rebound.  Set equality alone is insufficient:
        # a listener can exit/reparent and the kernel can reuse the same PID
        # between the ancestry proof above and this query.
        for owner_pid in confirmed_owner_pids:
            expected_identity = owner_identities.get(owner_pid)
            if expected_identity is None:
                return False
            owner = psutil.Process(owner_pid)
            if not _process_identity_matches(
                owner, owner_pid, expected_identity[1]
            ):
                return False
            if owner_pid == pid:
                if not _process_start_times_match(
                    expected_identity[1], expected_root_start_time
                ):
                    return False
                continue
            ancestors = owner.parents()
            if not isinstance(ancestors, (list, tuple)):
                return False
            seen_ancestor_pids: set[int] = set()
            found_root = False
            for ancestor in ancestors:
                ancestor_identity = _process_identity(ancestor)
                if (
                    ancestor_identity is None
                    or ancestor_identity[0] in seen_ancestor_pids
                ):
                    return False
                seen_ancestor_pids.add(ancestor_identity[0])
                if ancestor_identity[0] != pid:
                    continue
                if not _process_start_times_match(
                    ancestor_identity[1], expected_root_start_time
                ):
                    return False
                found_root = True
                break
            if not found_root:
                return False
        return _process_identity_matches(
            psutil.Process(pid), pid, expected_root_start_time
        )
    except Exception:
        return False


def _runtime_is_valid(
    runtime: _NativeRuntime,
    request: NativeProfileRequest,
    snapshot: Path,
    executable_fingerprint: str,
) -> bool:
    def invalid(reason: str) -> bool:
        logger.info("native runtime proof failed [%s]", reason)
        return False

    if runtime.executable_fingerprint != executable_fingerprint:
        return invalid("executable_fingerprint")
    manifest = _manifest_for(request, snapshot, executable_fingerprint)
    if not manifest or manifest.get("snapshot_uuid") != runtime.snapshot_uuid:
        return invalid("manifest")
    try:
        import psutil

        process = psutil.Process(runtime.process.pid)
        if not _process_identity_matches(
            process,
            runtime.process.pid,
            runtime.process_start_time,
        ):
            return invalid("root_identity_before")
        if not _process_executable_matches(process, STABLE_CHROME_EXECUTABLE):
            return invalid("root_executable_before")
        _validate_live_process_signature(process.pid)
        argv = process.cmdline()
        if not argv_owns_data_dir(argv, str(snapshot)):
            return invalid("root_data_dir_before")
    except Exception:
        return invalid("root_inspection_before")
    if not _probe_cdp(snapshot, runtime.cdp_port):
        return invalid("cdp_probe")
    if not _listener_is_loopback_only(
        runtime.process.pid, runtime.cdp_port, runtime.process_start_time
    ):
        return invalid("listener_proof")
    try:
        process = psutil.Process(runtime.process.pid)
        if not _process_identity_matches(
            process,
            runtime.process.pid,
            runtime.process_start_time,
        ):
            return invalid("root_identity_after")
        if not _process_executable_matches(process, STABLE_CHROME_EXECUTABLE):
            return invalid("root_executable_after")
        _validate_live_process_signature(process.pid)
        argv = process.cmdline()
        if not argv_owns_data_dir(argv, str(snapshot)):
            return invalid("root_data_dir_after")
    except Exception:
        return invalid("root_inspection_after")
    return True


def _runtime_paths_are_immutable(runtime: _NativeRuntime) -> bool:
    """Require runtime paths to remain bound to one canonical Hermes home."""
    if not runtime.hermes_home or not runtime.snapshot_path or not runtime.supervisor_path:
        return False
    canonical_home, snapshot, supervisor = _profile_scope(runtime.hermes_home)
    try:
        lock_identity = runtime.lock.identity
    except Exception:
        return False
    return (
        runtime.hermes_home == canonical_home
        and runtime.snapshot_path == str(snapshot)
        and runtime.supervisor_path == str(supervisor)
        and runtime.lease_path == str(supervisor / "runtime.json")
        and lock_identity == (runtime.lock_device, runtime.lock_inode)
    )


def _runtime_lease_payload(runtime: _NativeRuntime) -> dict[str, object]:
    """Serialize only immutable, redacted runtime identity fields."""
    if not _runtime_paths_are_immutable(runtime):
        raise NativeProfileError(
            "native_runtime_lease_invalid",
            "Native runtime lease paths are not immutable.",
        )
    try:
        pid = runtime.process.pid
    except Exception as exc:
        raise NativeProfileError(
            "native_runtime_lease_invalid",
            "Native runtime lease process identity is invalid.",
        ) from exc
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or isinstance(runtime.process_start_time, bool)
        or not isinstance(runtime.process_start_time, (int, float))
        or not math.isfinite(runtime.process_start_time)
        or runtime.process_start_time < 0
        or isinstance(runtime.cdp_port, bool)
        or not isinstance(runtime.cdp_port, int)
        or runtime.cdp_port < 0
        or runtime.cdp_port > 65535
        or runtime.executable_path != STABLE_CHROME_EXECUTABLE
        or isinstance(runtime.lock_device, bool)
        or not isinstance(runtime.lock_device, int)
        or runtime.lock_device <= 0
        or isinstance(runtime.lock_inode, bool)
        or not isinstance(runtime.lock_inode, int)
        or runtime.lock_inode <= 0
    ):
        raise NativeProfileError(
            "native_runtime_lease_invalid",
            "Native runtime lease identity is invalid.",
        )
    values = (
        runtime.snapshot_uuid,
        runtime.executable_fingerprint,
        runtime.source_profile_hash,
        runtime.expected_account_hash,
        runtime.runtime_generation,
    )
    if any(not isinstance(value, str) or not value for value in values):
        raise NativeProfileError(
            "native_runtime_lease_invalid",
            "Native runtime lease identity is incomplete.",
        )
    expected_cdp = (
        f"http://127.0.0.1:{runtime.cdp_port}" if runtime.cdp_port else ""
    )
    if runtime.cdp_url != expected_cdp:
        raise NativeProfileError(
            "native_runtime_lease_invalid",
            "Native runtime lease endpoint is invalid.",
        )
    return {
        "version": _RUNTIME_LEASE_VERSION,
        "state": "ready" if runtime.cdp_port else "launching",
        "hermes_home": runtime.hermes_home,
        "snapshot_path": runtime.snapshot_path,
        "supervisor_path": runtime.supervisor_path,
        "lease_path": runtime.lease_path,
        "pid": pid,
        "process_start_time": runtime.process_start_time,
        "cdp_url": runtime.cdp_url,
        "cdp_port": runtime.cdp_port,
        "snapshot_uuid": runtime.snapshot_uuid,
        "executable_fingerprint": runtime.executable_fingerprint,
        "source_profile_hash": runtime.source_profile_hash,
        "expected_account_hash": runtime.expected_account_hash,
        "executable_path": runtime.executable_path,
        "runtime_generation": runtime.runtime_generation,
        "lock_device": runtime.lock_device,
        "lock_inode": runtime.lock_inode,
    }


def _validate_runtime_lease_file(path: Path) -> os.stat_result:
    """Open a lease nofollow and prove exact inode, owner, type, and mode."""
    try:
        before = os.lstat(path)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise OSError("runtime lease is not a regular file")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags)
        try:
            after = os.fstat(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        raise NativeProfileError(
            "native_runtime_lease_unsafe",
            "Native runtime lease has unsafe identity.",
        ) from exc
    if (
        (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        or not stat.S_ISREG(after.st_mode)
        or stat.S_IMODE(after.st_mode) != 0o600
        or before.st_nlink != 1
        or after.st_nlink != 1
        or (hasattr(os, "getuid") and after.st_uid != os.getuid())
    ):
        raise NativeProfileError(
            "native_runtime_lease_unsafe",
            "Native runtime lease has unsafe ownership or permissions.",
        )
    return after


def _validate_runtime_supervisor(runtime: _NativeRuntime) -> Path:
    supervisor = Path(runtime.supervisor_path)
    try:
        canonical = _canonical_existing_dir(
            runtime.supervisor_path, label="native supervisor"
        )
        info = os.lstat(canonical)
    except (NativeProfileError, OSError) as exc:
        raise NativeProfileError(
            "native_runtime_lease_unsafe",
            "Native runtime supervisor has unsafe identity.",
        ) from exc
    if (
        canonical != runtime.supervisor_path
        or stat.S_IMODE(info.st_mode) != 0o700
        or (hasattr(os, "getuid") and info.st_uid != os.getuid())
    ):
        raise NativeProfileError(
            "native_runtime_lease_unsafe",
            "Native runtime supervisor has unsafe ownership or permissions.",
        )
    return supervisor


def _write_runtime_lease(runtime: _NativeRuntime) -> None:
    """Atomically persist an owner-only, nofollow exact-runtime lease."""
    runtime.lock.prove()
    payload = _runtime_lease_payload(runtime)
    supervisor = _validate_runtime_supervisor(runtime)
    lease_path = Path(runtime.lease_path)
    if os.path.lexists(lease_path):
        _validate_runtime_lease_file(lease_path)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if len(encoded) > _RUNTIME_LEASE_MAX_BYTES:
        raise NativeProfileError(
            "native_runtime_lease_invalid", "Native runtime lease is too large."
        )
    temporary = supervisor / f".runtime-{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd: int | None = None
    try:
        fd = os.open(temporary, flags, 0o600)
        os.fchmod(fd, 0o600)
        offset = 0
        while offset < len(encoded):
            written = os.write(fd, encoded[offset:])
            if written <= 0:
                raise OSError("short runtime lease write")
            offset += written
        os.fsync(fd)
        os.close(fd)
        fd = None
        _validate_runtime_supervisor(runtime)
        runtime.lock.prove()
        if os.path.lexists(lease_path):
            _validate_runtime_lease_file(lease_path)
        os.replace(temporary, lease_path)
        _validate_runtime_lease_file(lease_path)
        runtime.lock.prove()
        directory_fd = os.open(supervisor, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except NativeProfileError:
        raise
    except OSError as exc:
        raise NativeProfileError(
            "native_runtime_lease_write_failed",
            "Native runtime lease could not be persisted safely.",
        ) from exc
    finally:
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(temporary)


def _read_runtime_lease(
    hermes_home: str,
    snapshot: Path,
    supervisor: Path,
    lock: NativeProfileLock,
) -> _NativeRuntime | None:
    """Read and validate one nofollow lease bound to the supplied scope."""
    lock.prove()
    lease_path = supervisor / "runtime.json"
    if not os.path.lexists(lease_path):
        return None
    try:
        encoded = _read_private_file_exact(
            lease_path,
            max_bytes=_RUNTIME_LEASE_MAX_BYTES,
            error_code="native_runtime_lease_unsafe",
        )
        lock.prove()
        payload = json.loads(encoded.decode("utf-8"))
    except NativeProfileError:
        raise
    except (OSError, UnicodeDecodeError, ValueError, TypeError) as exc:
        raise NativeProfileError(
            "native_runtime_lease_unsafe",
            "Native runtime lease could not be read with exact identity.",
        ) from exc
    expected_keys = {
        "version",
        "state",
        "hermes_home",
        "snapshot_path",
        "supervisor_path",
        "lease_path",
        "pid",
        "process_start_time",
        "cdp_url",
        "cdp_port",
        "snapshot_uuid",
        "executable_fingerprint",
        "source_profile_hash",
        "expected_account_hash",
        "executable_path",
        "runtime_generation",
        "lock_device",
        "lock_inode",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise NativeProfileError(
            "native_runtime_lease_invalid", "Native runtime lease schema is invalid."
        )
    pid = payload["pid"]
    start_time = payload["process_start_time"]
    port = payload["cdp_port"]
    lease_state = payload["state"]
    lock_device = payload["lock_device"]
    lock_inode = payload["lock_inode"]
    strings = {
        key: payload[key]
        for key in (
            "hermes_home",
            "snapshot_path",
            "supervisor_path",
            "lease_path",
            "cdp_url",
            "snapshot_uuid",
            "executable_fingerprint",
            "source_profile_hash",
            "expected_account_hash",
            "executable_path",
            "runtime_generation",
        )
    }
    if (
        isinstance(payload["version"], bool)
        or not isinstance(payload["version"], int)
        or payload["version"] != _RUNTIME_LEASE_VERSION
        or isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or isinstance(start_time, bool)
        or not isinstance(start_time, (int, float))
        or not math.isfinite(start_time)
        or start_time < 0
        or isinstance(port, bool)
        or not isinstance(port, int)
        or not 0 <= port <= 65535
        or not isinstance(lease_state, str)
        or lease_state != ("ready" if port else "launching")
        or isinstance(lock_device, bool)
        or not isinstance(lock_device, int)
        or lock_device <= 0
        or isinstance(lock_inode, bool)
        or not isinstance(lock_inode, int)
        or lock_inode <= 0
        or lock.identity != (lock_device, lock_inode)
        or any(not isinstance(value, str) for value in strings.values())
        or any(
            not strings[key]
            for key in (
                "snapshot_uuid",
                "executable_fingerprint",
                "source_profile_hash",
                "expected_account_hash",
                "runtime_generation",
            )
        )
        or strings["hermes_home"] != hermes_home
        or strings["snapshot_path"] != str(snapshot)
        or strings["supervisor_path"] != str(supervisor)
        or strings["lease_path"] != str(lease_path)
        or strings["executable_path"] != STABLE_CHROME_EXECUTABLE
        or strings["cdp_url"]
        != (f"http://127.0.0.1:{port}" if port else "")
    ):
        raise NativeProfileError(
            "native_runtime_lease_invalid", "Native runtime lease identity is invalid."
        )
    runtime = _NativeRuntime(
        process=_PidHandle(pid, float(start_time)),
        process_start_time=float(start_time),
        cdp_url=strings["cdp_url"],
        cdp_port=port,
        snapshot_uuid=strings["snapshot_uuid"],
        executable_fingerprint=strings["executable_fingerprint"],
        lock=lock,
        hermes_home=hermes_home,
        snapshot_path=str(snapshot),
        supervisor_path=str(supervisor),
        source_profile_hash=strings["source_profile_hash"],
        expected_account_hash=strings["expected_account_hash"],
        executable_path=strings["executable_path"],
        lease_path=str(lease_path),
        runtime_generation=strings["runtime_generation"],
        lock_device=lock_device,
        lock_inode=lock_inode,
    )
    if _runtime_lease_payload(runtime) != payload:
        raise NativeProfileError(
            "native_runtime_lease_invalid", "Native runtime lease did not round-trip."
        )
    return runtime


def _remove_runtime_lease(runtime: _NativeRuntime) -> None:
    """Remove only a secure lease whose complete payload matches runtime."""
    lease_path = Path(runtime.lease_path)
    runtime.lock.prove()
    if not os.path.lexists(lease_path):
        return
    loaded = _read_runtime_lease(
        runtime.hermes_home,
        Path(runtime.snapshot_path),
        Path(runtime.supervisor_path),
        runtime.lock,
    )
    if loaded is None or _runtime_lease_payload(loaded) != _runtime_lease_payload(runtime):
        raise NativeProfileError(
            "native_runtime_lease_unsafe",
            "Native runtime lease changed before removal.",
        )
    _validate_runtime_lease_file(lease_path)
    runtime.lock.prove()
    try:
        os.unlink(lease_path)
        runtime.lock.prove()
        directory_fd = os.open(runtime.supervisor_path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise NativeProfileError(
            "native_runtime_lease_remove_failed",
            "Native runtime lease could not be removed safely.",
        ) from exc


def _runtime_manifest_matches(runtime: _NativeRuntime) -> bool:
    if not _runtime_paths_are_immutable(runtime):
        return False
    snapshot = Path(runtime.snapshot_path)
    try:
        if _canonical_existing_dir(
            runtime.snapshot_path, label="native snapshot"
        ) != runtime.snapshot_path:
            return False
        snapshot_info = os.lstat(snapshot)
        if (
            not stat.S_ISDIR(snapshot_info.st_mode)
            or stat.S_IMODE(snapshot_info.st_mode) != 0o700
            or (hasattr(os, "getuid") and snapshot_info.st_uid != os.getuid())
        ):
            return False
        manifest = json.loads(
            _read_private_file_exact(
                snapshot / _MANIFEST_NAME,
                max_bytes=_MANIFEST_MAX_BYTES,
                error_code="native_manifest_unsafe",
            ).decode("utf-8")
        )
        copied = 0
        for relative in _NATIVE_PROFILE_FILES:
            credential = snapshot / "Default" / relative
            if credential.exists():
                validate_credential_input(str(credential))
                copied += 1
        if copied == 0:
            return False
    except (NativeProfileError, OSError, ValueError, TypeError):
        return False
    expected = {
        "snapshot_uuid": runtime.snapshot_uuid,
        "executable_fingerprint": runtime.executable_fingerprint,
        "source_profile_hash": runtime.source_profile_hash,
        "expected_account_hash": runtime.expected_account_hash,
    }
    return (
        isinstance(manifest, dict)
        and set(manifest) == {"version", *expected}
        and not isinstance(manifest.get("version"), bool)
        and manifest.get("version") == 1
        and all(
            isinstance(value, str)
            and bool(value)
            and manifest.get(key) == value
            for key, value in expected.items()
        )
    )


def _runtime_root_process(runtime: _NativeRuntime):
    """Return a freshly proven psutil root process, or ``None`` on ambiguity."""
    try:
        import psutil

        process = psutil.Process(runtime.process.pid)
        if not _process_identity_matches(
            process, runtime.process.pid, runtime.process_start_time
        ):
            return None
        if runtime.executable_path != STABLE_CHROME_EXECUTABLE:
            return None
        if not _process_executable_matches(process, runtime.executable_path):
            return None
        _validate_live_process_signature(process.pid)
        argv = process.cmdline()
        if (
            not isinstance(argv, (list, tuple))
            or not argv
            or any(not isinstance(arg, str) for arg in argv)
            or not argv_owns_data_dir(argv, runtime.snapshot_path)
        ):
            return None
        return process
    except Exception:
        return None


def _prove_recorded_runtime(runtime: _NativeRuntime) -> str:
    """Freshly prove every immutable runtime binding and return its WS URL.

    This proof is intentionally repeated immediately before each destructive
    action.  Any unreadable or changing component is ambiguity, not absence.
    """

    def reject() -> NativeProfileError:
        return NativeProfileError(
            "native_cleanup_identity_ambiguous",
            "Native cleanup retained state because exact runtime identity could not be proven.",
        )

    if not _runtime_manifest_matches(runtime):
        raise reject()
    try:
        if _validate_stable_chrome() != runtime.executable_fingerprint:
            raise reject()
    except NativeProfileError:
        raise reject()
    if _runtime_root_process(runtime) is None:
        raise reject()
    websocket_url = _validated_cdp_websocket(
        Path(runtime.snapshot_path), runtime.cdp_port
    )
    if websocket_url is None:
        raise reject()
    if not _listener_is_loopback_only(
        runtime.process.pid, runtime.cdp_port, runtime.process_start_time
    ):
        raise reject()
    if _runtime_root_process(runtime) is None:
        raise reject()
    try:
        if _validate_stable_chrome() != runtime.executable_fingerprint:
            raise reject()
    except NativeProfileError:
        raise reject()
    if not _runtime_manifest_matches(runtime):
        raise reject()
    final_websocket_url = _validated_cdp_websocket(
        Path(runtime.snapshot_path), runtime.cdp_port
    )
    if final_websocket_url != websocket_url:
        raise reject()
    return final_websocket_url


def _native_startup_timeout(value: object) -> float:
    """Return one finite positive startup wait capped independently of tool time."""
    try:
        timeout = float(str(value))
    except (TypeError, ValueError) as exc:
        raise NativeProfileError(
            "native_command_timeout_invalid",
            "Native startup timeout must be a finite positive number.",
        ) from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise NativeProfileError(
            "native_command_timeout_invalid",
            "Native startup timeout must be a finite positive number.",
        )
    return min(timeout, 120.0)


def _wait_for_native_ready(process: subprocess.Popen, snapshot: Path, timeout: float) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise NativeProfileError(
                "native_launch_exited", "Stable Chrome exited before native startup completed."
            )
        try:
            port = int(_read_active_port_lines(snapshot)[0])
        except (NativeProfileError, OSError, ValueError, IndexError):
            time.sleep(0.1)
            continue
        if _probe_cdp(snapshot, port):
            return port
        time.sleep(0.1)
    raise NativeProfileError(
        "native_launch_timeout",
        "Stable Chrome did not expose a validated loopback endpoint; check for a permission prompt, close it, and retry once.",
    )


def _close_recorded_runtime(runtime: _NativeRuntime, *, timeout: float = 10.0) -> None:
    import psutil

    try:
        process = psutil.Process(runtime.process.pid)
    except psutil.NoSuchProcess:
        if _processes_owning_data_dir(runtime.snapshot_path):
            raise NativeProfileError(
                "native_cleanup_uncertain",
                "Native root exited but a detached snapshot owner remains.",
            )
        if not _recorded_listener_is_absent(runtime.cdp_port):
            raise NativeProfileError(
                "native_cleanup_uncertain",
                "Native root exited but recorded listener absence could not be proven.",
            )
        return
    if not _process_identity_matches(
        process, runtime.process.pid, runtime.process_start_time
    ):
        raise NativeProfileError(
            "native_pid_reused", "Native cleanup stopped because process identity changed."
        )

    # Browser.close is destructive too: prove the immutable signed artifact,
    # snapshot manifest, exact argv, CDP identity, and listener tree before
    # connecting to the endpoint.  Never trust the URL stored in a lease by
    # itself.
    websocket_url = _prove_recorded_runtime(runtime)
    try:
        import websocket

        ws = websocket.create_connection(websocket_url, timeout=2)
        try:
            ws.send(json.dumps({"id": 1, "method": "Browser.close"}))
        finally:
            ws.close()
    except Exception:
        pass
    try:
        process.wait(timeout=min(timeout, 5.0))
    except psutil.TimeoutExpired:
        # Recreate and fully reprove the process immediately before signaling.
        # A PID exec/reuse, manifest drift, listener reparent, or CDP change
        # retains the runtime and sends no signal.
        _prove_recorded_runtime(runtime)
        process = psutil.Process(runtime.process.pid)
        if not _process_identity_matches(
            process, runtime.process.pid, runtime.process_start_time
        ):
            raise NativeProfileError(
                "native_pid_reused",
                "Native cleanup stopped because process identity changed.",
            )
        try:
            executable_matches = _process_executable_matches(
                process, runtime.executable_path
            )
            _validate_live_process_signature(process.pid)
            argv = process.cmdline()
        except Exception as exc:
            raise NativeProfileError(
                "native_cleanup_identity_ambiguous",
                "Native cleanup stopped because process identity could not be reproven.",
            ) from exc
        if (
            not executable_matches
            or not isinstance(argv, (list, tuple))
            or not argv
            or any(not isinstance(arg, str) for arg in argv)
            or not argv_owns_data_dir(argv, runtime.snapshot_path)
            or not _process_identity_matches(
                process, runtime.process.pid, runtime.process_start_time
            )
        ):
            raise NativeProfileError(
                "native_cleanup_identity_ambiguous",
                "Native cleanup stopped because process identity changed.",
            )
        process.terminate()
        try:
            process.wait(timeout=max(1.0, timeout - 5.0))
        except psutil.TimeoutExpired as exc:
            raise NativeProfileError(
                "native_cleanup_uncertain", "Native browser cleanup could not prove process exit."
            ) from exc
    try:
        remaining = psutil.Process(runtime.process.pid)
    except psutil.NoSuchProcess:
        remaining = None
    if remaining is not None:
        # Even a reused PID is ambiguity here.  It is never signaled, and the
        # durable runtime remains available for a later explicit cleanup.
        raise NativeProfileError(
            "native_cleanup_uncertain", "Native browser cleanup could not prove process exit."
        )
    if _probe_cdp(Path(runtime.snapshot_path), runtime.cdp_port):
        raise NativeProfileError(
            "native_cleanup_uncertain", "Native browser cleanup could not prove endpoint absence."
        )
    if not _recorded_listener_is_absent(runtime.cdp_port):
        raise NativeProfileError(
            "native_cleanup_uncertain",
            "Native browser cleanup could not prove listener absence.",
        )
    if _processes_owning_data_dir(runtime.snapshot_path):
        raise NativeProfileError(
            "native_cleanup_uncertain",
            "Native root exited but a detached snapshot owner remains.",
        )


class NativeProfileSupervisor:
    """Single profile-scoped owner for native Chrome clients and cleanup."""

    _instances_guard = threading.RLock()
    _instances: dict[str, "NativeProfileSupervisor"] = {}

    def __init__(self, hermes_home: str):
        self.hermes_home = hermes_home

    @classmethod
    def for_profile(
        cls, home: str | Path | None = None
    ) -> "NativeProfileSupervisor":
        canonical_home, _snapshot, _supervisor = _profile_scope(home)
        with cls._instances_guard:
            instance = cls._instances.get(canonical_home)
            if instance is None:
                instance = cls(canonical_home)
                cls._instances[canonical_home] = instance
            return instance

    def acquire(
        self,
        browser_config: Mapping[str, object],
        env: Mapping[str, str],
    ) -> NativeProfileClient:
        """Atomically install a client token, then adopt or launch Chrome."""
        selected, request = native_intent(
            browser_config, env, system=platform.system()
        )
        if not selected or request is None:
            raise NativeProfileError(
                "native_not_selected", "Native real-profile mode is off."
            )
        try:
            delay = float(str(browser_config.get("inactivity_timeout", 120) or 120))
        except (TypeError, ValueError):
            delay = 120.0
        if not math.isfinite(delay) or delay <= 0:
            delay = 120.0

        home = self.hermes_home
        guard = _runtime_guard_for(home)
        with guard:
            timer = _cleanup_timers.pop(home, None)
            if timer is not None:
                timer.cancel()
            token = uuid.uuid4().hex
            _client_leases.setdefault(home, set()).add(token)
            try:
                cdp_url = _resolve_native_profile_cdp(
                    browser_config, env, hermes_home=home
                )
                runtime = _runtimes.get(home)
                if (
                    runtime is None
                    or not runtime.runtime_generation
                    or runtime.cdp_url != cdp_url
                ):
                    raise NativeProfileError(
                        "native_runtime_transaction_failed",
                        "Native runtime was not atomically bound to the client.",
                    )
                return NativeProfileClient(
                    hermes_home=home,
                    token=token,
                    cdp_url=cdp_url,
                    runtime_namespace=runtime.runtime_generation,
                    inactivity_delay=delay,
                )
            except Exception:
                tokens = _client_leases.get(home)
                if tokens is not None:
                    tokens.discard(token)
                    if not tokens:
                        _client_leases.pop(home, None)
                runtime = _runtimes.get(home)
                if not _client_leases.get(home) and runtime is not None:
                    _schedule_native_profile_cleanup(
                        delay,
                        hermes_home=home,
                        runtime_generation=runtime.runtime_generation,
                    )
                raise

    def release(self, client: NativeProfileClient) -> None:
        """Idempotently release the exact client and arm generation cleanup."""
        if not isinstance(client, NativeProfileClient) or (
            client.hermes_home != self.hermes_home
        ):
            raise NativeProfileError(
                "native_client_invalid", "Native profile client is invalid."
            )
        guard = _runtime_guard_for(self.hermes_home)
        with guard:
            tokens = _client_leases.get(self.hermes_home)
            if not tokens or client.token not in tokens:
                return
            tokens.remove(client.token)
            if tokens:
                return
            _client_leases.pop(self.hermes_home, None)
            _schedule_native_profile_cleanup(
                client.inactivity_delay,
                hermes_home=self.hermes_home,
                runtime_generation=client.runtime_namespace,
            )

    def cleanup(self, *, delete_snapshot: bool = False) -> None:
        _cleanup_native_profile(
            delete_snapshot=delete_snapshot, hermes_home=self.hermes_home
        )

    @classmethod
    def cleanup_all(cls) -> None:
        with cls._instances_guard:
            instances = tuple(cls._instances.values())
        errors: list[Exception] = []
        for instance in instances:
            guard = _runtime_guard_for(instance.hermes_home)
            with guard:
                _client_leases.pop(instance.hermes_home, None)
                timer = _cleanup_timers.pop(instance.hermes_home, None)
                if timer is not None:
                    timer.cancel()
            try:
                instance.cleanup(delete_snapshot=False)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]


def _rmtree_contents_fd(directory_fd: int) -> None:
    """Remove a directory tree through an already-validated descriptor."""
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise OSError("O_NOFOLLOW is required for native snapshot cleanup")
    directory_flags |= nofollow

    for name in os.listdir(directory_fd):
        entry_info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(entry_info.st_mode):
            if stat.S_IMODE(entry_info.st_mode) != 0o700 or (
                hasattr(os, "getuid") and entry_info.st_uid != os.getuid()
            ):
                raise NativeProfileError(
                    "native_snapshot_entry_invalid",
                    "Native snapshot contains an unsafe directory; retained it.",
                )
            child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
            try:
                opened_info = os.fstat(child_fd)
                if (opened_info.st_dev, opened_info.st_ino) != (
                    entry_info.st_dev,
                    entry_info.st_ino,
                ):
                    raise NativeProfileError(
                        "native_snapshot_identity_changed",
                        "Native snapshot entry changed during cleanup; retained it.",
                    )
                _rmtree_contents_fd(child_fd)
                current_info = os.stat(
                    name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if (current_info.st_dev, current_info.st_ino) != (
                    opened_info.st_dev,
                    opened_info.st_ino,
                ):
                    raise NativeProfileError(
                        "native_snapshot_identity_changed",
                        "Native snapshot entry changed during cleanup; retained it.",
                    )
                os.rmdir(name, dir_fd=directory_fd)
            finally:
                os.close(child_fd)
            continue

        if (
            not stat.S_ISREG(entry_info.st_mode)
            or entry_info.st_nlink != 1
            or stat.S_IMODE(entry_info.st_mode) != 0o600
            or (hasattr(os, "getuid") and entry_info.st_uid != os.getuid())
        ):
            raise NativeProfileError(
                "native_snapshot_entry_invalid",
                "Native snapshot contains an unsafe entry; retained it.",
            )
        os.unlink(name, dir_fd=directory_fd)


def _validate_snapshot_tree_fd(directory_fd: int) -> None:
    """Preflight the complete snapshot tree before deleting any entry."""
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise OSError("O_NOFOLLOW is required for native snapshot cleanup")
    directory_flags |= nofollow
    for name in os.listdir(directory_fd):
        entry_info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(entry_info.st_mode):
            if stat.S_IMODE(entry_info.st_mode) != 0o700 or (
                hasattr(os, "getuid") and entry_info.st_uid != os.getuid()
            ):
                raise NativeProfileError(
                    "native_snapshot_entry_invalid",
                    "Native snapshot contains an unsafe directory; retained it.",
                )
            child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
            try:
                opened = os.fstat(child_fd)
                if (opened.st_dev, opened.st_ino) != (
                    entry_info.st_dev,
                    entry_info.st_ino,
                ):
                    raise NativeProfileError(
                        "native_snapshot_identity_changed",
                        "Native snapshot entry changed during cleanup; retained it.",
                    )
                _validate_snapshot_tree_fd(child_fd)
                after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if (after.st_dev, after.st_ino) != (
                    entry_info.st_dev,
                    entry_info.st_ino,
                ):
                    raise NativeProfileError(
                        "native_snapshot_identity_changed",
                        "Native snapshot entry changed during cleanup; retained it.",
                    )
            finally:
                os.close(child_fd)
            continue
        if (
            not stat.S_ISREG(entry_info.st_mode)
            or entry_info.st_nlink != 1
            or stat.S_IMODE(entry_info.st_mode) != 0o600
            or (hasattr(os, "getuid") and entry_info.st_uid != os.getuid())
        ):
            raise NativeProfileError(
                "native_snapshot_entry_invalid",
                "Native snapshot contains an unsafe entry; retained it.",
            )


def _preflight_existing_snapshot(path: Path) -> None:
    """Retain and reject an existing snapshot whose tree is not owner-private."""
    if not os.path.lexists(path):
        return

    snapshot_fd: int | None = None
    try:
        before = os.lstat(path)
        if (
            not stat.S_ISDIR(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o700
            or (hasattr(os, "getuid") and before.st_uid != os.getuid())
        ):
            raise OSError("snapshot root has unsafe identity")
        snapshot_fd = _open_nofollow_components(
            str(path), os.O_RDONLY | os.O_DIRECTORY
        )
        opened = os.fstat(snapshot_fd)
        if (
            (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or stat.S_IMODE(opened.st_mode) != 0o700
            or (hasattr(os, "getuid") and opened.st_uid != os.getuid())
        ):
            raise OSError("snapshot root changed identity")
        _validate_snapshot_tree_fd(snapshot_fd)
        after = os.lstat(path)
        if (
            (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino)
            or stat.S_IMODE(after.st_mode) != 0o700
            or (hasattr(os, "getuid") and after.st_uid != os.getuid())
        ):
            raise OSError("snapshot root changed during validation")
    except (OSError, NativeProfileError) as exc:
        raise NativeProfileError(
            "native_snapshot_unsafe",
            "The existing native snapshot has unsafe identity; retained it.",
        ) from exc
    finally:
        if snapshot_fd is not None:
            os.close(snapshot_fd)


def _delete_validated_snapshot(
    path: str, expected_identity: tuple[int, int]
) -> None:
    """Delete only the directory entry whose device/inode were validated."""
    snapshot_path = Path(path)
    parent_fd = _open_nofollow_components(
        str(snapshot_path.parent), os.O_RDONLY | os.O_DIRECTORY
    )
    snapshot_fd: int | None = None
    try:
        current = os.stat(snapshot_path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != expected_identity:
            raise NativeProfileError(
                "native_snapshot_unsafe",
                "Native cleanup retained a snapshot whose identity changed.",
            )
        snapshot_fd = os.open(
            snapshot_path.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        opened = os.fstat(snapshot_fd)
        if (opened.st_dev, opened.st_ino) != expected_identity:
            raise NativeProfileError(
                "native_snapshot_unsafe",
                "Native cleanup retained a snapshot whose identity changed.",
            )
        if stat.S_IMODE(opened.st_mode) != 0o700 or (
            hasattr(os, "getuid") and opened.st_uid != os.getuid()
        ):
            raise NativeProfileError(
                "native_snapshot_unsafe",
                "Native cleanup retained a snapshot with unsafe identity.",
            )
        _validate_snapshot_tree_fd(snapshot_fd)
        current = os.stat(
            snapshot_path.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        opened = os.fstat(snapshot_fd)
        if (
            (current.st_dev, current.st_ino) != expected_identity
            or (opened.st_dev, opened.st_ino) != expected_identity
        ):
            raise NativeProfileError(
                "native_snapshot_identity_changed",
                "Native snapshot identity changed during cleanup; retained it.",
            )
        _rmtree_contents_fd(snapshot_fd)
        current = os.stat(
            snapshot_path.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        opened = os.fstat(snapshot_fd)
        if (
            (current.st_dev, current.st_ino) != expected_identity
            or (opened.st_dev, opened.st_ino) != expected_identity
        ):
            raise NativeProfileError(
                "native_snapshot_identity_changed",
                "Native snapshot identity changed during cleanup; retained it.",
            )
        os.rmdir(snapshot_path.name, dir_fd=parent_fd)
    except NativeProfileError:
        raise
    except OSError as exc:
        raise NativeProfileError(
            "native_snapshot_unsafe",
            "Native cleanup retained a snapshot whose identity could not be proven.",
        ) from exc
    finally:
        if snapshot_fd is not None:
            os.close(snapshot_fd)
        os.close(parent_fd)


def _cleanup_native_profile(
    *,
    delete_snapshot: bool = False,
    hermes_home: str | Path | None = None,
) -> None:
    """Close only the exact profile-scoped runtime, then delete if proven idle."""
    home, snapshot, _supervisor_root = _profile_scope(hermes_home)
    guard = _runtime_guard_for(home)
    with guard:
        if _client_leases.get(home):
            raise NativeProfileError(
                "native_clients_active",
                "Native cleanup retained the browser because clients are active.",
            )
        timer = _cleanup_timers.pop(home, None)
        if timer is not None:
            timer.cancel()
        runtime = _runtimes.get(home)
        operation_lock = runtime.lock if runtime is not None else None
        try:
            if operation_lock is None:
                operation_lock = NativeProfileLock(str(_supervisor_root)).acquire()
            operation_lock.prove()
            if runtime is None and os.path.lexists(
                _supervisor_root / "runtime.json"
            ):
                runtime = _read_runtime_lease(
                    home, snapshot, _supervisor_root, operation_lock
                )
            operation_lock.prove()
            if runtime is not None:
                # State remains installed until every close/exit proof succeeds.
                operation_lock.prove()
                _close_recorded_runtime(runtime)
                operation_lock.prove()
                _remove_runtime_lease(runtime)
                if _runtimes.get(home) is runtime:
                    _runtimes.pop(home, None)
            snapshot_present = os.path.lexists(snapshot)
            snapshot_identity: tuple[int, int] | None = None
            if snapshot_present and snapshot.is_symlink():
                raise NativeProfileError(
                    "native_snapshot_unsafe",
                    "Native cleanup retained a symlinked snapshot path.",
                )
            if snapshot_present:
                try:
                    initial_snapshot_info = os.lstat(snapshot)
                except OSError as exc:
                    raise NativeProfileError(
                        "native_snapshot_identity_changed",
                        "Native snapshot identity changed during cleanup; retained it.",
                    ) from exc
                snapshot_identity = (
                    initial_snapshot_info.st_dev,
                    initial_snapshot_info.st_ino,
                )
            owners = (
                _processes_owning_data_dir(str(snapshot)) if snapshot_present else []
            )
            operation_lock.prove()
            if snapshot_identity is not None:
                try:
                    current_snapshot_info = os.lstat(snapshot)
                except OSError as exc:
                    raise NativeProfileError(
                        "native_snapshot_identity_changed",
                        "Native snapshot identity changed during cleanup; retained it.",
                    ) from exc
                if (
                    current_snapshot_info.st_dev,
                    current_snapshot_info.st_ino,
                ) != snapshot_identity:
                    raise NativeProfileError(
                        "native_snapshot_identity_changed",
                        "Native snapshot identity changed during cleanup; retained it.",
                    )
            if owners:
                raise NativeProfileError(
                    "native_cleanup_owner_ambiguous",
                    "Native cleanup retained the snapshot because another owner is present.",
                )
            if delete_snapshot and snapshot_present:
                operation_lock.prove()
                canonical = _canonical_existing_dir(
                    str(snapshot), label="native snapshot"
                )
                info = os.lstat(canonical)
                if (info.st_dev, info.st_ino) != snapshot_identity:
                    raise NativeProfileError(
                        "native_snapshot_identity_changed",
                        "Native snapshot identity changed during cleanup; retained it.",
                    )
                if stat.S_IMODE(info.st_mode) != 0o700 or (
                    hasattr(os, "getuid") and info.st_uid != os.getuid()
                ):
                    raise NativeProfileError(
                        "native_snapshot_unsafe",
                        "Native cleanup retained a snapshot with unsafe identity.",
                    )
                _delete_validated_snapshot(
                    canonical, (info.st_dev, info.st_ino)
                )
                operation_lock.prove()
        finally:
            if operation_lock is not None and (
                runtime is None or _runtimes.get(home) is not runtime
            ):
                operation_lock.release()


def _schedule_native_profile_cleanup(
    delay_seconds: float,
    *,
    hermes_home: str | Path | None = None,
    runtime_generation: str | None = None,
) -> None:
    """Arm cleanup only for the zero-client runtime generation observed now."""
    home, _snapshot, _supervisor = _profile_scope(hermes_home)
    guard = _runtime_guard_for(home)

    def close_after_idle() -> None:
        guard = _runtime_guard_for(home)
        with guard:
            runtime = _runtimes.get(home)
            if runtime_generation is not None and (
                runtime is None
                or runtime.runtime_generation != runtime_generation
            ):
                return
        with contextlib.suppress(Exception):
            _cleanup_native_profile(delete_snapshot=False, hermes_home=home)

    with guard:
        if _client_leases.get(home):
            return
        timer = _cleanup_timers.pop(home, None)
        if timer is not None:
            timer.cancel()
        timer = threading.Timer(max(1.0, delay_seconds), close_after_idle)
        timer.daemon = True
        _cleanup_timers[home] = timer
        timer.start()


def _cleanup_failed_launch(
    process: subprocess.Popen,
    runtime: _NativeRuntime | None,
) -> None:
    """Clean a failed launch without signaling an unpinned runtime PID."""
    if runtime is None:
        raise NativeProfileError(
            "native_cleanup_uncertain",
            "Native launch failed before exact process identity was pinned.",
        )
    if process.poll() is not None:
        if _processes_owning_data_dir(runtime.snapshot_path):
            raise NativeProfileError(
                "native_cleanup_uncertain",
                "Native launch exited but a detached snapshot owner remains.",
            )
        _remove_runtime_lease(runtime)
        return
    _close_recorded_runtime(runtime, timeout=5)
    _remove_runtime_lease(runtime)


def _refresh_provisional_runtime(
    runtime: _NativeRuntime,
    request: NativeProfileRequest,
    executable_fingerprint: str,
) -> bool:
    """Promote a pinned port-zero lease only after the complete proof."""
    if runtime.cdp_port != 0 or runtime.cdp_url:
        return False
    try:
        lines = _read_active_port_lines(Path(runtime.snapshot_path))
        port = int(lines[0])
    except (NativeProfileError, OSError, ValueError, IndexError):
        return False
    if not 1 <= port <= 65535:
        return False
    ready_runtime = replace(
        runtime,
        cdp_port=port,
        cdp_url=f"http://127.0.0.1:{port}",
    )
    if not _runtime_is_valid(
        ready_runtime,
        request,
        Path(ready_runtime.snapshot_path),
        executable_fingerprint,
    ):
        return False
    try:
        _prove_recorded_runtime(ready_runtime)
    except NativeProfileError:
        return False
    _write_runtime_lease(ready_runtime)
    runtime.cdp_port = ready_runtime.cdp_port
    runtime.cdp_url = ready_runtime.cdp_url
    return True


def _recorded_process_is_absent(runtime: _NativeRuntime) -> bool:
    """Return true only when the recorded PID has no current process."""
    try:
        import psutil

        psutil.Process(runtime.process.pid)
    except psutil.NoSuchProcess:
        return True
    except Exception as exc:
        raise NativeProfileError(
            "native_cleanup_identity_ambiguous",
            "Native runtime process absence could not be proven.",
        ) from exc
    return False


def _resolve_native_profile_cdp(
    browser_config: Mapping[str, object],
    env: Mapping[str, str],
    *,
    hermes_home: str | Path | None = None,
) -> str:
    """Validate, provision if needed, launch, and prove native stable Chrome."""
    selected, request = native_intent(
        browser_config, env, system=platform.system()
    )
    if not selected or request is None:
        raise NativeProfileError("native_not_selected", "Native real-profile mode is off.")
    executable_fingerprint = _validate_stable_chrome()
    home, snapshot, supervisor_root = _profile_scope(hermes_home)
    source = str(Path.home() / "Library" / "Application Support" / "Google" / "Chrome")
    guard = _runtime_guard_for(home)
    with guard:
        timer = _cleanup_timers.pop(home, None)
        if timer is not None:
            timer.cancel()
        cached = _runtimes.get(home)
        if cached is not None:
            valid = _runtime_is_valid(
                cached, request, snapshot, executable_fingerprint
            ) or _refresh_provisional_runtime(
                cached, request, executable_fingerprint
            )
            if valid:
                return cached.cdp_url
            if len(_client_leases.get(home, ())) > 1:
                raise NativeProfileError(
                    "native_runtime_in_use",
                    "The native runtime changed while another client was active.",
                )
            # A request/config change may invalidate reuse while the old
            # immutable runtime is still fully provable. Close that exact old
            # owner and continue into relaunch in this same call.
            _close_recorded_runtime(cached)
            _remove_runtime_lease(cached)
            _runtimes.pop(home, None)
            cached.lock.release()

        lock = NativeProfileLock(str(supervisor_root)).acquire()
        lock_retained = False
        try:
            persisted = _read_runtime_lease(home, snapshot, supervisor_root, lock)
            if persisted is not None:
                valid = (
                    persisted.executable_fingerprint == executable_fingerprint
                    and (
                        _runtime_is_valid(
                            persisted, request, snapshot, executable_fingerprint
                        )
                        or _refresh_provisional_runtime(
                            persisted, request, executable_fingerprint
                        )
                    )
                )
                if valid:
                    _runtimes[home] = persisted
                    lock_retained = True
                    _prove_recorded_runtime(persisted)
                    return persisted.cdp_url

                # A live persisted owner is never ignored. If its immutable
                # identity still proves, close only that owner and continue;
                # otherwise retain the lease and fail closed without signaling.
                try:
                    _prove_recorded_runtime(persisted)
                except NativeProfileError as proof_error:
                    if not _recorded_process_is_absent(persisted):
                        raise NativeProfileError(
                            "native_cached_proof_failed",
                            "A persisted native browser owner could not be proven.",
                        ) from proof_error
                    if not _recorded_listener_is_absent(persisted.cdp_port):
                        raise NativeProfileError(
                            "native_cached_proof_failed",
                            "A persisted native browser listener could not be proven absent.",
                        ) from proof_error
                    if snapshot.exists() and _processes_owning_data_dir(str(snapshot)):
                        raise NativeProfileError(
                            "native_snapshot_owned",
                            "The native snapshot has an unproven incumbent owner.",
                        ) from proof_error
                    _remove_runtime_lease(persisted)
                else:
                    _close_recorded_runtime(persisted)
                    _remove_runtime_lease(persisted)

            if _processes_owning_data_dir(str(snapshot)):
                raise NativeProfileError(
                    "native_snapshot_owned",
                    "The native snapshot has an incumbent browser owner; close that exact session and retry.",
                )
            _preflight_existing_snapshot(snapshot)
            manifest = _manifest_for(request, snapshot, executable_fingerprint)
            if manifest is None:
                if os.path.lexists(snapshot):
                    raise NativeProfileError(
                        "native_snapshot_unsafe",
                        "An existing native snapshot could not be proven; retained it for explicit cleanup.",
                    )
                if _processes_owning_data_dir(
                    source,
                    implicit_default_executable=STABLE_CHROME_EXECUTABLE,
                ):
                    raise NativeProfileError(
                        "native_source_in_use",
                        "Work Chrome is open. Fully quit it once before provisioning the native snapshot.",
                    )
                manifest = provision_native_snapshot(
                    request,
                    source,
                    str(snapshot),
                    executable_fingerprint=executable_fingerprint,
                )
            supervisor_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            process = subprocess.Popen(
                native_chrome_argv(STABLE_CHROME_EXECUTABLE, str(snapshot)),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
                umask=0o077,
            )
            runtime: _NativeRuntime | None = None
            try:
                import psutil

                pinned = psutil.Process(process.pid)
                identity = _process_identity(pinned)
                if identity is None or identity[0] != process.pid:
                    raise NativeProfileError(
                        "native_launch_identity_unavailable",
                        "Stable Chrome process identity could not be pinned.",
                    )
                if not _process_executable_matches(
                    pinned, STABLE_CHROME_EXECUTABLE
                ):
                    raise NativeProfileError(
                        "native_launch_identity_unavailable",
                        "Stable Chrome executable identity could not be pinned.",
                    )
                _validate_live_process_signature(process.pid)
                if (
                    not _process_identity_matches(
                        pinned, process.pid, identity[1]
                    )
                    or not _process_executable_matches(
                        pinned, STABLE_CHROME_EXECUTABLE
                    )
                ):
                    raise NativeProfileError(
                        "native_launch_identity_unavailable",
                        "Stable Chrome identity changed before it was recorded.",
                    )
                lock_device, lock_inode = lock.identity
                runtime = _NativeRuntime(
                    process=process,
                    process_start_time=identity[1],
                    cdp_url="",
                    cdp_port=0,
                    snapshot_uuid=str(manifest["snapshot_uuid"]),
                    executable_fingerprint=executable_fingerprint,
                    lock=lock,
                    hermes_home=home,
                    snapshot_path=str(snapshot),
                    supervisor_path=str(supervisor_root),
                    source_profile_hash=str(manifest["source_profile_hash"]),
                    expected_account_hash=str(manifest["expected_account_hash"]),
                    executable_path=STABLE_CHROME_EXECUTABLE,
                    lease_path=str(supervisor_root / "runtime.json"),
                    runtime_generation=uuid.uuid4().hex,
                    lock_device=lock_device,
                    lock_inode=lock_inode,
                )
                _runtimes[home] = runtime
                lock_retained = True
                _write_runtime_lease(runtime)
                timeout_value = browser_config.get("command_timeout", 30)
                timeout = _native_startup_timeout(timeout_value)
                port = _wait_for_native_ready(process, snapshot, timeout)
                ready_runtime = replace(
                    runtime,
                    cdp_url=f"http://127.0.0.1:{port}",
                    cdp_port=port,
                )
                _write_runtime_lease(ready_runtime)
                runtime = ready_runtime
                _runtimes[home] = runtime
                if not _runtime_is_valid(runtime, request, snapshot, executable_fingerprint):
                    raise NativeProfileError(
                        "native_launch_proof_failed", "Native browser launch proof failed."
                    )
                _prove_recorded_runtime(runtime)
            except Exception as launch_error:
                try:
                    _cleanup_failed_launch(process, runtime)
                except Exception as cleanup_error:
                    if runtime is not None:
                        _runtimes[home] = runtime
                        lock_retained = True
                    raise NativeProfileError(
                        "native_cleanup_uncertain",
                        "Native launch failed and exact process cleanup could not be proven.",
                    ) from cleanup_error
                _runtimes.pop(home, None)
                lock_retained = False
                raise launch_error
            logger.info("native real-profile ready [native_ready]")
            return runtime.cdp_url
        finally:
            if not lock_retained:
                lock.release()


def _atexit_cleanup() -> None:
    with _state_guard:
        for home in tuple(_client_leases):
            _client_leases.pop(home, None)
    with contextlib.suppress(Exception):
        NativeProfileSupervisor.cleanup_all()


atexit.register(_atexit_cleanup)
