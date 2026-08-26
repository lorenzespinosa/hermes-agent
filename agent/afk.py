"""Durable AFK (away-from-keyboard) availability state.

``/afk`` — and Slack's thread-safe ``!afk`` — records that the operator has
stepped away, so agent turns that happen while they are gone know not to wait
on them and, just as importantly, know that nobody is there to approve
anything.

**One record for the whole machine.** Availability is a fact about the human,
not about a Hermes instance, so the state lives at the Hermes *root* —
:func:`hermes_constants.get_default_hermes_root`, resolved at call time — not
under the active profile home. A ``/afk`` typed in the work profile is seen and
clearable from the personal profile, the CLI, cron, and every gateway session.
This is a deliberate exception to the profiles-are-islands rule: isolating
availability per profile would mean the same person is simultaneously present
and absent, which is exactly the confusion the command exists to remove.

**Durability before success, serialized.** :func:`engage` holds a no-follow
cross-process advisory file lock (``fcntl`` on POSIX, ``msvcrt`` on Windows)
across write **and** readback, so
the payload it returns is its own committed write rather than a neighbour's.
The write itself is atomic (temp file + fsync + rename), so a concurrent reader
never sees a torn document.

**Permissions, stated truthfully.** On POSIX the file is written 0o600 and the
mode is *verified afterwards*; a write that cannot be proven owner-only fails
and removes the record. Native Windows has no POSIX mode bits and we install no
DACL, so we claim nothing beyond the per-user ``%LOCALAPPDATA%`` location —
see :func:`enforces_owner_only_permissions`.

**Corrupt or partial state fails engaged.** The presence of the file is the
status; its body is only metadata. A truncated, empty or hand-``touch``-ed file
still reports AFK with unknown metadata, because the operator saying "I'm away"
should not be undone by a bad byte.

**The reason is untrusted data and never reaches the model.** It is bounded and
neutralized on every *read* (not just on write — the file is bytes on disk that
anything could have produced), and it is used only for the operator's own
``/afk status`` reply. :func:`turn_context_note` carries status and timestamp
only.

**AFK is never authorization.** The note says so explicitly, and the shared
approval boundary backs it with behaviour: while AFK, an approval prompt is
never sent and every pending request is denied outright
(:data:`APPROVAL_DENY_REASON`).

**No identity is stored.** The state records availability and nothing else — no
platform, no chat, no user id — so it cannot become a backdoor way to carry a
Slack/Telegram identity into another surface. "The operator" means the owner of
this machine's Hermes; gateway mutation requires an explicitly configured
administrator, while a local CLI invocation already runs as the machine user.
"""

from __future__ import annotations

import errno
import json
import os
import stat
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

STATE_NAME = "afk.json"
LOCK_NAME = "afk.lock"

# Owner-only on POSIX. Availability plus a free-text reason is
# personal-schedule data, and the Hermes root may be group-readable.
STATE_MODE = 0o600

# Upper bound on the reason. Chat-supplied text, bounded on read and write.
MAX_REASON_CHARS = 200

# Upper bound on the timestamp field read back off disk. Nothing legitimate
# comes close; the cap exists because the field is read from a file.
MAX_TIMESTAMP_CHARS = 64

# The whole point of the note: an absent operator has not delegated authority.
NO_AUTHORITY_SENTENCE = (
    "AFK was a status, not a permission: at receipt time it did NOT widen "
    "approvals, grant authority, or authorize any consequential or irreversible "
    "action on the operator's behalf. Anything requiring approval remained "
    "unauthorized unless it was separately approved."
)

# Deterministic denial handed to the tool that asked for approval while the
# operator was away. Constant, not a rendering: the agent must get the same
# explanation every time, and it must be unmistakable that being away is not
# a decision about the request.
APPROVAL_DENY_REASON = (
    "Denied automatically: the operator is AFK and cannot answer an approval "
    "request. AFK is availability, not approval — it is not consent and not a "
    "judgement about this command. Do the safe, reversible parts of the task "
    "and re-request this step once they are back."
)

APPROVAL_STATUS_UNKNOWN_REASON = (
    "Denied automatically: Hermes could not safely verify the machine-global "
    "AFK status, so it cannot establish that the operator is available to "
    "approve this request. No consent was granted."
)


class AfkStateError(RuntimeError):
    """The AFK state could not be durably written, verified, or removed."""


def _afk_root() -> Path:
    """Resolve the machine-wide Hermes root at call time.

    ``get_default_hermes_root()`` maps ``<root>/profiles/<name>`` back to
    ``<root>``, so every profile — and the default home — resolves to the same
    directory. Freshness-correct: it re-derives from ``HERMES_HOME`` on each
    call rather than caching a boot-time snapshot.
    """
    try:
        from hermes_constants import get_default_hermes_root

        return Path(get_default_hermes_root())
    except Exception as exc:
        # Machine-global state must never split across an inferred fallback.
        # A write to ~/.hermes after profile-root resolution failed would
        # silently create a second truth, so all mutations fail closed.
        raise AfkStateError(f"could not resolve the machine-global Hermes root: {exc}") from exc


def state_path() -> Path:
    """Path of the single machine-wide AFK state file."""
    return _afk_root() / STATE_NAME


def _lock_path() -> Path:
    return _afk_root() / LOCK_NAME


def _reject_symlink(path: Path, *, label: str) -> None:
    """Reject an existing symlink without following it."""
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    except OSError as exc:
        raise AfkStateError(f"could not inspect AFK {label} at {path}: {exc}") from exc
    if stat.S_ISLNK(mode):
        raise AfkStateError(f"refusing symlinked AFK {label} at {path}")


def _nofollow_flags(flags: int) -> int:
    return flags | getattr(os, "O_NOFOLLOW", 0)


class _NoFollowFileLock:
    """Advisory lock whose leaf path is never followed through a symlink."""

    def __init__(self, path: Path):
        self.path = path
        self._fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _reject_symlink(self.path, label="lock")
        flags = _nofollow_flags(os.O_RDWR | os.O_CREAT)
        try:
            fd = os.open(self.path, flags, STATE_MODE)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise AfkStateError(
                    f"refusing symlinked AFK lock at {self.path}"
                ) from exc
            raise AfkStateError(f"AFK state lock unavailable: {exc}") from exc
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode):
                raise AfkStateError(
                    f"AFK lock at {self.path} is not a regular file"
                )
            current = self.path.lstat()
            if (
                stat.S_ISLNK(current.st_mode)
                or current.st_dev != opened.st_dev
                or current.st_ino != opened.st_ino
            ):
                raise AfkStateError(
                    f"AFK lock at {self.path} changed while it was opened"
                )
            self._fh = os.fdopen(fd, "a+b")
            fd = -1
            if os.name == "nt":
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
            return self
        except Exception:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            elif fd >= 0:
                os.close(fd)
            raise

    def __exit__(self, exc_type, exc, tb):
        if self._fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None


_STATUS_MUTEX = threading.RLock()
_STATUS_LOCAL = threading.local()


def _state_lock():
    return _NoFollowFileLock(_lock_path())


@contextmanager
def status_transaction():
    """Serialize AFK state transitions with approval queue decisions.

    Nested use on the same thread reuses the held file lock. This matters for
    notification callbacks that synchronously resolve their own queue entry.
    """
    with _STATUS_MUTEX:
        depth = getattr(_STATUS_LOCAL, "depth", 0)
        if depth:
            _STATUS_LOCAL.depth = depth + 1
            try:
                yield
            finally:
                _STATUS_LOCAL.depth -= 1
            return
        try:
            lock = _state_lock()
        except Exception as exc:
            if isinstance(exc, AfkStateError):
                raise
            raise AfkStateError(f"AFK state lock unavailable: {exc}") from exc
        with lock:
            _STATUS_LOCAL.depth = 1
            try:
                yield
            finally:
                _STATUS_LOCAL.depth = 0


# ---------------------------------------------------------------------------
# Untrusted-field handling
# ---------------------------------------------------------------------------


def _neutralize(value: Any, max_chars: int) -> Optional[str]:
    """Bound an untrusted string and strip its injection surface.

    Control characters (newlines included) collapse to spaces so the value
    cannot open a fake section wherever it is rendered, and square brackets
    soften to parentheses so it cannot forge a ``[System note: ...]`` block.
    Returns None for blank or non-string input.
    """
    if not isinstance(value, str):
        return None
    text = "".join(" " if ch < " " or ch == "\x7f" else ch for ch in value)
    text = text.replace("[", "(").replace("]", ")")
    text = " ".join(text.split())
    if not text:
        return None
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def _sanitize_reason(reason: Any) -> Optional[str]:
    """Bound and neutralize an operator-supplied reason."""
    return _neutralize(reason, MAX_REASON_CHARS)


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------


def enforces_owner_only_permissions() -> bool:
    """True where the OS gives us POSIX mode bits we can actually enforce.

    False on native Windows: ``os.chmod`` there only toggles the read-only
    attribute, and we deliberately do not hand-roll a DACL. On that platform
    the only guarantee is the per-user ``%LOCALAPPDATA%`` location, which we
    state rather than dress up as owner-only.
    """
    return os.name != "nt"


def _mode_is_owner_only(mode: int) -> bool:
    """True when *mode* grants nothing to group or other. Pure predicate."""
    return mode & 0o077 == 0


def _file_mode(path: Path) -> Optional[int]:
    """Return permission bits from a no-follow descriptor, or None."""
    try:
        fd = os.open(path, _nofollow_flags(os.O_RDONLY))
    except OSError:
        return None
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            return None
        return stat.S_IMODE(opened.st_mode)
    except OSError:
        return None
    finally:
        os.close(fd)


def _verify_owner_only(path: Path) -> None:
    """Raise unless *path* is provably owner-only (POSIX hosts only).

    One repair attempt first — an inherited umask or a pre-existing
    world-readable file is recoverable — then a hard failure. A record we
    cannot prove is private is not a successful write.
    """
    if not enforces_owner_only_permissions():
        return
    mode = _file_mode(path)
    if mode is not None and _mode_is_owner_only(mode):
        return
    try:
        fd = os.open(path, _nofollow_flags(os.O_RDONLY))
        try:
            os.fchmod(fd, STATE_MODE)
        finally:
            os.close(fd)
    except OSError as exc:
        raise AfkStateError(
            f"AFK state at {path} could not be made owner-only: {exc}"
        ) from exc
    mode = _file_mode(path)
    if mode is None:
        raise AfkStateError(
            f"AFK state at {path} was written but its permissions could not be "
            "verified"
        )
    if not _mode_is_owner_only(mode):
        raise AfkStateError(
            f"AFK state at {path} is {oct(mode)}, not owner-only"
        )


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------


def _read_state_unlocked() -> Optional[dict]:
    """Read and sanitize the state file. Caller decides about locking."""
    path = state_path()
    _reject_symlink(path, label="state")
    try:
        fd = os.open(path, _nofollow_flags(os.O_RDONLY))
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise AfkStateError(f"refusing symlinked AFK state at {path}") from exc
        # Present but unreadable: keep the operator marked away.
        return {"engaged_at": None, "reason": None}
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise AfkStateError(f"AFK state at {path} is not a regular file")
        with os.fdopen(fd, "r", encoding="utf-8") as fh:
            fd = -1
            raw_text = fh.read()
    except AfkStateError:
        raise
    except (OSError, UnicodeError):
        return {"engaged_at": None, "reason": None}
    finally:
        if fd >= 0:
            os.close(fd)

    try:
        raw = json.loads(raw_text)
    except ValueError:
        raw = None

    engaged_at = None
    reason = None
    if isinstance(raw, dict):
        # Sanitized on READ, not only on write: this file is bytes on disk.
        engaged_at = _neutralize(raw.get("engaged_at"), MAX_TIMESTAMP_CHARS)
        reason = _sanitize_reason(raw.get("reason"))
    return {"engaged_at": engaged_at, "reason": reason}


def get_state() -> Optional[dict]:
    """Return ``{"engaged_at": ..., "reason": ...}``, or None when available.

    Both fields are untrusted data, bounded and neutralized on every read. A
    state file that exists but cannot be parsed still reports AFK with both
    fields ``None`` — the away-status is authoritative, its metadata is not.

    Lock-free by design: the write path replaces the file atomically, so a
    reader either sees the old document or the new one, never a torn one.
    """
    return _read_state_unlocked()


def is_afk() -> bool:
    """True when the operator is currently marked away.

    Derived from :func:`get_state` so the two can never disagree about what a
    corrupt or unreadable state file means.
    """
    return get_state() is not None


def _atomic_replace_json(path: Path, payload: dict) -> None:
    """Write *payload* beside *path* and atomically replace the leaf.

    The target is checked with ``lstat`` and all opened leaves use
    ``O_NOFOLLOW`` where available. ``os.replace`` replaces a target symlink
    itself rather than following it, so a swap after the check still cannot
    modify the symlink's victim.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink(path, label="state")
    fd = -1
    temp_name = None
    try:
        fd, temp_name = tempfile.mkstemp(prefix=f".{STATE_NAME}.", dir=path.parent)
        if enforces_owner_only_permissions():
            os.fchmod(fd, STATE_MODE)
        data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        with os.fdopen(fd, "wb") as fh:
            fd = -1
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if fd >= 0:
            os.close(fd)
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def engage(reason: Optional[str] = None) -> dict:
    """Mark the operator away and return the state read back off disk.

    The write, the permission check and the readback happen inside one lock,
    so the returned payload is this call's own committed write. Idempotent:
    re-engaging replaces the timestamp and reason.

    Raises:
        AfkStateError: the state could not be written, could not be proven
            owner-only, or did not survive the write. Callers must not report
            success in that case.
    """
    path = state_path()
    payload = {
        "engaged_at": datetime.now(timezone.utc).isoformat(),
        "reason": _sanitize_reason(reason),
    }
    try:
        with status_transaction():
            try:
                _atomic_replace_json(path, payload)
            except OSError as exc:
                raise AfkStateError(
                    f"could not write AFK state to {path}: {exc}"
                ) from exc
            try:
                _verify_owner_only(path)
            except AfkStateError:
                # Never leave a record we could not make private.
                try:
                    path.unlink()
                except OSError:
                    pass
                raise
            state = _read_state_unlocked()
            if state is None:
                raise AfkStateError(
                    f"AFK state at {path} did not survive the write"
                )
            # The transition and queue denial share this transaction. By the
            # time engage() returns, no already-pending interactive approval
            # remains capable of granting consent or waiting indefinitely.
            from tools.approval import deny_pending_approvals_for_afk

            deny_pending_approvals_for_afk()
            return state
    except AfkStateError:
        raise
    except (OSError, RuntimeError) as exc:
        raise AfkStateError(f"AFK state transaction failed: {exc}") from exc


def clear() -> bool:
    """Mark the operator back. True when an AFK state was actually lifted.

    Taken under the same lock as :func:`engage` so a clear can never land
    between a concurrent write and its readback. ``unlink`` returning is itself
    the durable proof, so there is no confirming read: it could only be raced
    by a later legitimate ``/afk on``.

    Raises:
        AfkStateError: a state exists but could not be removed — a caller must
            not say "welcome back" over an AFK that still stands.
    """
    path = state_path()
    try:
        with status_transaction():
            _reject_symlink(path, label="state")
            try:
                path.unlink()
                return True
            except FileNotFoundError:
                return False
            except OSError as exc:
                raise AfkStateError(
                    f"could not clear AFK state at {path}: {exc}"
                ) from exc
    except AfkStateError:
        raise
    except (OSError, RuntimeError) as exc:
        raise AfkStateError(f"AFK state transaction failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Shared slash-command grammar and rendering (CLI/TUI/gateway)
# ---------------------------------------------------------------------------

AFK_CLEAR_VERBS = frozenset({"off", "back", "return"})
AFK_USAGE = (
    "Usage: `/afk` or `/afk on [reason]` to step away · "
    "`/afk off` (also `back`, `return`) when you're back · `/afk status`."
)


def _command_reply(*lines: str) -> str:
    return "\n".join(
        [
            *lines,
            "_AFK never widens approvals or authorizes consequential work — "
            "anything that needs your say-so still waits for you._",
        ]
    )


def _since(state: dict) -> str:
    when = state.get("engaged_at")
    return f"since {when}" if when else "since an unknown time"


def _because(state: dict) -> str:
    reason = state.get("reason")
    return f" ({reason})" if reason else ""


def _state_location() -> str:
    """Truthful machine-global location for diagnostics, with no fallback."""
    try:
        return str(state_path())
    except AfkStateError:
        return "the unresolved machine-global Hermes root"


def _command_engage(reason: Optional[str], *, bare: bool = False) -> str:
    if bare:
        try:
            existing = get_state()
        except Exception:
            existing = None
        if existing is not None:
            return _command_reply(
                f"🌙 Already AFK {_since(existing)}{_because(existing)}.",
                "Use `/afk off` when you're back.",
            )
    try:
        state = engage(reason=reason)
    except AfkStateError:
        return _command_reply(
            "⚠️ Couldn't record AFK — the machine-global state at "
            f"{_state_location()} could not be durably written and read back. "
            "Nothing changed."
        )
    return _command_reply(
        f"🌙 AFK recorded {_since(state)}{_because(state)}.",
        "Turns that happen while you're away will be told you're not at the "
        "keyboard. Use `/afk off` when you're back.",
    )


def _command_clear() -> str:
    try:
        lifted = clear()
    except AfkStateError:
        return _command_reply(
            "⚠️ Couldn't clear AFK — the machine-global state at "
            f"{_state_location()} could not be removed. You're still marked away."
        )
    if not lifted:
        return _command_reply("☀️ You weren't marked AFK.")
    return _command_reply("☀️ Welcome back — AFK cleared.")


def _command_status() -> str:
    try:
        state = get_state()
    except Exception:
        return _command_reply(
            "⚠️ Couldn't read the machine-global AFK state at "
            f"{_state_location()} — availability is unknown."
        )
    if state is None:
        return _command_reply("☀️ Not AFK — you're marked available.")
    return _command_reply(
        f"🌙 AFK {_since(state)}{_because(state)}.",
        "Use `/afk off` when you're back.",
    )


def handle_command(args: str = "") -> str:
    """Execute the closed ``/afk`` grammar for every command surface."""
    args = (args or "").strip()
    parts = args.split(None, 1)
    verb = parts[0].lower() if parts else ""
    rest = parts[1].strip() if len(parts) > 1 else ""
    if not args:
        return _command_engage(None, bare=True)
    if verb == "on":
        return _command_engage(rest or None)
    if verb in AFK_CLEAR_VERBS and not rest:
        return _command_clear()
    if verb == "status" and not rest:
        return _command_status()
    return _command_reply(AFK_USAGE)


def turn_context_note() -> Optional[str]:
    """Render the per-turn availability note, or None when available.

    Status and timestamp only. The free-text reason is deliberately absent:
    it is operator-typed content, and there is no version of "quote the
    human's note into every prompt" that is worth the injection surface when
    the agent only needs to know *that* nobody is at the keyboard.

    Delivered centrally on the *current user message* through the agent's
    API-bound content path — never the system prompt and never a rewrite of
    clean durable content. The receipt-time wording makes a replayed sidecar a
    historical fact, while every new turn gets a fresh current availability
    read.
    """
    state = get_state()
    if state is None:
        return None
    when = state.get("engaged_at") or "an unknown time"
    return (
        "[System note: Availability at receipt time: the operator was AFK "
        f"(away from keyboard) since {when}. This records availability when "
        "this user message was received, not the operator's current status "
        "when the message is replayed.\n"
        f"{NO_AUTHORITY_SENTENCE}]"
    )
