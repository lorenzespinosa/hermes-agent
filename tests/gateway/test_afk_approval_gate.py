"""AFK enforcement at the shared approval queue/wait/resolve boundary.

Messaging, API, TUI/desktop, ACP, and other interactive adapters all register
different notification callbacks, but they wait and resolve through
``tools.approval``. AFK therefore belongs at that boundary: no surface may
notify while the operator is already away, an AFK transition must deny and
signal existing entries, and a stale client response must never turn AFK into
once/session/always consent.

The race tests coordinate with ``threading.Event`` rather than sleeps.
"""

from __future__ import annotations

import subprocess
import sys
import threading

import pytest

from agent import afk
from tools import approval


@pytest.fixture
def hermes_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def session_key():
    key = "afk-approval-test"
    yield key
    approval.unregister_gateway_notify(key)
    approval.clear_session(key)


def _blocking_approval(
    session_key, notify_cb, command="rm -rf /tmp/x", *, surface="gateway"
):
    outcome: dict = {}
    done = threading.Event()

    def _run():
        try:
            outcome["result"] = approval._await_gateway_decision(
                session_key,
                notify_cb,
                {
                    "command": command,
                    "description": "test",
                    "pattern_key": "test_key",
                    "pattern_keys": ["test_key"],
                },
                surface=surface,
            )
        except BaseException as exc:  # pragma: no cover - failure detail
            outcome["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return outcome, done, thread


def _configure_run_gate(monkeypatch, *, gateway: bool) -> None:
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(
        approval, "is_current_session_yolo_enabled", lambda: False
    )
    monkeypatch.setattr(
        approval, "_is_single_query_approval_context", lambda: False
    )
    monkeypatch.setattr(approval, "_is_cron_approval_context", lambda: False)
    monkeypatch.setattr(
        approval, "_is_gateway_approval_context", lambda: gateway
    )
    monkeypatch.setattr(approval, "_is_interactive_cli", lambda: not gateway)


def _real_run_gate(
    session_key: str,
    *,
    pattern_key: str,
    approval_callback=None,
) -> dict:
    token = approval.set_current_session_key(session_key)
    try:
        return approval._run_approval_gate(
            pattern_key=pattern_key,
            description="barrier-controlled test action",
            display_target="<barrier-test>",
            approval_callback=approval_callback,
            cron_deny_message="cron",
            single_query_deny_message="single",
            autoapprove_log_prefix="test",
        )
    finally:
        approval.reset_current_session_key(token)


def _run_in_thread(target):
    outcome: dict = {}
    done = threading.Event()

    def _run():
        try:
            outcome["result"] = target()
        except BaseException as exc:  # pragma: no cover - failure detail
            outcome["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return outcome, done, thread


@pytest.mark.parametrize("surface", ["messaging", "api", "tui", "acp"])
def test_afk_before_notification_denies_every_interactive_surface(
    hermes_root, surface
):
    afk.engage(reason="school run")
    notified: list[dict] = []

    outcome, done, thread = _blocking_approval(
        f"afk-{surface}",
        lambda data: notified.append(dict(data)),
        surface=surface,
    )

    assert done.wait(timeout=5), f"{surface} approval stayed blocked while AFK"
    thread.join(timeout=1)
    assert "error" not in outcome, outcome.get("error")
    assert notified == []
    assert outcome["result"]["resolved"] is True
    assert outcome["result"]["choice"] == "deny"
    assert outcome["result"]["reason"] == afk.APPROVAL_DENY_REASON
    assert outcome["result"]["afk_denied"] is True


def test_unreadable_state_before_notification_suppresses_callback(
    hermes_root, session_key, monkeypatch
):
    notified: list[dict] = []

    def _unreadable():
        raise RuntimeError("simulated unreadable AFK state")

    monkeypatch.setattr(afk, "is_afk", _unreadable)
    outcome, done, thread = _blocking_approval(
        session_key, lambda data: notified.append(dict(data))
    )

    assert done.wait(timeout=5)
    thread.join(timeout=1)
    assert "error" not in outcome, outcome.get("error")
    assert notified == []
    assert outcome["result"]["choice"] == "deny"
    assert outcome["result"]["availability_denied"] is True
    assert outcome["result"]["reason"] == afk.APPROVAL_STATUS_UNKNOWN_REASON


def test_engaging_afk_signals_an_already_pending_approval(
    hermes_root, session_key
):
    notified = threading.Event()
    notify_count = 0
    notify_lock = threading.Lock()

    def _notify(_data):
        nonlocal notify_count
        with notify_lock:
            notify_count += 1
        notified.set()

    outcome, done, thread = _blocking_approval(session_key, _notify)
    assert notified.wait(timeout=5), "approval never reached the pending state"
    assert approval.has_blocking_approval(session_key)

    afk.engage(reason="stepped away mid-run")

    assert done.wait(timeout=5), "AFK transition did not signal the pending wait"
    thread.join(timeout=1)
    assert "error" not in outcome, outcome.get("error")
    assert outcome["result"]["choice"] == "deny"
    assert outcome["result"]["afk_denied"] is True
    assert outcome["result"]["reason"] == afk.APPROVAL_DENY_REASON
    assert approval.list_gateway_approvals(session_key) == []
    with notify_lock:
        assert notify_count == 1, "engaging AFK sent another approval ping"


def test_afk_engaged_in_another_process_unblocks_pending_approval(
    hermes_root, session_key
):
    notified = threading.Event()
    outcome, done, thread = _blocking_approval(
        session_key, lambda _data: notified.set()
    )
    assert notified.wait(timeout=5)

    completed = subprocess.run(
        [sys.executable, "-c", "from agent import afk; afk.engage()"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert done.wait(timeout=5), "cross-process AFK did not unblock approval"
    thread.join(timeout=1)
    assert outcome["result"]["choice"] == "deny"
    assert outcome["result"]["afk_denied"] is True


def test_afk_engaged_by_pre_approval_hook_still_prevents_notification(
    hermes_root, session_key, monkeypatch
):
    notified: list[dict] = []

    def _hook(name, **_kwargs):
        if name == "pre_approval_request":
            afk.engage(reason="hook transition")

    monkeypatch.setattr(approval, "_fire_approval_hook", _hook)
    outcome, done, thread = _blocking_approval(
        session_key, lambda data: notified.append(dict(data))
    )

    assert done.wait(timeout=5)
    thread.join(timeout=1)
    assert notified == []
    assert outcome["result"]["choice"] == "deny"
    assert outcome["result"]["afk_denied"] is True


@pytest.mark.parametrize("attempted_choice", ["once", "session", "always"])
def test_resolution_while_afk_can_only_deny(
    hermes_root, session_key, attempted_choice
):
    """A stale card/API response cannot convert absence into authorization."""
    afk.engage()
    entry = approval._ApprovalEntry(
        {
            "command": "danger",
            "description": "test",
            "pattern_key": "test_key",
            "pattern_keys": ["test_key"],
        }
    )
    with approval._lock:
        approval._gateway_queues[session_key] = [entry]

    resolved = approval.resolve_gateway_approval(
        session_key,
        attempted_choice,
        request_id=entry.data["request_id"],
    )

    assert resolved == 1
    assert entry.event.is_set()
    assert entry.result == "deny"
    assert entry.reason == afk.APPROVAL_DENY_REASON
    assert entry.resolution_source == "afk"


def test_available_operator_still_resolves_normally(hermes_root, session_key):
    def _answer(data):
        assert approval.resolve_gateway_approval(
            session_key, "once", request_id=data["request_id"]
        ) == 1

    outcome, done, thread = _blocking_approval(session_key, _answer)

    assert done.wait(timeout=5)
    thread.join(timeout=1)
    assert outcome["result"]["choice"] == "once"
    assert outcome["result"].get("afk_denied") is not True


def test_tool_result_names_afk_not_a_user_denial(
    hermes_root, session_key, monkeypatch
):
    afk.engage(reason="lunch")
    approval.register_gateway_notify(session_key, lambda _data: pytest.fail("pinged"))
    monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: True)
    token = approval.set_current_session_key(session_key)
    try:
        result = approval._run_approval_gate(
            pattern_key="plugin_rule:test",
            description="test action",
            display_target="<test>",
            cron_deny_message="cron",
            single_query_deny_message="single",
            autoapprove_log_prefix="test",
        )
    finally:
        approval.reset_current_session_key(token)

    assert result["approved"] is False
    assert "BLOCKED" in result["message"]
    assert "operator is AFK" in result["message"]
    assert "denied by user" not in result["message"]
    assert "Reason given by the user" not in result["message"]
    assert result["user_consent"] is False


@pytest.mark.parametrize(
    "state_mode, expected_outcome, expected_reason",
    [
        ("afk", "afk_denied", afk.APPROVAL_DENY_REASON),
        (
            "unreadable",
            "availability_unknown",
            afk.APPROVAL_STATUS_UNKNOWN_REASON,
        ),
    ],
)
def test_local_tui_callback_is_not_presented_when_operator_unavailable(
    hermes_root,
    monkeypatch,
    state_mode,
    expected_outcome,
    expected_reason,
):
    session = f"local-presentation-{state_mode}"
    pattern = f"local_presentation_{state_mode}"
    _configure_run_gate(monkeypatch, gateway=False)
    calls = []
    hooks = []
    monkeypatch.setattr(
        approval,
        "_fire_approval_hook",
        lambda name, **_kwargs: hooks.append(name),
    )

    if state_mode == "afk":
        afk.engage(reason="already away")
    else:
        def _unreadable():
            raise RuntimeError("simulated unreadable AFK state")

        monkeypatch.setattr(afk, "is_afk", _unreadable)

    result = _real_run_gate(
        session,
        pattern_key=pattern,
        approval_callback=lambda *_args, **_kwargs: calls.append("presented"),
    )

    assert calls == []
    assert hooks == []
    assert result["approved"] is False
    assert result["outcome"] == expected_outcome
    assert result["deny_reason"] == expected_reason
    assert approval.is_approved(session, pattern) is False


@pytest.mark.parametrize("choice", ["once", "session", "always"])
def test_real_run_gate_callback_grant_finishing_after_afk_is_denied(
    hermes_root, monkeypatch, choice
):
    """CLI/TUI/ACP callbacks cannot return stale pre-AFK consent."""
    session = f"afk-local-callback-{choice}"
    pattern = f"afk_local_callback_{choice}"
    approval.clear_session(session)
    with approval._lock:
        approval._permanent_approved.discard(pattern)
    _configure_run_gate(monkeypatch, gateway=False)
    saved: list[set] = []
    monkeypatch.setattr(
        approval,
        "save_permanent_allowlist",
        lambda patterns: saved.append(set(patterns)),
    )
    callback_entered = threading.Event()
    release_callback = threading.Event()

    def _callback(*_args, **_kwargs):
        callback_entered.set()
        assert release_callback.wait(timeout=5)
        return choice

    outcome, done, thread = _run_in_thread(
        lambda: _real_run_gate(
            session, pattern_key=pattern, approval_callback=_callback
        )
    )
    assert callback_entered.wait(timeout=5)

    afk.engage(reason="committed while callback was open")
    release_callback.set()

    assert done.wait(timeout=5)
    thread.join(timeout=1)
    assert "error" not in outcome, outcome.get("error")
    assert outcome["result"]["approved"] is False
    assert outcome["result"]["outcome"] == "afk_denied"
    assert approval.is_approved(session, pattern) is False
    assert saved == []


def test_real_run_gate_paused_event_set_cannot_commit_stale_gateway_grant(
    hermes_root, monkeypatch
):
    """Exact reviewer race: AFK commits after resolve but before event.set."""
    session = "afk-paused-event-set"
    pattern = "afk_paused_event_set"
    approval.clear_session(session)
    with approval._lock:
        approval._permanent_approved.discard(pattern)
    _configure_run_gate(monkeypatch, gateway=True)
    notified = threading.Event()
    approval.register_gateway_notify(session, lambda _data: notified.set())
    saved: list[set] = []
    monkeypatch.setattr(
        approval,
        "save_permanent_allowlist",
        lambda patterns: saved.append(set(patterns)),
    )

    outcome, done, gate_thread = _run_in_thread(
        lambda: _real_run_gate(session, pattern_key=pattern)
    )
    assert notified.wait(timeout=5)
    with approval._lock:
        entry = approval._gateway_queues[session][0]

    event_set_entered = threading.Event()
    release_event_set = threading.Event()
    original_set = entry.event.set

    def _paused_set():
        event_set_entered.set()
        assert release_event_set.wait(timeout=5)
        original_set()

    monkeypatch.setattr(entry.event, "set", _paused_set)
    resolved: dict = {}
    resolver = threading.Thread(
        target=lambda: resolved.setdefault(
            "count",
            approval.resolve_gateway_approval(
                session,
                "always",
                request_id=entry.data["request_id"],
            ),
        ),
        daemon=True,
    )
    resolver.start()
    assert event_set_entered.wait(timeout=5)

    # The resolver has removed the entry and released its first transaction,
    # so this commit must complete while event.set is still paused.
    afk.engage(reason="reviewer race")
    assert afk.is_afk() is True
    release_event_set.set()

    resolver.join(timeout=5)
    assert not resolver.is_alive(), "resolver deadlocked"
    assert done.wait(timeout=5), "approval waiter remained stuck"
    gate_thread.join(timeout=1)
    assert "error" not in outcome, outcome.get("error")
    assert resolved["count"] == 1
    assert outcome["result"]["approved"] is False
    assert outcome["result"]["outcome"] == "afk_denied"
    assert approval.is_approved(session, pattern) is False
    assert saved == []


def test_paused_event_set_race_stress_has_zero_grants_after_afk_commit(
    hermes_root, monkeypatch
):
    _configure_run_gate(monkeypatch, gateway=True)
    approvals_after_commit = 0

    for iteration in range(100):
        afk.clear()
        session = f"afk-event-stress-{iteration}"
        pattern = f"afk_event_stress_{iteration}"
        approval.clear_session(session)
        notified = threading.Event()
        approval.register_gateway_notify(session, lambda _data: notified.set())
        outcome, done, gate_thread = _run_in_thread(
            lambda s=session, p=pattern: _real_run_gate(s, pattern_key=p)
        )
        assert notified.wait(timeout=5), iteration
        with approval._lock:
            entry = approval._gateway_queues[session][0]

        event_set_entered = threading.Event()
        release_event_set = threading.Event()
        original_set = entry.event.set

        def _paused_set(
            entered=event_set_entered,
            release=release_event_set,
            real_set=original_set,
        ):
            entered.set()
            assert release.wait(timeout=5)
            real_set()

        entry.event.set = _paused_set
        resolver = threading.Thread(
            target=approval.resolve_gateway_approval,
            args=(session, "once"),
            kwargs={"request_id": entry.data["request_id"]},
            daemon=True,
        )
        resolver.start()
        assert event_set_entered.wait(timeout=5), iteration
        afk.engage(reason=f"stress-{iteration}")
        release_event_set.set()
        resolver.join(timeout=5)
        assert not resolver.is_alive(), f"resolver deadlocked at {iteration}"
        assert done.wait(timeout=5), f"waiter stuck at {iteration}"
        gate_thread.join(timeout=1)
        assert "error" not in outcome, outcome.get("error")
        approvals_after_commit += int(outcome["result"]["approved"])
        approval.unregister_gateway_notify(session)
        approval.clear_session(session)

    assert approvals_after_commit == 0


@pytest.mark.parametrize("afk_timing", ["before-presentation", "mid-prompt"])
def test_selected_plugin_transport_obeys_afk_presentation_and_finalization(
    hermes_root, monkeypatch, afk_timing
):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    session = "afk-selected-transport"
    pattern = "afk_selected_transport"
    manager = PluginManager()
    context = PluginContext(
        PluginManifest(
            name="afk-transport-fixture",
            version="1.0.0",
            description="fixture",
            source="user",
            key="afk-transport-fixture",
        ),
        manager,
    )
    transport_entered = threading.Event()
    release_transport = threading.Event()

    def _present(request):
        transport_entered.set()
        assert release_transport.wait(timeout=5)
        return request.respond("always")

    context.register_approval_transport("afk-phone", _present)
    monkeypatch.setattr(approval, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
    monkeypatch.setattr(
        approval, "_is_gateway_approval_context", lambda: False
    )
    monkeypatch.setattr(
        approval, "_is_single_query_approval_context", lambda: False
    )
    monkeypatch.setattr(approval, "detect_hardline_command", lambda _c: (False, ""))
    monkeypatch.setattr(approval, "_check_sudo_stdin_guard", lambda _c: (False, ""))
    monkeypatch.setattr(approval, "_match_user_deny_rule", lambda _c: None)
    monkeypatch.setattr(
        approval, "_command_matches_permanent_allowlist", lambda _c: False
    )
    monkeypatch.setattr(
        approval,
        "detect_dangerous_command",
        lambda _c: (True, pattern, "dangerous"),
    )
    monkeypatch.setattr(approval, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(
        approval,
        "_get_approval_transport_config",
        lambda: ("afk-phone", None),
    )
    monkeypatch.setattr(
        "tools.tirith_security.check_command_security",
        lambda _c: {"action": "allow"},
    )
    saved: list[set] = []
    monkeypatch.setattr(
        approval,
        "save_permanent_allowlist",
        lambda patterns: saved.append(set(patterns)),
    )
    with approval._lock:
        approval._permanent_approved.discard(pattern)
    approval.clear_session(session)

    def _run_transport():
        token = approval.set_current_session_key(session)
        try:
            return approval.check_all_command_guards(
                "rm -rf /tmp/afk-transport", "local"
            )
        finally:
            approval.reset_current_session_key(token)

    if afk_timing == "before-presentation":
        afk.engage(reason="already away")
    outcome, done, thread = _run_in_thread(_run_transport)
    if afk_timing == "mid-prompt":
        assert transport_entered.wait(timeout=5)
        afk.engage(reason="transport was still open")
        release_transport.set()

    assert done.wait(timeout=5)
    thread.join(timeout=1)
    assert "error" not in outcome, outcome.get("error")
    assert transport_entered.is_set() is (afk_timing == "mid-prompt")
    assert outcome["result"]["approved"] is False
    assert outcome["result"]["outcome"] == "afk_denied"
    assert approval.is_approved(session, pattern) is False
    assert saved == []


@pytest.mark.parametrize("afk_timing", ["before-presentation", "mid-prompt"])
def test_mcp_callback_obeys_afk_presentation_and_finalization(
    hermes_root, monkeypatch, afk_timing
):
    from tools.terminal_tool import set_approval_callback

    monkeypatch.setattr(
        approval, "_is_gateway_approval_context", lambda: False
    )
    callback_entered = threading.Event()
    release_callback = threading.Event()

    def _callback(*_args, **_kwargs):
        callback_entered.set()
        assert release_callback.wait(timeout=5)
        return "once"

    def _request():
        set_approval_callback(_callback)
        try:
            return approval.request_elicitation_consent(
                "MCP wants input", "test elicitation", timeout_seconds=5
            )
        finally:
            set_approval_callback(None)

    if afk_timing == "before-presentation":
        afk.engage(reason="already away")
    outcome, done, thread = _run_in_thread(_request)
    if afk_timing == "mid-prompt":
        assert callback_entered.wait(timeout=5)
        afk.engage(reason="MCP callback still open")
        release_callback.set()

    assert done.wait(timeout=5)
    thread.join(timeout=1)
    assert "error" not in outcome, outcome.get("error")
    assert callback_entered.is_set() is (afk_timing == "mid-prompt")
    assert outcome["result"] == "decline"


def test_real_pending_entry_is_signalled_when_state_resolution_raises(
    hermes_root, session_key, monkeypatch
):
    notified = threading.Event()
    outcome, done, thread = _blocking_approval(
        session_key, lambda _data: notified.set()
    )
    assert notified.wait(timeout=5)
    assert approval.has_blocking_approval(session_key)

    main_thread = threading.current_thread()
    real_is_afk = afk.is_afk

    def _raise_only_for_resolver():
        if threading.current_thread() is main_thread:
            raise RuntimeError("simulated state resolver failure")
        return real_is_afk()

    monkeypatch.setattr(afk, "is_afk", _raise_only_for_resolver)
    assert approval.resolve_gateway_approval(session_key, "once") == 1

    assert done.wait(timeout=5), "state error left approval waiter stuck"
    thread.join(timeout=1)
    assert "error" not in outcome, outcome.get("error")
    assert outcome["result"]["choice"] == "deny"
    assert outcome["result"]["availability_denied"] is True
    assert outcome["result"]["reason"] == afk.APPROVAL_STATUS_UNKNOWN_REASON
    assert approval.list_gateway_approvals(session_key) == []


def test_invalid_utf8_state_denies_and_signals_a_real_pending_entry(
    hermes_root, session_key
):
    notified = threading.Event()
    outcome, done, thread = _blocking_approval(
        session_key, lambda _data: notified.set()
    )
    assert notified.wait(timeout=5)
    afk.state_path().write_bytes(b"\xff\xfe\x80")

    assert approval._enforce_current_availability_on_pending() is True
    assert done.wait(timeout=5), "corrupt state left approval waiter stuck"
    thread.join(timeout=1)
    assert "error" not in outcome, outcome.get("error")
    assert outcome["result"]["choice"] == "deny"
    assert outcome["result"]["afk_denied"] is True
    assert approval.list_gateway_approvals(session_key) == []
