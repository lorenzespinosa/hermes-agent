from __future__ import annotations

import json
import os
import sys
import threading
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest


_NATIVE_CONFIG = {
    "backend": "browser-use",
    "use_real_profile": True,
    "real_profile_macos_native": True,
    "real_profile_source_browser": "chrome",
    "real_profile_source_profile": "Profile 1",
    "real_profile_expected_account": "worker@example.test",
    "headed": True,
    "cloud_provider": "local",
    "cdp_url": "",
    "inactivity_timeout": 120,
}


def _tool_payload(result):
    if isinstance(result, str):
        return json.loads(result)
    return result


def test_native_supervisor_is_the_only_public_lifecycle_api():
    import hermes_cli.native_real_profile as native

    supervisor = native.NativeProfileSupervisor.for_profile("/tmp/hermes-native-test")

    assert supervisor.hermes_home == os.path.realpath("/tmp/hermes-native-test")
    assert callable(supervisor.acquire)
    assert callable(supervisor.release)
    assert callable(supervisor.cleanup)
    assert callable(native.NativeProfileSupervisor.cleanup_all)
    for obsolete in (
        "acquire_native_profile_client",
        "resolve_and_acquire_native_profile_client",
        "release_native_profile_client",
        "resolve_native_profile_cdp",
        "schedule_native_profile_cleanup",
    ):
        assert not hasattr(native, obsolete)


@pytest.mark.parametrize(
    "argv",
    [
        ["chrome", "--user-data-dir=/tmp/profile", "--user-data-dir=/tmp/profile"],
        ["chrome", "--user-data-dir"],
        ["chrome", "--user-data-dir", "--headless"],
        ["chrome", "--user-data-directory=/tmp/profile"],
        ["chrome", "--user-data-dir=relative/profile"],
    ],
)
def test_duplicate_or_malformed_user_data_dir_is_ambiguous(argv):
    import hermes_cli.native_real_profile as native

    assert native.argv_owns_data_dir(argv, "/tmp/profile") is None


def test_nonloopback_lsof_listener_is_rejected():
    import hermes_cli.native_real_profile as native

    output = (
        b"p4242\0cGoogle Chrome\0\n"
        b"f12\0PTCP\0n0.0.0.0:43123\0TST=LISTEN\0\n"
    )

    assert native._parse_lsof_listener_owners(output, 43123) is None


def test_second_lsof_same_pid_reuse_is_rejected(monkeypatch):
    import psutil

    import hermes_cli.native_real_profile as native

    calls = 0

    def process(_pid):
        nonlocal calls
        calls += 1
        start_time = 101.0 if calls >= 4 else 100.0
        return SimpleNamespace(
            pid=4242,
            create_time=lambda: start_time,
            parents=lambda: [],
        )

    monkeypatch.setattr(psutil, "Process", process)
    monkeypatch.setattr(native, "_query_lsof_listener_owners", lambda _port: (4242,))

    assert native._listener_is_loopback_only(4242, 43123, 100.0) is False


def test_loopback_port_hijacked_by_unrelated_pid_is_rejected(monkeypatch):
    import psutil

    import hermes_cli.native_real_profile as native

    processes = {
        4242: SimpleNamespace(
            pid=4242, create_time=lambda: 100.0, parents=lambda: []
        ),
        5000: SimpleNamespace(
            pid=5000, create_time=lambda: 200.0, parents=lambda: []
        ),
    }
    monkeypatch.setattr(psutil, "Process", lambda pid: processes[pid])
    monkeypatch.setattr(native, "_query_lsof_listener_owners", lambda _port: (5000,))

    assert native._listener_is_loopback_only(4242, 43123, 100.0) is False


def test_native_mode_preempts_camofox_before_backend_selection(monkeypatch):
    import tools.browser_camofox as camofox
    import tools.browser_use_cli as browser_use

    monkeypatch.setattr(browser_use, "_read_browser_cfg", lambda **_kw: dict(_NATIVE_CONFIG))
    monkeypatch.setattr(camofox, "is_camofox_mode", lambda: True)

    assert browser_use.is_browser_use_cli_mode() is True


def test_native_schema_and_exec_preempt_camofox_lightpanda_and_cloud(monkeypatch):
    import tools.browser_use_cli as browser_use

    hostile = {
        **_NATIVE_CONFIG,
        "engine": "lightpanda",
        "cloud_provider": "browserbase",
    }
    monkeypatch.setattr(browser_use, "_read_browser_cfg", lambda **_kw: hostile)
    forbidden = Mock(side_effect=AssertionError("native validation must run first"))
    monkeypatch.setattr(browser_use, "_find_cli", forbidden)
    monkeypatch.setattr(browser_use, "_resolve_backend_cdp", forbidden)

    assert browser_use.is_browser_use_cli_mode() is True
    payload = _tool_payload(browser_use.browser_exec("print('x')"))

    assert "native_selector_conflict" in payload["error"]
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    ("updates", "env", "code"),
    [
        ({"backend": ""}, {}, "native_backend_required"),
        ({"backend": "off"}, {}, "native_backend_required"),
        ({"cloud_provider": "browserbase"}, {}, "native_local_only"),
        ({"cdp_url": "http://127.0.0.1:9222"}, {}, "native_override_conflict"),
        ({"engine": "lightpanda"}, {}, "native_selector_conflict"),
        ({}, {"AGENT_BROWSER_ENGINE": "lightpanda"}, "native_selector_conflict"),
        ({}, {"CAMOFOX_URL": "http://127.0.0.1:9377"}, "native_selector_conflict"),
    ],
)
def test_native_intent_rejects_every_non_browser_use_route(updates, env, code):
    from hermes_cli.native_real_profile import NativeProfileError, native_intent

    config = {**_NATIVE_CONFIG, **updates}

    with pytest.raises(NativeProfileError) as raised:
        native_intent(config, env, system="Darwin")

    assert raised.value.code == code


@pytest.mark.parametrize("engine", ["", "auto", "chrome"])
def test_native_intent_allows_nondivergent_ambient_engine(engine):
    from hermes_cli.native_real_profile import native_intent

    selected, request = native_intent(
        _NATIVE_CONFIG,
        {"AGENT_BROWSER_ENGINE": engine},
        system="Darwin",
    )

    assert selected is True
    assert request is not None


def test_native_browser_exec_uses_only_supervisor_endpoint_and_sanitized_env(
    monkeypatch, tmp_path
):
    import hermes_cli.native_real_profile as native
    import tools.browser_use_cli as browser_use
    from tools.environments import local as local_env

    client = SimpleNamespace(
        hermes_home="/tmp/hermes-native-test",
        cdp_url="http://127.0.0.1:43123",
        runtime_namespace="generation-7",
        token="client-1",
    )
    supervisor = Mock()
    supervisor.acquire.return_value = client
    monkeypatch.setattr(
        native.NativeProfileSupervisor,
        "for_profile",
        classmethod(lambda cls, home=None: supervisor),
    )
    monkeypatch.setattr(browser_use, "_read_browser_cfg", lambda **_kw: dict(_NATIVE_CONFIG))
    monkeypatch.setattr(browser_use, "_find_cli", lambda: ["browser-use", "--json"])
    monkeypatch.setattr(browser_use, "_workspace_dir", lambda _task_id: str(tmp_path))
    monkeypatch.setattr(browser_use, "_blocked_url_in_code", lambda _code: None)
    monkeypatch.setattr(
        local_env,
        "hermes_subprocess_env",
        lambda **_kwargs: {
            "PATH": "/usr/bin:/bin",
            "HOME": "/tmp/home",
            "HERMES_HOME": "/tmp/hermes-native-test",
            "KEEP_ME": "must-not-cross-native-boundary",
            "BU_CDP_URL": "https://hostile.example",
            "BU_CDP_WS": "wss://hostile.example",
            "BROWSER_CDP_URL": "https://hostile.example",
            "BU_AUTOSPAWN": "1",
            "BROWSER_USE_API_KEY": "secret",
            "BROWSERBASE_API_KEY": "secret",
            "FIRECRAWL_API_KEY": "secret",
            "CAMOFOX_URL": "https://hostile.example",
            "AGENT_BROWSER_ENGINE": "lightpanda",
        },
    )
    backend = Mock(side_effect=AssertionError("generic backend resolver must not run"))
    monkeypatch.setattr(browser_use, "_resolve_backend_cdp", backend)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured.update(cmd=cmd, **kwargs)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(browser_use.subprocess, "run", fake_run)

    payload = _tool_payload(
        browser_use.browser_exec(
            "print('ok')", session="work", timeout_s=30, task_id="task-1"
        )
    )

    assert payload["success"] is True
    supervisor.acquire.assert_called_once()
    supervisor.release.assert_called_once_with(client)
    backend.assert_not_called()
    env = captured["env"]
    assert env["BU_CDP_URL"] == client.cdp_url
    assert env["BU_NAME"].startswith("hermes_native_")
    assert len(env["BU_NAME"]) <= 30
    assert env["BU_NAME"] == browser_use._native_daemon_session_name(
        "work", client.hermes_home, client.runtime_namespace
    )
    assert env["BU_NAME"] != browser_use._native_daemon_session_name(
        "work", client.hermes_home, "generation-8"
    )
    for key in (
        "BU_CDP_WS",
        "BROWSER_CDP_URL",
        "BU_AUTOSPAWN",
        "BROWSER_USE_API_KEY",
        "BROWSERBASE_API_KEY",
        "FIRECRAWL_API_KEY",
        "CAMOFOX_URL",
        "AGENT_BROWSER_ENGINE",
        "HERMES_HOME",
        "KEEP_ME",
    ):
        assert key not in env
    assert set(env) <= {
        "PATH",
        "HOME",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TERM",
        "PYTHONUTF8",
        "ANONYMIZED_TELEMETRY",
        "BU_CDP_URL",
        "BU_NAME",
        "BH_AGENT_WORKSPACE",
    }


def test_native_browser_exec_releases_client_on_subprocess_timeout(monkeypatch):
    import hermes_cli.native_real_profile as native
    import tools.browser_use_cli as browser_use
    from tools.environments import local as local_env

    client = SimpleNamespace(
        hermes_home="/tmp/hermes-native-test",
        cdp_url="http://127.0.0.1:43123",
        runtime_namespace="generation-8",
        token="client-2",
    )
    supervisor = Mock()
    supervisor.acquire.return_value = client
    monkeypatch.setattr(
        native.NativeProfileSupervisor,
        "for_profile",
        classmethod(lambda cls, home=None: supervisor),
    )
    monkeypatch.setattr(browser_use, "_read_browser_cfg", lambda **_kw: dict(_NATIVE_CONFIG))
    monkeypatch.setattr(browser_use, "_find_cli", lambda: ["browser-use"])
    monkeypatch.setattr(browser_use, "_blocked_url_in_code", lambda _code: None)
    monkeypatch.setattr(
        local_env,
        "hermes_subprocess_env",
        lambda **_kwargs: {"PATH": "/usr/bin"},
    )
    monkeypatch.setattr(
        browser_use.subprocess,
        "run",
        Mock(side_effect=browser_use.subprocess.TimeoutExpired("browser-use", 30)),
    )

    payload = _tool_payload(browser_use.browser_exec("print('x')", timeout_s=30))

    assert "timed out" in payload["error"]
    supervisor.release.assert_called_once_with(client)


def test_concurrent_native_browser_exec_calls_hold_independent_clients(
    monkeypatch, tmp_path
):
    import hermes_cli.native_real_profile as native
    import tools.browser_use_cli as browser_use
    from tools.environments import local as local_env

    supervisor = Mock()
    acquire_guard = threading.Lock()
    clients = []

    def acquire(*_args, **_kwargs):
        with acquire_guard:
            client = SimpleNamespace(
                hermes_home=str(tmp_path),
                cdp_url="http://127.0.0.1:43123",
                runtime_namespace="generation-shared",
                token=f"client-{len(clients)}",
            )
            clients.append(client)
            return client

    supervisor.acquire.side_effect = acquire
    monkeypatch.setattr(
        native.NativeProfileSupervisor,
        "for_profile",
        classmethod(lambda cls, home=None: supervisor),
    )
    monkeypatch.setattr(browser_use, "_read_browser_cfg", lambda **_kw: dict(_NATIVE_CONFIG))
    monkeypatch.setattr(browser_use, "_find_cli", lambda: ["browser-use"])
    monkeypatch.setattr(browser_use, "_blocked_url_in_code", lambda _code: None)
    monkeypatch.setattr(browser_use, "_workspace_dir", lambda _task: str(tmp_path))
    monkeypatch.setattr(
        local_env,
        "hermes_subprocess_env",
        lambda **_kwargs: {"PATH": "/usr/bin", "HOME": str(tmp_path)},
    )
    entered = threading.Barrier(2)
    checked = threading.Barrier(2)
    captured_envs = []

    def run_cli(_cmd, **kwargs):
        captured_envs.append(dict(kwargs["env"]))
        entered.wait(timeout=5)
        assert supervisor.release.call_count == 0
        checked.wait(timeout=5)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(browser_use.subprocess, "run", run_cli)
    results = []
    threads = [
        threading.Thread(
            target=lambda name=name: results.append(
                _tool_payload(browser_use.browser_exec("print('x')", session=name))
            )
        )
        for name in ("work-a", "work-b")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)

    assert len(results) == 2
    assert all(result["success"] is True for result in results)
    assert len(captured_envs) == 2
    assert len({env["BU_NAME"] for env in captured_envs}) == 2
    assert all(env["BU_NAME"].startswith("hermes_native_") for env in captured_envs)
    assert all(len(env["BU_NAME"]) <= 30 for env in captured_envs)
    assert {call.args[0].token for call in supervisor.release.call_args_list} == {
        "client-0",
        "client-1",
    }


def test_new_runtime_generation_gets_new_native_daemon_namespace(monkeypatch, tmp_path):
    import hermes_cli.native_real_profile as native
    import tools.browser_use_cli as browser_use
    from tools.environments import local as local_env

    supervisor = Mock()
    supervisor.acquire.side_effect = [
        SimpleNamespace(
            hermes_home=str(tmp_path),
            cdp_url="http://127.0.0.1:43121",
            runtime_namespace="generation-before-restart",
            token="before",
        ),
        SimpleNamespace(
            hermes_home=str(tmp_path),
            cdp_url="http://127.0.0.1:43122",
            runtime_namespace="generation-after-restart",
            token="after",
        ),
    ]
    monkeypatch.setattr(
        native.NativeProfileSupervisor,
        "for_profile",
        classmethod(lambda cls, home=None: supervisor),
    )
    monkeypatch.setattr(browser_use, "_read_browser_cfg", lambda **_kw: dict(_NATIVE_CONFIG))
    monkeypatch.setattr(browser_use, "_find_cli", lambda: ["browser-use"])
    monkeypatch.setattr(browser_use, "_blocked_url_in_code", lambda _code: None)
    monkeypatch.setattr(browser_use, "_workspace_dir", lambda _task: str(tmp_path))
    monkeypatch.setattr(
        local_env,
        "hermes_subprocess_env",
        lambda **_kwargs: {"PATH": "/usr/bin", "HOME": str(tmp_path)},
    )
    names = []
    monkeypatch.setattr(
        browser_use.subprocess,
        "run",
        lambda _cmd, **kwargs: (
            names.append(kwargs["env"]["BU_NAME"])
            or SimpleNamespace(returncode=0, stdout="ok", stderr="")
        ),
    )

    first = _tool_payload(browser_use.browser_exec("print('x')", session="work"))
    second = _tool_payload(browser_use.browser_exec("print('x')", session="work"))

    assert first["success"] is True
    assert second["success"] is True
    assert len(names) == 2
    assert names[0] != names[1]
    assert all(name.startswith("hermes_native_") for name in names)
    assert all(len(name) <= 30 for name in names)


def test_launch_persists_launching_then_ready_with_one_generation(monkeypatch, tmp_path):
    import psutil

    import hermes_cli.native_real_profile as native

    home = tmp_path / "profile"
    home.mkdir()
    fingerprint = "f" * 64
    manifest = {
        "version": 1,
        "snapshot_uuid": "snapshot-1",
        "source_profile_hash": "source-hash",
        "expected_account_hash": "account-hash",
        "executable_fingerprint": fingerprint,
    }
    process = Mock(pid=4242, poll=Mock(return_value=None))
    pinned = Mock(
        pid=4242,
        create_time=Mock(return_value=100.0),
        exe=Mock(return_value=native.STABLE_CHROME_EXECUTABLE),
    )
    launch = Mock(return_value=process)
    monkeypatch.setattr(native, "_validate_stable_chrome", lambda: fingerprint)
    monkeypatch.setattr(native, "_processes_owning_data_dir", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(native, "_manifest_for", lambda *_args: manifest)
    provision = Mock(side_effect=AssertionError("a valid snapshot must not read the source"))
    monkeypatch.setattr(native, "provision_native_snapshot", provision)
    monkeypatch.setattr(native.subprocess, "Popen", launch)
    monkeypatch.setattr(psutil, "Process", lambda _pid: pinned)
    monkeypatch.setattr(native, "_validate_live_process_signature", lambda _pid: None)
    wait_for_ready = Mock(return_value=43123)
    monkeypatch.setattr(native, "_wait_for_native_ready", wait_for_ready)
    monkeypatch.setattr(native, "_runtime_is_valid", lambda *_args: True)
    monkeypatch.setattr(
        native,
        "_prove_recorded_runtime",
        lambda _runtime: "ws://127.0.0.1:43123/devtools/browser/abc",
    )
    writes = []
    original_write = native._write_runtime_lease

    def capture_write(runtime):
        original_write(runtime)
        writes.append(native._runtime_lease_payload(runtime))

    monkeypatch.setattr(native, "_write_runtime_lease", capture_write)
    supervisor = native.NativeProfileSupervisor.for_profile(home)
    client = supervisor.acquire(
        {**_NATIVE_CONFIG, "command_timeout": 10**100}, {}
    )
    runtime = native._runtimes[supervisor.hermes_home]

    try:
        assert [payload["state"] for payload in writes] == ["launching", "ready"]
        assert writes[0]["runtime_generation"] == writes[1]["runtime_generation"]
        assert client.runtime_namespace == writes[1]["runtime_generation"]
        assert writes[0]["cdp_port"] == 0
        assert writes[1]["cdp_port"] == 43123
        provision.assert_not_called()
        command = launch.call_args.args[0]
        assert command == native.native_chrome_argv(
            native.STABLE_CHROME_EXECUTABLE, runtime.snapshot_path
        )
        assert launch.call_args.kwargs["umask"] == 0o077
        assert launch.call_args.kwargs["start_new_session"] is True
        assert launch.call_args.kwargs["close_fds"] is True
        wait_for_ready.assert_called_once_with(
            process, native.Path(runtime.snapshot_path), 120.0
        )
    finally:
        native._client_leases.pop(supervisor.hermes_home, None)
        native._runtimes.pop(supervisor.hermes_home, None)
        native._cleanup_timers.pop(supervisor.hermes_home, None)
        runtime.lock.release()


def test_daily_chrome_opening_during_provision_discards_unpublished_copy(
    monkeypatch, tmp_path
):
    import hermes_cli.native_real_profile as native

    source = tmp_path / "source"
    profile = source / "Profile 1"
    profile.mkdir(parents=True)
    local_state = {
        "profile": {
            "info_cache": {
                "Profile 1": {"email": "worker@example.test"},
            }
        }
    }
    state_path = source / "Local State"
    state_path.write_text(json.dumps(local_state), encoding="utf-8")
    state_path.chmod(0o600)
    cookies = profile / "Cookies"
    cookies.write_bytes(b"cookies")
    cookies.chmod(0o600)
    destination = tmp_path / "supervisor" / "snapshot"
    owner = SimpleNamespace(pid=777)

    def owners(path, **_kwargs):
        return [owner] if os.path.realpath(path) == os.path.realpath(source) else []

    monkeypatch.setattr(native, "_processes_owning_data_dir", owners)

    with pytest.raises(native.NativeProfileError) as raised:
        native.provision_native_snapshot(
            native.NativeProfileRequest(
                "chrome", "Profile 1", "worker@example.test"
            ),
            str(source),
            str(destination),
            executable_fingerprint="f" * 64,
        )

    assert raised.value.code == "native_source_raced"
    assert not destination.exists()


def test_permission_prompt_timeout_is_typed_once_without_retry_or_fallback(
    monkeypatch, tmp_path
):
    import psutil

    import hermes_cli.native_real_profile as native

    home = tmp_path / "profile"
    home.mkdir()
    fingerprint = "f" * 64
    manifest = {
        "version": 1,
        "snapshot_uuid": "snapshot-1",
        "source_profile_hash": "source-hash",
        "expected_account_hash": "account-hash",
        "executable_fingerprint": fingerprint,
    }
    process = Mock(pid=4242, poll=Mock(return_value=1))
    pinned = Mock(
        pid=4242,
        create_time=Mock(return_value=100.0),
        exe=Mock(return_value=native.STABLE_CHROME_EXECUTABLE),
    )
    launch = Mock(return_value=process)
    wait = Mock(
        side_effect=native.NativeProfileError(
            "native_launch_timeout", "permission prompt was not approved"
        )
    )
    fallback = Mock(side_effect=AssertionError("native timeout cannot fall back"))
    monkeypatch.setattr(native, "_validate_stable_chrome", lambda: fingerprint)
    monkeypatch.setattr(native, "_processes_owning_data_dir", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(native, "_manifest_for", lambda *_args: manifest)
    monkeypatch.setattr(native, "provision_native_snapshot", fallback)
    monkeypatch.setattr(native.subprocess, "Popen", launch)
    monkeypatch.setattr(psutil, "Process", lambda _pid: pinned)
    monkeypatch.setattr(native, "_validate_live_process_signature", lambda _pid: None)
    monkeypatch.setattr(native, "_wait_for_native_ready", wait)
    supervisor = native.NativeProfileSupervisor.for_profile(home)

    with pytest.raises(native.NativeProfileError) as raised:
        supervisor.acquire(_NATIVE_CONFIG, {})

    assert raised.value.code == "native_launch_timeout"
    launch.assert_called_once()
    wait.assert_called_once()
    fallback.assert_not_called()
    assert supervisor.hermes_home not in native._client_leases
    assert supervisor.hermes_home not in native._runtimes
    _canonical, _snapshot, root = native._profile_scope(home)
    assert not (root / "runtime.json").exists()


def test_ready_record_write_failure_rolls_back_matching_launch_record(
    monkeypatch, tmp_path
):
    import psutil

    import hermes_cli.native_real_profile as native

    home = tmp_path / "profile"
    home.mkdir()
    fingerprint = "f" * 64
    manifest = {
        "version": 1,
        "snapshot_uuid": "snapshot-1",
        "source_profile_hash": "source-hash",
        "expected_account_hash": "account-hash",
        "executable_fingerprint": fingerprint,
    }
    process = Mock(pid=4242, poll=Mock(return_value=1))
    pinned = Mock(
        pid=4242,
        create_time=Mock(return_value=100.0),
        exe=Mock(return_value=native.STABLE_CHROME_EXECUTABLE),
    )
    monkeypatch.setattr(native, "_validate_stable_chrome", lambda: fingerprint)
    monkeypatch.setattr(native, "_processes_owning_data_dir", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(native, "_manifest_for", lambda *_args: manifest)
    monkeypatch.setattr(native.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(psutil, "Process", lambda _pid: pinned)
    monkeypatch.setattr(native, "_validate_live_process_signature", lambda _pid: None)
    monkeypatch.setattr(native, "_wait_for_native_ready", lambda *_args: 43123)
    original_write = native._write_runtime_lease
    writes = 0

    def fail_ready_write(runtime):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise native.NativeProfileError(
                "native_runtime_lease_write_failed", "ready write failed"
            )
        original_write(runtime)

    monkeypatch.setattr(native, "_write_runtime_lease", fail_ready_write)
    supervisor = native.NativeProfileSupervisor.for_profile(home)

    with pytest.raises(native.NativeProfileError) as raised:
        supervisor.acquire(_NATIVE_CONFIG, {})

    assert raised.value.code == "native_runtime_lease_write_failed"
    assert writes == 2
    assert supervisor.hermes_home not in native._runtimes
    assert supervisor.hermes_home not in native._client_leases
    _canonical, _snapshot, root = native._profile_scope(home)
    assert not (root / "runtime.json").exists()


@pytest.mark.parametrize("proof_is_valid", [True, False])
def test_durable_runtime_is_adopted_only_with_complete_live_proof(
    monkeypatch, tmp_path, proof_is_valid
):
    import hermes_cli.native_real_profile as native

    home = tmp_path / "profile"
    home.mkdir()
    canonical_home, snapshot, root = native._profile_scope(home)
    root.parent.mkdir(parents=True, mode=0o700)
    root.parent.chmod(0o700)
    root.mkdir(mode=0o700)
    lock = native.NativeProfileLock(str(root)).acquire()
    lock_device, lock_inode = lock.identity
    generation = "generation-durable"
    runtime = native._NativeRuntime(
        process=SimpleNamespace(pid=4242),
        process_start_time=100.0,
        cdp_url="http://127.0.0.1:43123",
        cdp_port=43123,
        snapshot_uuid="snapshot-1",
        executable_fingerprint="f" * 64,
        lock=lock,
        hermes_home=canonical_home,
        snapshot_path=str(snapshot),
        supervisor_path=str(root),
        source_profile_hash="source-hash",
        expected_account_hash="account-hash",
        lease_path=str(root / "runtime.json"),
        runtime_generation=generation,
        lock_device=lock_device,
        lock_inode=lock_inode,
    )
    native._write_runtime_lease(runtime)
    lock.release()
    monkeypatch.setattr(native, "_validate_stable_chrome", lambda: "f" * 64)
    monkeypatch.setattr(native, "_runtime_is_valid", lambda *_args: proof_is_valid)
    monkeypatch.setattr(native, "_refresh_provisional_runtime", lambda *_args: False)
    launch = Mock(side_effect=AssertionError("durable owner must not be bypassed"))
    monkeypatch.setattr(native.subprocess, "Popen", launch)
    if proof_is_valid:
        monkeypatch.setattr(
            native,
            "_prove_recorded_runtime",
            lambda _runtime: "ws://127.0.0.1:43123/devtools/browser/abc",
        )
    else:
        monkeypatch.setattr(
            native,
            "_prove_recorded_runtime",
            Mock(
                side_effect=native.NativeProfileError(
                    "native_cleanup_identity_ambiguous", "ambiguous"
                )
            ),
        )
        monkeypatch.setattr(native, "_recorded_process_is_absent", lambda _runtime: False)
    supervisor = native.NativeProfileSupervisor.for_profile(home)

    try:
        if proof_is_valid:
            client = supervisor.acquire(_NATIVE_CONFIG, {})
            assert client.runtime_namespace == generation
            assert native._runtimes[canonical_home].runtime_generation == generation
        else:
            with pytest.raises(native.NativeProfileError) as raised:
                supervisor.acquire(_NATIVE_CONFIG, {})
            assert raised.value.code == "native_cached_proof_failed"
            assert canonical_home not in native._runtimes
            assert (root / "runtime.json").exists()
        launch.assert_not_called()
    finally:
        native._client_leases.pop(canonical_home, None)
        adopted = native._runtimes.pop(canonical_home, None)
        if adopted is not None:
            adopted.lock.release()


def test_failed_final_durable_adoption_restores_generation_cleanup(
    monkeypatch, tmp_path
):
    import hermes_cli.native_real_profile as native

    home = tmp_path / "profile"
    home.mkdir()
    canonical_home, snapshot, root = native._profile_scope(home)
    root.parent.mkdir(parents=True, mode=0o700)
    root.parent.chmod(0o700)
    root.mkdir(mode=0o700)
    lock = native.NativeProfileLock(str(root)).acquire()
    lock_device, lock_inode = lock.identity
    generation = "generation-late-proof-failure"
    runtime = native._NativeRuntime(
        process=SimpleNamespace(pid=4242),
        process_start_time=100.0,
        cdp_url="http://127.0.0.1:43123",
        cdp_port=43123,
        snapshot_uuid="snapshot-1",
        executable_fingerprint="f" * 64,
        lock=lock,
        hermes_home=canonical_home,
        snapshot_path=str(snapshot),
        supervisor_path=str(root),
        source_profile_hash="source-hash",
        expected_account_hash="account-hash",
        lease_path=str(root / "runtime.json"),
        runtime_generation=generation,
        lock_device=lock_device,
        lock_inode=lock_inode,
    )
    native._write_runtime_lease(runtime)
    lock.release()
    monkeypatch.setattr(native, "_validate_stable_chrome", lambda: "f" * 64)
    monkeypatch.setattr(native, "_runtime_is_valid", lambda *_args: True)
    monkeypatch.setattr(native, "_refresh_provisional_runtime", lambda *_args: False)
    monkeypatch.setattr(
        native,
        "_prove_recorded_runtime",
        Mock(
            side_effect=native.NativeProfileError(
                "native_cleanup_identity_ambiguous", "late proof failed"
            )
        ),
    )
    scheduled = Mock()
    monkeypatch.setattr(native, "_schedule_native_profile_cleanup", scheduled)
    supervisor = native.NativeProfileSupervisor.for_profile(home)

    try:
        with pytest.raises(native.NativeProfileError) as raised:
            supervisor.acquire(_NATIVE_CONFIG, {})

        assert raised.value.code == "native_cleanup_identity_ambiguous"
        assert canonical_home not in native._client_leases
        assert native._runtimes[canonical_home].runtime_generation == generation
        assert (root / "runtime.json").exists()
        scheduled.assert_called_once_with(
            120.0,
            hermes_home=canonical_home,
            runtime_generation=generation,
        )
    finally:
        native._client_leases.pop(canonical_home, None)
        adopted = native._runtimes.pop(canonical_home, None)
        if adopted is not None:
            adopted.lock.release()


@pytest.mark.parametrize(
    ("listener_absent", "snapshot_owners", "expected_code"),
    [
        (False, [], "native_cached_proof_failed"),
        (True, [8888], "native_snapshot_owned"),
    ],
)
def test_stale_durable_adoption_retains_lease_until_all_absence_is_proven(
    monkeypatch, tmp_path, listener_absent, snapshot_owners, expected_code
):
    import hermes_cli.native_real_profile as native

    home = tmp_path / "profile"
    home.mkdir()
    canonical_home, snapshot, root = native._profile_scope(home)
    root.parent.mkdir(parents=True, mode=0o700)
    root.parent.chmod(0o700)
    root.mkdir(mode=0o700)
    lock = native.NativeProfileLock(str(root)).acquire()
    lock_device, lock_inode = lock.identity
    runtime = native._NativeRuntime(
        process=SimpleNamespace(pid=999999),
        process_start_time=100.0,
        cdp_url="http://127.0.0.1:43123",
        cdp_port=43123,
        snapshot_uuid="snapshot-1",
        executable_fingerprint="f" * 64,
        lock=lock,
        hermes_home=canonical_home,
        snapshot_path=str(snapshot),
        supervisor_path=str(root),
        source_profile_hash="source-hash",
        expected_account_hash="account-hash",
        lease_path=str(root / "runtime.json"),
        runtime_generation="generation-stale-listener",
        lock_device=lock_device,
        lock_inode=lock_inode,
    )
    native._write_runtime_lease(runtime)
    lock.release()
    monkeypatch.setattr(native, "_validate_stable_chrome", lambda: "f" * 64)
    monkeypatch.setattr(native, "_runtime_is_valid", lambda *_args: False)
    monkeypatch.setattr(native, "_refresh_provisional_runtime", lambda *_args: False)
    monkeypatch.setattr(
        native,
        "_prove_recorded_runtime",
        Mock(side_effect=native.NativeProfileError("native_cached_proof_failed", "stale")),
    )
    monkeypatch.setattr(native, "_recorded_process_is_absent", lambda _runtime: True)
    monkeypatch.setattr(
        native, "_recorded_listener_is_absent", lambda _port: listener_absent
    )
    monkeypatch.setattr(
        native,
        "_processes_owning_data_dir",
        lambda path, **_kwargs: snapshot_owners if path == str(snapshot) else [],
    )
    provision = Mock(side_effect=AssertionError("ambiguous listener must block provisioning"))
    monkeypatch.setattr(native, "provision_native_snapshot", provision)
    launch = Mock(side_effect=AssertionError("ambiguous listener must block relaunch"))
    monkeypatch.setattr(native.subprocess, "Popen", launch)

    with pytest.raises(native.NativeProfileError) as raised:
        native.NativeProfileSupervisor.for_profile(home).acquire(_NATIVE_CONFIG, {})

    assert raised.value.code == expected_code
    assert (root / "runtime.json").exists()
    provision.assert_not_called()
    launch.assert_not_called()


def test_native_browser_exec_never_consults_or_publishes_builtin_session_cache(
    monkeypatch,
):
    import hermes_cli.native_real_profile as native
    import tools.browser_tool as built_in
    import tools.browser_use_cli as browser_use
    from tools.environments import local as local_env
    from tools.process_registry import ProcessRegistry

    foreign = {"task-1": {"cdp_url": "https://foreign.example"}}
    monkeypatch.setattr(built_in, "_active_sessions", foreign)
    terminate = Mock(side_effect=AssertionError("native lane must never signal pid files"))
    monkeypatch.setattr(ProcessRegistry, "_terminate_host_pid", terminate)
    client = SimpleNamespace(
        hermes_home="/tmp/hermes-native-test",
        cdp_url="http://127.0.0.1:43123",
        runtime_namespace="generation-9",
        token="client-3",
    )
    supervisor = Mock(acquire=Mock(return_value=client))
    monkeypatch.setattr(
        native.NativeProfileSupervisor,
        "for_profile",
        classmethod(lambda cls, home=None: supervisor),
    )
    monkeypatch.setattr(browser_use, "_read_browser_cfg", lambda **_kw: dict(_NATIVE_CONFIG))
    monkeypatch.setattr(browser_use, "_find_cli", lambda: ["browser-use"])
    monkeypatch.setattr(browser_use, "_blocked_url_in_code", lambda _code: None)
    monkeypatch.setattr(
        local_env,
        "hermes_subprocess_env",
        lambda **_kwargs: {"PATH": "/usr/bin", "HOME": "/tmp/home"},
    )
    monkeypatch.setattr(
        browser_use.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="ok", stderr=""
        ),
    )

    payload = _tool_payload(
        browser_use.browser_exec("print('ok')", task_id="task-1")
    )

    assert payload["success"] is True
    assert built_in._active_sessions is foreign
    assert foreign == {"task-1": {"cdp_url": "https://foreign.example"}}
    terminate.assert_not_called()
    supervisor.release.assert_called_once_with(client)


def test_native_lifecycle_never_calls_process_registry_or_agent_browser_kill(
    monkeypatch, tmp_path
):
    import hermes_cli.native_real_profile as native
    from tools.process_registry import ProcessRegistry

    home = tmp_path / "profile"
    home.mkdir()
    stale_registry = home / "agent-browser" / "processes"
    stale_registry.mkdir(parents=True)
    (stale_registry / "native.json").write_text(
        json.dumps({"pid": os.getpid()}), encoding="utf-8"
    )
    signal = Mock(side_effect=AssertionError("native lifecycle cannot signal PID files"))
    monkeypatch.setattr(ProcessRegistry, "_terminate_host_pid", signal)
    cleanup = Mock()
    monkeypatch.setattr(native, "_cleanup_native_profile", cleanup)
    supervisor = native.NativeProfileSupervisor.for_profile(home)

    supervisor.cleanup()
    native.NativeProfileSupervisor.cleanup_all()

    signal.assert_not_called()
    assert cleanup.call_count >= 2
    assert (stale_registry / "native.json").exists()


def test_revocation_cannot_reenter_native_daemon_namespace(
    monkeypatch, tmp_path
):
    import hermes_cli.native_real_profile as native
    import tools.browser_use_cli as browser_use

    home = tmp_path / "profile"
    state_root = home / "browser-supervisor" / "native-real-profile"
    (state_root / "snapshot").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    revoked = {
        **_NATIVE_CONFIG,
        "use_real_profile": False,
        "real_profile_macos_native": False,
    }
    supervisor = Mock()
    monkeypatch.setattr(
        native.NativeProfileSupervisor,
        "for_profile",
        classmethod(lambda cls, profile=None: supervisor),
    )
    monkeypatch.setattr(browser_use, "_read_browser_cfg", lambda **_kw: revoked)
    monkeypatch.setattr(browser_use, "_blocked_url_in_code", lambda _code: None)
    launch = Mock(side_effect=AssertionError("revocation must refuse before launch"))
    monkeypatch.setattr(browser_use, "_find_cli", launch)

    payload = _tool_payload(
        browser_use.browser_exec("print('must not run')", session="work")
    )

    assert "consent" in payload["error"].lower()
    supervisor.cleanup.assert_called_once_with(delete_snapshot=True)
    supervisor.acquire.assert_not_called()
    launch.assert_not_called()


def test_native_toggle_revocation_cleans_state_before_generic_route(
    monkeypatch, tmp_path
):
    import hermes_cli.native_real_profile as native
    import tools.browser_use_cli as browser_use

    home = tmp_path / "profile"
    state_root = home / "browser-supervisor" / "native-real-profile"
    (state_root / "snapshot").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    revoked = {
        **_NATIVE_CONFIG,
        "use_real_profile": True,
        "real_profile_macos_native": False,
    }
    supervisor = Mock()
    monkeypatch.setattr(
        native.NativeProfileSupervisor,
        "for_profile",
        classmethod(lambda cls, profile=None: supervisor),
    )
    monkeypatch.setattr(browser_use, "_read_browser_cfg", lambda **_kw: revoked)
    monkeypatch.setattr(browser_use, "_blocked_url_in_code", lambda _code: None)
    generic_route = Mock(
        side_effect=AssertionError("native revocation must refuse the generic route")
    )
    monkeypatch.setattr(browser_use, "_base_subprocess_env", generic_route)

    payload = _tool_payload(browser_use.browser_exec("print('must not run')"))

    assert "new session" in payload["error"].lower()
    supervisor.cleanup.assert_called_once_with(delete_snapshot=True)
    supervisor.acquire.assert_not_called()
    generic_route.assert_not_called()


def test_native_false_cli_unavailable_precedes_legacy_real_profile_resolution(
    monkeypatch, tmp_path
):
    import tools.browser_use_cli as browser_use

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    config = {
        **_NATIVE_CONFIG,
        "use_real_profile": True,
        "real_profile_macos_native": False,
    }
    monkeypatch.setattr(browser_use, "_read_browser_cfg", lambda **_kw: config)
    monkeypatch.setattr(browser_use, "_blocked_url_in_code", lambda _code: None)
    monkeypatch.setattr(browser_use, "_find_cli", lambda: None)
    legacy_resolution = Mock(
        side_effect=AssertionError("CLI availability must precede browser resolution")
    )
    monkeypatch.setattr(browser_use, "_resolve_real_profile_cdp", legacy_resolution)

    payload = _tool_payload(browser_use.browser_exec("print('must not run')"))

    assert "CLI not found" in payload["error"]
    legacy_resolution.assert_not_called()


@pytest.mark.parametrize("contents", ["browser: [", "not-a-mapping\n"])
def test_native_false_without_state_preserves_tolerant_generic_config_read(
    monkeypatch, tmp_path, contents
):
    import hermes_cli.config as config
    import tools.browser_use_cli as browser_use

    home = tmp_path / "profile"
    home.mkdir()
    config_path = home / "config.yaml"
    config_path.write_text(contents, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(config, "get_config_path", lambda: config_path)
    monkeypatch.setattr(browser_use, "_blocked_url_in_code", lambda _code: None)
    monkeypatch.setattr(browser_use, "_find_cli", lambda: ["browser-use"])
    monkeypatch.setattr(browser_use, "_base_subprocess_env", lambda: {})
    monkeypatch.setattr(browser_use, "_resolve_real_profile_cdp", lambda *_args, **_kw: None)
    monkeypatch.setattr(browser_use, "_resolve_backend_cdp", lambda *_args, **_kw: None)
    monkeypatch.setattr(
        browser_use.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="generic", stderr=""
        ),
    )

    payload = _tool_payload(browser_use.browser_exec("print('generic')"))

    assert payload["success"] is True
    assert payload["output"] == "generic"


def test_corrupt_config_with_native_state_fails_before_cleanup_or_generic_route(
    monkeypatch, tmp_path
):
    import hermes_cli.config as config
    import hermes_cli.native_real_profile as native
    import tools.browser_use_cli as browser_use

    home = tmp_path / "profile"
    state_root = home / "browser-supervisor" / "native-real-profile"
    (state_root / "snapshot").mkdir(parents=True)
    config_path = home / "config.yaml"
    config_path.write_text("browser: [", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(config, "get_config_path", lambda: config_path)
    monkeypatch.setattr(browser_use, "_blocked_url_in_code", lambda _code: None)
    supervisor = Mock()
    monkeypatch.setattr(
        native.NativeProfileSupervisor,
        "for_profile",
        classmethod(lambda cls, profile=None: supervisor),
    )
    generic_route = Mock(
        side_effect=AssertionError("corrupt native state must fail before generic routing")
    )
    monkeypatch.setattr(browser_use, "_find_cli", generic_route)

    payload = _tool_payload(browser_use.browser_exec("print('must not run')"))

    assert "configuration is unavailable" in payload["error"].lower()
    supervisor.cleanup.assert_not_called()
    generic_route.assert_not_called()


def test_user_session_cannot_claim_reserved_native_namespace(monkeypatch):
    import hermes_cli.native_real_profile as native
    import tools.browser_use_cli as browser_use

    supervisor = Mock()
    monkeypatch.setattr(
        native.NativeProfileSupervisor,
        "for_profile",
        classmethod(lambda cls, profile=None: supervisor),
    )
    monkeypatch.setattr(browser_use, "_read_browser_cfg", lambda **_kw: dict(_NATIVE_CONFIG))
    monkeypatch.setattr(browser_use, "_blocked_url_in_code", lambda _code: None)

    payload = _tool_payload(
        browser_use.browser_exec("print('x')", session="hermes_native_stolen")
    )

    assert "reserved" in payload["error"].lower()
    supervisor.acquire.assert_not_called()


def test_revocation_propagates_typed_cleanup_ambiguity(monkeypatch, tmp_path):
    import hermes_cli.native_real_profile as native
    import tools.browser_use_cli as browser_use

    home = tmp_path / "profile"
    snapshot = home / "browser-supervisor" / "native-real-profile" / "snapshot"
    snapshot.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    supervisor = Mock()
    supervisor.cleanup.side_effect = native.NativeProfileError(
        "native_cleanup_owner_ambiguous", "exact owner is ambiguous"
    )
    monkeypatch.setattr(
        native.NativeProfileSupervisor,
        "for_profile",
        classmethod(lambda cls, profile=None: supervisor),
    )
    monkeypatch.setattr(
        browser_use,
        "_read_browser_cfg",
        lambda **_kw: {
            **_NATIVE_CONFIG,
            "use_real_profile": False,
            "real_profile_macos_native": False,
        },
    )
    monkeypatch.setattr(browser_use, "_blocked_url_in_code", lambda _code: None)
    launch = Mock(side_effect=AssertionError("ambiguity must stop launch"))
    monkeypatch.setattr(browser_use, "_find_cli", launch)

    payload = _tool_payload(browser_use.browser_exec("print('x')"))

    assert "native_cleanup_owner_ambiguous" in payload["error"]
    assert snapshot.exists()
    launch.assert_not_called()


def test_descriptive_google_identity_cannot_replace_fixed_requirement(monkeypatch):
    import hermes_cli.native_real_profile as native

    descriptive = (
        "Identifier=com.google.Chrome\n"
        "TeamIdentifier=EQHXZ8M8AV\n"
        'designated => identifier "com.google.Chrome" and anchor apple generic '
        "and certificate leaf[subject.OU] = EQHXZ8M8AV"
    )
    native.validate_codesign_output(descriptive)
    monkeypatch.setattr(
        native,
        "_security_check_guest_requirement",
        lambda pid, requirement: False,
        raising=False,
    )
    cli_codesign = Mock(
        side_effect=AssertionError(
            "live PID authority must be Security.framework, not codesign text"
        )
    )
    monkeypatch.setattr(native.subprocess, "run", cli_codesign)

    with pytest.raises(native.NativeProfileError) as raised:
        native._validate_live_process_signature(4242)

    assert raised.value.code == "native_live_signature_invalid"
    cli_codesign.assert_not_called()


def test_live_pid_requirement_failure_blocks_adopt_close_and_signal(
    monkeypatch, tmp_path
):
    import psutil

    import hermes_cli.native_real_profile as native

    home = tmp_path / "profile"
    home.mkdir()
    canonical_home, snapshot, supervisor_root = native._profile_scope(home)
    snapshot.mkdir(parents=True, mode=0o700)
    supervisor_root.parent.chmod(0o700)
    supervisor_root.chmod(0o700)
    snapshot.chmod(0o700)
    fingerprint = "f" * 64
    manifest = {
        "version": 1,
        "snapshot_uuid": "snapshot-1",
        "source_profile_hash": "source-hash",
        "expected_account_hash": "account-hash",
        "executable_fingerprint": fingerprint,
    }
    manifest_path = snapshot / native._MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_path.chmod(0o600)
    default = snapshot / "Default"
    default.mkdir(mode=0o700)
    credential = default / "Cookies"
    credential.write_bytes(b"credential")
    credential.chmod(0o600)
    process = Mock(
        pid=4242,
        create_time=Mock(return_value=100.0),
        exe=Mock(return_value=native.STABLE_CHROME_EXECUTABLE),
        cmdline=Mock(
            return_value=native.native_chrome_argv(
                native.STABLE_CHROME_EXECUTABLE, str(snapshot)
            )
        ),
    )
    runtime = native._NativeRuntime(
        process=process,
        process_start_time=100.0,
        cdp_url="http://127.0.0.1:43123",
        cdp_port=43123,
        snapshot_uuid="snapshot-1",
        executable_fingerprint=fingerprint,
        lock=Mock(identity=(11, 12)),
        hermes_home=canonical_home,
        snapshot_path=str(snapshot),
        supervisor_path=str(supervisor_root),
        source_profile_hash="source-hash",
        expected_account_hash="account-hash",
        executable_path=native.STABLE_CHROME_EXECUTABLE,
        lease_path=str(supervisor_root / "runtime.json"),
        runtime_generation="generation-1",
        lock_device=11,
        lock_inode=12,
    )
    native._runtimes[canonical_home] = runtime
    monkeypatch.setattr(psutil, "Process", lambda _pid: process)
    monkeypatch.setattr(native, "_validate_stable_chrome", lambda: fingerprint)
    live_requirement = Mock(return_value=False)
    monkeypatch.setattr(
        native, "_security_check_guest_requirement", live_requirement, raising=False
    )
    browser_close = Mock(
        side_effect=AssertionError("failed live requirement must block Browser.close")
    )
    monkeypatch.setitem(
        sys.modules,
        "websocket",
        SimpleNamespace(create_connection=browser_close),
    )

    try:
        with pytest.raises(native.NativeProfileError) as raised:
            native.NativeProfileSupervisor.for_profile(home).cleanup()
        assert raised.value.code == "native_cleanup_identity_ambiguous"
        browser_close.assert_not_called()
        process.terminate.assert_not_called()
        live_requirement.assert_called_once_with(
            4242, native.STABLE_CHROME_REQUIREMENT
        )
        assert native._runtimes[canonical_home] is runtime
        assert snapshot.exists()
    finally:
        native._runtimes.pop(canonical_home, None)


def test_adoption_rejects_kernel_executable_drift(monkeypatch, tmp_path):
    import psutil

    import hermes_cli.native_real_profile as native

    process = SimpleNamespace(
        pid=4242,
        create_time=lambda: 100.0,
        exe=lambda: "/tmp/not-google-chrome",
        cmdline=lambda: ["chrome", f"--user-data-dir={tmp_path}"],
    )
    runtime = SimpleNamespace(
        process=SimpleNamespace(pid=4242),
        process_start_time=100.0,
        executable_fingerprint="f" * 64,
        snapshot_uuid="snapshot-1",
        cdp_port=43123,
    )
    monkeypatch.setattr(psutil, "Process", lambda _pid: process)
    monkeypatch.setattr(
        native,
        "_manifest_for",
        lambda *_args: {"snapshot_uuid": "snapshot-1"},
    )
    signature = Mock()
    monkeypatch.setattr(native, "_validate_live_process_signature", signature)

    assert (
        native._runtime_is_valid(
            runtime,
            native.NativeProfileRequest("chrome", "Profile 1", "worker@example.test"),
            tmp_path,
            "f" * 64,
        )
        is False
    )
    signature.assert_not_called()


def test_browser_close_boundary_rejects_pid_reuse_before_endpoint(monkeypatch):
    import psutil

    import hermes_cli.native_real_profile as native

    process = Mock(
        pid=4242,
        create_time=Mock(return_value=101.0),
    )
    monkeypatch.setattr(psutil, "Process", lambda _pid: process)
    websocket_open = Mock()
    monkeypatch.setitem(
        sys.modules,
        "websocket",
        SimpleNamespace(create_connection=websocket_open),
    )
    runtime = SimpleNamespace(
        process=SimpleNamespace(pid=4242),
        process_start_time=100.0,
        snapshot_path="/private/tmp/native-snapshot",
        cdp_port=43123,
    )

    with pytest.raises(native.NativeProfileError) as raised:
        native._close_recorded_runtime(runtime)

    assert raised.value.code == "native_pid_reused"
    websocket_open.assert_not_called()
    process.terminate.assert_not_called()


def test_sigterm_boundary_rejects_pid_reuse_after_browser_close(monkeypatch):
    import psutil

    import hermes_cli.native_real_profile as native

    before = Mock(
        pid=4242,
        create_time=Mock(return_value=100.0),
        wait=Mock(side_effect=psutil.TimeoutExpired(5)),
    )
    reused = Mock(pid=4242, create_time=Mock(return_value=101.0))
    processes = iter((before, reused))
    monkeypatch.setattr(psutil, "Process", lambda _pid: next(processes))
    monkeypatch.setattr(
        native,
        "_prove_recorded_runtime",
        lambda _runtime: "ws://127.0.0.1:43123/devtools/browser/abc",
    )
    websocket = Mock()
    websocket_open = Mock(return_value=websocket)
    monkeypatch.setitem(
        sys.modules,
        "websocket",
        SimpleNamespace(create_connection=websocket_open),
    )
    runtime = SimpleNamespace(
        process=SimpleNamespace(pid=4242),
        process_start_time=100.0,
        snapshot_path="/private/tmp/native-snapshot",
        cdp_port=43123,
    )

    with pytest.raises(native.NativeProfileError) as raised:
        native._close_recorded_runtime(runtime)

    assert raised.value.code == "native_pid_reused"
    websocket_open.assert_called_once()
    before.terminate.assert_not_called()
    reused.terminate.assert_not_called()


def test_sigterm_boundary_rejects_executable_drift_after_browser_close(monkeypatch):
    import psutil

    import hermes_cli.native_real_profile as native

    before = Mock(
        pid=4242,
        create_time=Mock(return_value=100.0),
        wait=Mock(side_effect=psutil.TimeoutExpired(5)),
    )
    drifted = Mock(
        pid=4242,
        create_time=Mock(return_value=100.0),
        exe=Mock(return_value="/tmp/not-google-chrome"),
    )
    processes = iter((before, drifted))
    monkeypatch.setattr(psutil, "Process", lambda _pid: next(processes))
    monkeypatch.setattr(
        native,
        "_prove_recorded_runtime",
        lambda _runtime: "ws://127.0.0.1:43123/devtools/browser/abc",
    )
    websocket = Mock()
    monkeypatch.setitem(
        sys.modules,
        "websocket",
        SimpleNamespace(create_connection=Mock(return_value=websocket)),
    )
    runtime = SimpleNamespace(
        process=SimpleNamespace(pid=4242),
        process_start_time=100.0,
        snapshot_path="/private/tmp/native-snapshot",
        executable_path=native.STABLE_CHROME_EXECUTABLE,
        cdp_port=43123,
    )

    with pytest.raises(native.NativeProfileError) as raised:
        native._close_recorded_runtime(runtime)

    assert raised.value.code == "native_cleanup_identity_ambiguous"
    before.terminate.assert_not_called()
    drifted.terminate.assert_not_called()


def test_failed_acquire_rolls_back_exact_token_and_restores_zero_client_cleanup(
    monkeypatch, tmp_path
):
    import hermes_cli.native_real_profile as native

    home = tmp_path / "profile"
    home.mkdir()
    supervisor = native.NativeProfileSupervisor.for_profile(home)
    runtime = SimpleNamespace(
        cdp_url="http://127.0.0.1:43123",
        runtime_generation="generation-rollback",
    )
    native._runtimes[supervisor.hermes_home] = runtime
    old_timer = Mock()
    native._cleanup_timers[supervisor.hermes_home] = old_timer
    observed_tokens = []

    def fail_after_token(*_args, **_kwargs):
        observed_tokens.extend(native._client_leases[supervisor.hermes_home])
        raise native.NativeProfileError("native_profile_busy", "busy")

    created_timers = []

    class FakeTimer:
        def __init__(self, delay, callback):
            self.delay = delay
            self.callback = callback
            self.started = False
            self.daemon = False
            created_timers.append(self)

        def start(self):
            self.started = True

        def cancel(self):
            return None

    monkeypatch.setattr(native, "_resolve_native_profile_cdp", fail_after_token)
    monkeypatch.setattr(native.threading, "Timer", FakeTimer)

    try:
        with pytest.raises(native.NativeProfileError) as raised:
            supervisor.acquire(_NATIVE_CONFIG, {})
        assert raised.value.code == "native_profile_busy"
        assert len(observed_tokens) == 1
        assert native._client_leases.get(supervisor.hermes_home) is None
        old_timer.cancel.assert_called_once()
        assert len(created_timers) == 1
        assert created_timers[0].started is True
        assert native._cleanup_timers[supervisor.hermes_home] is created_timers[0]
    finally:
        native._runtimes.pop(supervisor.hermes_home, None)
        native._cleanup_timers.pop(supervisor.hermes_home, None)
        native._client_leases.pop(supervisor.hermes_home, None)


def test_failed_acquire_removes_only_its_provisional_token(monkeypatch, tmp_path):
    import hermes_cli.native_real_profile as native

    home = tmp_path / "profile"
    home.mkdir()
    supervisor = native.NativeProfileSupervisor.for_profile(home)
    native._client_leases[supervisor.hermes_home] = {"existing-client"}
    monkeypatch.setattr(
        native,
        "_resolve_native_profile_cdp",
        Mock(side_effect=native.NativeProfileError("native_profile_busy", "busy")),
    )

    try:
        with pytest.raises(native.NativeProfileError):
            supervisor.acquire(_NATIVE_CONFIG, {})
        assert native._client_leases[supervisor.hermes_home] == {"existing-client"}
    finally:
        native._client_leases.pop(supervisor.hermes_home, None)


def test_acquire_installs_token_before_adopt_or_launch_and_excludes_cleanup(
    monkeypatch, tmp_path
):
    import hermes_cli.native_real_profile as native

    home = tmp_path / "profile"
    home.mkdir()
    supervisor = native.NativeProfileSupervisor.for_profile(home)
    runtime = SimpleNamespace(
        cdp_url="http://127.0.0.1:43123",
        runtime_generation="generation-atomic",
    )
    native._runtimes[supervisor.hermes_home] = runtime
    entered = threading.Event()
    allow_return = threading.Event()
    acquired = []
    cleanup_errors = []

    def paused_resolution(*_args, **_kwargs):
        assert len(native._client_leases[supervisor.hermes_home]) == 1
        entered.set()
        assert allow_return.wait(5)
        return runtime.cdp_url

    monkeypatch.setattr(native, "_resolve_native_profile_cdp", paused_resolution)
    monkeypatch.setattr(native, "_schedule_native_profile_cleanup", Mock())

    acquire_thread = threading.Thread(
        target=lambda: acquired.append(supervisor.acquire(_NATIVE_CONFIG, {}))
    )
    cleanup_thread = None

    try:
        acquire_thread.start()
        assert entered.wait(5)

        def run_cleanup():
            try:
                supervisor.cleanup()
            except Exception as exc:
                cleanup_errors.append(exc)

        cleanup_thread = threading.Thread(target=run_cleanup)
        cleanup_thread.start()
        assert cleanup_thread.is_alive()
        allow_return.set()
        acquire_thread.join(5)
        cleanup_thread.join(5)

        assert len(acquired) == 1
        assert len(cleanup_errors) == 1
        assert cleanup_errors[0].code == "native_clients_active"
        supervisor.release(acquired[0])
    finally:
        allow_return.set()
        acquire_thread.join(5)
        if cleanup_thread is not None:
            cleanup_thread.join(5)
        native._runtimes.pop(supervisor.hermes_home, None)
        native._client_leases.pop(supervisor.hermes_home, None)


def test_conflicting_acquire_cannot_close_runtime_held_by_another_client(
    monkeypatch, tmp_path
):
    import hermes_cli.native_real_profile as native

    home = tmp_path / "profile"
    home.mkdir()
    supervisor = native.NativeProfileSupervisor.for_profile(home)
    runtime = SimpleNamespace(
        cdp_url="http://127.0.0.1:43123",
        runtime_generation="generation-live",
    )
    native._runtimes[supervisor.hermes_home] = runtime
    native._client_leases[supervisor.hermes_home] = {"existing-client"}
    monkeypatch.setattr(native, "_validate_stable_chrome", lambda: "f" * 64)
    monkeypatch.setattr(native, "_runtime_is_valid", lambda *_args: False)
    monkeypatch.setattr(native, "_refresh_provisional_runtime", lambda *_args: False)
    close = Mock(side_effect=AssertionError("another client's runtime must stay open"))
    monkeypatch.setattr(native, "_close_recorded_runtime", close)

    try:
        with pytest.raises(native.NativeProfileError) as raised:
            supervisor.acquire(_NATIVE_CONFIG, {})
        assert raised.value.code == "native_runtime_in_use"
        assert native._client_leases[supervisor.hermes_home] == {"existing-client"}
        close.assert_not_called()
    finally:
        native._runtimes.pop(supervisor.hermes_home, None)
        native._client_leases.pop(supervisor.hermes_home, None)


def test_two_profiles_have_independent_clients_generations_and_cleanup(
    monkeypatch, tmp_path
):
    import hermes_cli.native_real_profile as native

    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()
    supervisor_a = native.NativeProfileSupervisor.for_profile(home_a)
    supervisor_b = native.NativeProfileSupervisor.for_profile(home_b)
    runtime_a = SimpleNamespace(
        cdp_url="http://127.0.0.1:43121",
        runtime_generation="generation-a",
    )
    runtime_b = SimpleNamespace(
        cdp_url="http://127.0.0.1:43122",
        runtime_generation="generation-b",
    )
    native._runtimes[supervisor_a.hermes_home] = runtime_a
    native._runtimes[supervisor_b.hermes_home] = runtime_b
    schedule = Mock()
    monkeypatch.setattr(native, "_schedule_native_profile_cleanup", schedule)
    monkeypatch.setattr(
        native,
        "_resolve_native_profile_cdp",
        lambda *_args, hermes_home=None, **_kwargs: native._runtimes[
            os.path.realpath(str(hermes_home))
        ].cdp_url,
    )

    try:
        client_a = supervisor_a.acquire(_NATIVE_CONFIG, {})
        client_b = supervisor_b.acquire(_NATIVE_CONFIG, {})
        assert client_a.hermes_home != client_b.hermes_home
        assert client_a.cdp_url == runtime_a.cdp_url
        assert client_b.cdp_url == runtime_b.cdp_url
        assert client_a.runtime_namespace == "generation-a"
        assert client_b.runtime_namespace == "generation-b"
        assert native._client_leases[client_a.hermes_home] == {client_a.token}
        assert native._client_leases[client_b.hermes_home] == {client_b.token}

        supervisor_a.release(client_a)
        assert client_a.hermes_home not in native._client_leases
        assert native._client_leases[client_b.hermes_home] == {client_b.token}
        schedule.assert_called_once_with(
            client_a.inactivity_delay,
            hermes_home=client_a.hermes_home,
            runtime_generation="generation-a",
        )

        supervisor_b.release(client_b)
        assert client_b.hermes_home not in native._client_leases
        assert schedule.call_args_list[-1].kwargs == {
            "hermes_home": client_b.hermes_home,
            "runtime_generation": "generation-b",
        }
    finally:
        for home in (supervisor_a.hermes_home, supervisor_b.hermes_home):
            native._runtimes.pop(home, None)
            native._client_leases.pop(home, None)


def test_cleanup_all_drains_every_instantiated_profile_after_one_failure(
    monkeypatch, tmp_path
):
    import hermes_cli.native_real_profile as native

    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()
    supervisor_a = native.NativeProfileSupervisor.for_profile(home_a)
    supervisor_b = native.NativeProfileSupervisor.for_profile(home_b)
    native._client_leases[supervisor_a.hermes_home] = {"client-a"}
    native._client_leases[supervisor_b.hermes_home] = {"client-b"}
    timer_a = Mock()
    timer_b = Mock()
    native._cleanup_timers[supervisor_a.hermes_home] = timer_a
    native._cleanup_timers[supervisor_b.hermes_home] = timer_b
    calls = []

    def cleanup(*, delete_snapshot, hermes_home):
        calls.append((delete_snapshot, hermes_home))
        if hermes_home == supervisor_a.hermes_home:
            raise native.NativeProfileError("native_cleanup_uncertain", "ambiguous")

    monkeypatch.setattr(native, "_cleanup_native_profile", cleanup)

    with pytest.raises(native.NativeProfileError) as raised:
        native.NativeProfileSupervisor.cleanup_all()

    assert raised.value.code == "native_cleanup_uncertain"
    assert (False, supervisor_a.hermes_home) in calls
    assert (False, supervisor_b.hermes_home) in calls
    assert supervisor_a.hermes_home not in native._client_leases
    assert supervisor_b.hermes_home not in native._client_leases
    timer_a.cancel.assert_called_once()
    timer_b.cancel.assert_called_once()


def test_cli_and_oneshot_shutdown_have_separate_native_cleanup(monkeypatch):
    import cli
    import hermes_cli.main as main
    import hermes_cli.native_real_profile as native

    calls = []
    monkeypatch.setattr(
        native.NativeProfileSupervisor,
        "cleanup_all",
        classmethod(lambda cls: calls.append("native")),
    )
    monkeypatch.setattr(cli, "_cleanup_all_terminals", lambda: None)
    monkeypatch.setattr(cli, "_cleanup_all_browsers", lambda: calls.append("browser"))
    monkeypatch.setattr(cli, "_arm_exit_watchdog", lambda: None)
    monkeypatch.setattr(cli, "_reset_terminal_input_modes_on_exit", lambda: None)
    monkeypatch.setattr("tools.async_delegation.interrupt_all", lambda **_kwargs: None)
    monkeypatch.setattr("tools.mcp_tool.shutdown_mcp_servers", lambda: None)
    monkeypatch.setattr("agent.auxiliary_client.shutdown_cached_clients", lambda: None)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_args, **_kwargs: None)
    original_cleanup_done = cli._cleanup_done
    cli._cleanup_done = False
    try:
        cli._run_cleanup(notify_session_finalize=False)
    finally:
        cli._cleanup_done = original_cleanup_done

    assert calls.count("native") == 1
    assert calls.count("browser") == 1

    calls.clear()
    monkeypatch.setattr("tools.terminal_tool.cleanup_all_environments", lambda: None)
    monkeypatch.setattr(
        "tools.browser_tool._emergency_cleanup_all_sessions",
        lambda: calls.append("browser"),
    )
    original_oneshot_done = main._oneshot_cleanup_done
    main._oneshot_cleanup_done = False
    try:
        main._cleanup_oneshot_runtime()
    finally:
        main._oneshot_cleanup_done = original_oneshot_done

    assert calls.count("native") == 1
    assert calls.count("browser") == 1


def test_cleanup_holds_flock_through_final_scan_and_fd_delete(
    monkeypatch, tmp_path
):
    import hermes_cli.native_real_profile as native

    home = tmp_path / "profile"
    home.mkdir()
    supervisor = native.NativeProfileSupervisor.for_profile(home)
    _canonical, snapshot, supervisor_root = native._profile_scope(home)
    snapshot.mkdir(parents=True, mode=0o700)
    supervisor_root.parent.chmod(0o700)
    supervisor_root.chmod(0o700)
    snapshot.chmod(0o700)
    (snapshot / "credential-copy").write_text("copy", encoding="utf-8")
    (snapshot / "credential-copy").chmod(0o600)
    entered_delete = threading.Event()
    allow_delete = threading.Event()
    cleanup_errors = []
    original_delete = native._delete_validated_snapshot

    def paused_delete(path, expected_identity):
        entered_delete.set()
        assert allow_delete.wait(5)
        return original_delete(path, expected_identity)

    monkeypatch.setattr(native, "_processes_owning_data_dir", lambda *_args: [])
    monkeypatch.setattr(native, "_delete_validated_snapshot", paused_delete)

    def run_cleanup():
        try:
            supervisor.cleanup(delete_snapshot=True)
        except Exception as exc:
            cleanup_errors.append(exc)

    thread = threading.Thread(target=run_cleanup)
    thread.start()
    try:
        assert entered_delete.wait(5)
        with pytest.raises(native.NativeProfileError) as raised:
            native.NativeProfileLock(str(supervisor_root)).acquire()
        assert raised.value.code == "native_profile_busy"
        assert snapshot.exists()
        allow_delete.set()
        thread.join(5)
        assert cleanup_errors == []
        assert not snapshot.exists()
        with native.NativeProfileLock(str(supervisor_root)):
            pass
    finally:
        allow_delete.set()
        thread.join(5)


def test_snapshot_cleanup_retains_hardlinked_credential(monkeypatch, tmp_path):
    import hermes_cli.native_real_profile as native

    home = tmp_path / "profile"
    home.mkdir()
    supervisor = native.NativeProfileSupervisor.for_profile(home)
    _canonical, snapshot, root = native._profile_scope(home)
    snapshot.mkdir(parents=True, mode=0o700)
    root.parent.chmod(0o700)
    root.chmod(0o700)
    snapshot.chmod(0o700)
    backing = home / "credential-backing"
    backing.write_text("credential", encoding="utf-8")
    backing.chmod(0o600)
    valid = snapshot / "Local State"
    valid.write_text("state", encoding="utf-8")
    valid.chmod(0o600)
    os.link(backing, snapshot / "Cookies")
    monkeypatch.setattr(native, "_processes_owning_data_dir", lambda *_args: [])

    with pytest.raises(native.NativeProfileError) as raised:
        supervisor.cleanup(delete_snapshot=True)

    assert raised.value.code == "native_snapshot_entry_invalid"
    assert snapshot.exists()
    assert backing.exists()
    assert valid.exists()


@pytest.mark.parametrize("unsafe_kind", ["symlink", "wrong-mode"])
def test_snapshot_root_unsafe_identity_is_retained(monkeypatch, tmp_path, unsafe_kind):
    import hermes_cli.native_real_profile as native

    home = tmp_path / "profile"
    home.mkdir()
    supervisor = native.NativeProfileSupervisor.for_profile(home)
    _canonical, snapshot, root = native._profile_scope(home)
    root.mkdir(parents=True, mode=0o700)
    root.parent.chmod(0o700)
    root.chmod(0o700)
    if unsafe_kind == "symlink":
        backing = home / "backing"
        backing.mkdir(mode=0o700)
        snapshot.symlink_to(backing, target_is_directory=True)
    else:
        snapshot.mkdir(mode=0o755)
        snapshot.chmod(0o755)
    monkeypatch.setattr(native, "_processes_owning_data_dir", lambda *_args: [])

    with pytest.raises(native.NativeProfileError) as raised:
        supervisor.cleanup(delete_snapshot=True)

    assert raised.value.code == "native_snapshot_unsafe"
    assert os.path.lexists(snapshot)


@pytest.mark.parametrize("unsafe_kind", ["wrong-mode", "hardlink"])
def test_acquire_retains_unsafe_existing_snapshot_before_source_read(
    monkeypatch, tmp_path, unsafe_kind
):
    import hermes_cli.native_real_profile as native

    home = tmp_path / "profile"
    home.mkdir()
    supervisor = native.NativeProfileSupervisor.for_profile(home)
    _canonical, snapshot, root = native._profile_scope(home)
    snapshot.mkdir(parents=True, mode=0o700)
    root.parent.chmod(0o700)
    root.chmod(0o700)
    if unsafe_kind == "wrong-mode":
        snapshot.chmod(0o755)
    else:
        backing = home / "credential-backing"
        backing.write_text("credential", encoding="utf-8")
        backing.chmod(0o600)
        os.link(backing, snapshot / "Cookies")
    monkeypatch.setattr(native, "_validate_stable_chrome", lambda: "f" * 64)
    monkeypatch.setattr(native, "_processes_owning_data_dir", lambda *_args, **_kwargs: [])
    provision = Mock(side_effect=AssertionError("unsafe snapshot must be retained"))
    monkeypatch.setattr(native, "provision_native_snapshot", provision)

    with pytest.raises(native.NativeProfileError) as raised:
        supervisor.acquire(_NATIVE_CONFIG, {})

    assert raised.value.code == "native_snapshot_unsafe"
    provision.assert_not_called()
    assert snapshot.exists()


@pytest.mark.parametrize("unsafe_kind", ["symlink", "wrong-mode"])
def test_runtime_lease_unsafe_identity_is_retained(tmp_path, unsafe_kind):
    import hermes_cli.native_real_profile as native

    home = tmp_path / "profile"
    home.mkdir()
    supervisor = native.NativeProfileSupervisor.for_profile(home)
    _canonical, _snapshot, root = native._profile_scope(home)
    root.mkdir(parents=True, mode=0o700)
    root.parent.chmod(0o700)
    root.chmod(0o700)
    runtime_path = root / "runtime.json"
    if unsafe_kind == "symlink":
        backing = home / "runtime-backing"
        backing.write_text("{}", encoding="utf-8")
        backing.chmod(0o600)
        runtime_path.symlink_to(backing)
    else:
        runtime_path.write_text("{}", encoding="utf-8")
        runtime_path.chmod(0o644)

    with pytest.raises(native.NativeProfileError) as raised:
        supervisor.cleanup()

    assert raised.value.code == "native_runtime_lease_unsafe"
    assert os.path.lexists(runtime_path)


def test_snapshot_replacement_after_owner_scan_is_retained(monkeypatch, tmp_path):
    import hermes_cli.native_real_profile as native

    home = tmp_path / "profile"
    home.mkdir()
    supervisor = native.NativeProfileSupervisor.for_profile(home)
    _canonical, snapshot, root = native._profile_scope(home)
    snapshot.mkdir(parents=True, mode=0o700)
    root.parent.chmod(0o700)
    root.chmod(0o700)
    snapshot.chmod(0o700)
    original_file = snapshot / "Local State"
    original_file.write_text("original", encoding="utf-8")
    original_file.chmod(0o600)
    replacement = root / "replacement"
    replacement.mkdir(mode=0o700)
    replacement_file = replacement / "Local State"
    replacement_file.write_text("replacement", encoding="utf-8")
    replacement_file.chmod(0o600)
    moved_original = root / "moved-original"
    swapped = False

    def owners(_path):
        nonlocal swapped
        if not swapped:
            swapped = True
            os.replace(snapshot, moved_original)
            os.replace(replacement, snapshot)
        return []

    monkeypatch.setattr(native, "_processes_owning_data_dir", owners)

    with pytest.raises(native.NativeProfileError) as raised:
        supervisor.cleanup(delete_snapshot=True)

    assert raised.value.code == "native_snapshot_identity_changed"
    assert (moved_original / "Local State").read_text(encoding="utf-8") == "original"
    assert (snapshot / "Local State").read_text(encoding="utf-8") == "replacement"


@pytest.mark.parametrize("unsafe_kind", ["symlink", "wrong-mode", "hardlink"])
def test_native_lock_rejects_unsafe_file_identity(tmp_path, unsafe_kind):
    import hermes_cli.native_real_profile as native

    root = tmp_path / unsafe_kind / "browser-supervisor" / "native-real-profile"
    root.mkdir(parents=True, mode=0o700)
    root.parent.chmod(0o700)
    root.chmod(0o700)
    lock_path = root / "native.lock"
    backing = tmp_path / unsafe_kind / "backing"
    backing.write_text("lock", encoding="utf-8")
    backing.chmod(0o600)
    if unsafe_kind == "symlink":
        lock_path.symlink_to(backing)
    elif unsafe_kind == "wrong-mode":
        lock_path.write_text("lock", encoding="utf-8")
        lock_path.chmod(0o644)
    else:
        os.link(backing, lock_path)

    with pytest.raises(native.NativeProfileError) as raised:
        native.NativeProfileLock(str(root)).acquire()

    assert raised.value.code == "native_lock_unsafe"


def test_credential_source_hardlink_is_rejected(tmp_path):
    import hermes_cli.native_real_profile as native

    backing = tmp_path / "backing"
    backing.write_text("credential", encoding="utf-8")
    backing.chmod(0o600)
    linked = tmp_path / "Cookies"
    os.link(backing, linked)

    with pytest.raises(native.NativeProfileError) as raised:
        native.validate_credential_input(str(linked))

    assert raised.value.code == "native_credential_hardlink"


def test_credential_source_inode_swap_during_copy_is_rejected(monkeypatch, tmp_path):
    import hermes_cli.native_real_profile as native

    source = tmp_path / "Cookies"
    source.write_bytes(b"credential")
    source.chmod(0o600)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"credential")
    replacement.chmod(0o600)
    destination = tmp_path / "snapshot" / "Cookies"
    original_copy = native._copy_native_descriptor

    def swapping_copy(source_fd, destination_path):
        original_copy(source_fd, destination_path)
        os.replace(replacement, source)

    monkeypatch.setattr(native, "_copy_native_descriptor", swapping_copy)

    with pytest.raises(native.NativeProfileError) as raised:
        native._copy_native_file(str(source), str(destination))

    assert raised.value.code == "native_credential_changed"


def test_manifest_inode_swap_during_read_is_rejected(monkeypatch, tmp_path):
    import hermes_cli.native_real_profile as native

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir(mode=0o700)
    request = native.NativeProfileRequest("chrome", "Profile 1", "worker@example.test")
    fingerprint = "f" * 64
    manifest = {
        "version": 1,
        "snapshot_uuid": "snapshot-1",
        "source_profile_hash": native.hashlib.sha256(b"Profile 1").hexdigest(),
        "expected_account_hash": native.hashlib.sha256(
            b"worker@example.test"
        ).hexdigest(),
        "executable_fingerprint": fingerprint,
    }
    path = snapshot / native._MANIFEST_NAME
    encoded = json.dumps(manifest).encode("utf-8")
    path.write_bytes(encoded)
    path.chmod(0o600)
    replacement = snapshot / "replacement"
    replacement.write_bytes(encoded)
    replacement.chmod(0o600)
    original_read = native.os.read
    swapped = False

    def swapping_read(fd, size):
        nonlocal swapped
        chunk = original_read(fd, size)
        if not swapped:
            swapped = True
            os.replace(replacement, path)
        return chunk

    monkeypatch.setattr(native.os, "read", swapping_read)

    assert native._manifest_for(request, snapshot, fingerprint) is None
    assert swapped is True


def test_devtools_active_port_hardlink_is_rejected(monkeypatch, tmp_path):
    import hermes_cli.native_real_profile as native

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir(mode=0o700)
    backing = tmp_path / "active-port"
    backing.write_text("43123\n/devtools/browser/abc\n", encoding="utf-8")
    backing.chmod(0o600)
    os.link(backing, snapshot / "DevToolsActivePort")
    process = Mock(poll=Mock(return_value=None))
    monotonic = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(native.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(native.time, "sleep", lambda _delay: None)
    monkeypatch.setattr(native, "_probe_cdp", lambda *_args, **_kwargs: True)

    with pytest.raises(native.NativeProfileError) as raised:
        native._wait_for_native_ready(process, snapshot, timeout=1.0)

    assert raised.value.code == "native_launch_timeout"


def test_held_native_lock_rejects_named_inode_replacement(tmp_path):
    import hermes_cli.native_real_profile as native

    root = tmp_path / "browser-supervisor" / "native-real-profile"
    root.parent.mkdir(parents=True, mode=0o700)
    root.parent.chmod(0o700)
    lock = native.NativeProfileLock(str(root)).acquire()
    replacement = root / "replacement.lock"
    replacement.write_text("", encoding="utf-8")
    replacement.chmod(0o600)
    os.replace(replacement, root / "native.lock")

    try:
        with pytest.raises(native.NativeProfileError) as raised:
            lock.prove()
        assert raised.value.code == "native_lock_unsafe"
    finally:
        lock.release()


def test_lock_acquire_releases_flock_when_final_identity_proof_fails(
    monkeypatch, tmp_path
):
    import hermes_cli.native_real_profile as native

    root = tmp_path / "browser-supervisor" / "native-real-profile"
    root.parent.mkdir(parents=True, mode=0o700)
    root.parent.chmod(0o700)
    lock = native.NativeProfileLock(str(root))
    monkeypatch.setattr(
        lock,
        "prove",
        Mock(
            side_effect=native.NativeProfileError(
                "native_lock_unsafe", "identity changed"
            )
        ),
    )

    with pytest.raises(native.NativeProfileError):
        lock.acquire()

    assert lock._fd is None
    with native.NativeProfileLock(str(root)):
        pass


def test_durable_runtime_rejects_replaced_lock_identity(tmp_path):
    import hermes_cli.native_real_profile as native

    home = tmp_path / "profile"
    home.mkdir()
    canonical_home, snapshot, root = native._profile_scope(home)
    root.parent.mkdir(parents=True, mode=0o700)
    root.parent.chmod(0o700)
    root.mkdir(mode=0o700)
    lock = native.NativeProfileLock(str(root)).acquire()
    lock_device, lock_inode = lock.identity
    runtime = native._NativeRuntime(
        process=SimpleNamespace(pid=4321),
        process_start_time=10.0,
        cdp_url="http://127.0.0.1:43123",
        cdp_port=43123,
        snapshot_uuid="snapshot-1",
        executable_fingerprint="f" * 64,
        lock=lock,
        hermes_home=canonical_home,
        snapshot_path=str(snapshot),
        supervisor_path=str(root),
        source_profile_hash="source-hash",
        expected_account_hash="account-hash",
        lease_path=str(root / "runtime.json"),
        runtime_generation="generation-1",
        lock_device=lock_device,
        lock_inode=lock_inode,
    )
    native._write_runtime_lease(runtime)
    lock.release()
    replacement = root / "replacement.lock"
    replacement.write_text("", encoding="utf-8")
    replacement.chmod(0o600)
    os.replace(replacement, root / "native.lock")

    try:
        with pytest.raises(native.NativeProfileError) as raised:
            native.NativeProfileSupervisor.for_profile(home).cleanup()
        assert raised.value.code == "native_runtime_lease_invalid"
        assert (root / "runtime.json").exists()
    finally:
        (root / "runtime.json").unlink()


def test_durable_runtime_hardlink_is_retained(monkeypatch, tmp_path):
    import psutil

    import hermes_cli.native_real_profile as native

    home = tmp_path / "profile"
    home.mkdir()
    canonical_home, snapshot, root = native._profile_scope(home)
    root.parent.mkdir(parents=True, mode=0o700)
    root.parent.chmod(0o700)
    root.mkdir(mode=0o700)
    lock = native.NativeProfileLock(str(root)).acquire()
    lock_device, lock_inode = lock.identity
    runtime = native._NativeRuntime(
        process=SimpleNamespace(pid=999999),
        process_start_time=10.0,
        cdp_url="http://127.0.0.1:43123",
        cdp_port=43123,
        snapshot_uuid="snapshot-1",
        executable_fingerprint="f" * 64,
        lock=lock,
        hermes_home=canonical_home,
        snapshot_path=str(snapshot),
        supervisor_path=str(root),
        source_profile_hash="source-hash",
        expected_account_hash="account-hash",
        lease_path=str(root / "runtime.json"),
        runtime_generation="generation-1",
        lock_device=lock_device,
        lock_inode=lock_inode,
    )
    native._write_runtime_lease(runtime)
    os.link(root / "runtime.json", root / "runtime-copy")
    lock.release()
    monkeypatch.setattr(
        psutil,
        "Process",
        Mock(side_effect=psutil.NoSuchProcess(999999)),
    )
    monkeypatch.setattr(native, "_processes_owning_data_dir", lambda *_args: [])

    with pytest.raises(native.NativeProfileError) as raised:
        native.NativeProfileSupervisor.for_profile(home).cleanup()

    assert raised.value.code == "native_runtime_lease_unsafe"
    assert (root / "runtime.json").exists()


def test_durable_runtime_growth_during_descriptor_read_is_retained(
    monkeypatch, tmp_path
):
    import psutil

    import hermes_cli.native_real_profile as native

    home = tmp_path / "profile"
    home.mkdir()
    canonical_home, snapshot, root = native._profile_scope(home)
    root.parent.mkdir(parents=True, mode=0o700)
    root.parent.chmod(0o700)
    root.mkdir(mode=0o700)
    lock = native.NativeProfileLock(str(root)).acquire()
    lock_device, lock_inode = lock.identity
    runtime = native._NativeRuntime(
        process=SimpleNamespace(pid=999999),
        process_start_time=10.0,
        cdp_url="http://127.0.0.1:43123",
        cdp_port=43123,
        snapshot_uuid="snapshot-1",
        executable_fingerprint="f" * 64,
        lock=lock,
        hermes_home=canonical_home,
        snapshot_path=str(snapshot),
        supervisor_path=str(root),
        source_profile_hash="source-hash",
        expected_account_hash="account-hash",
        lease_path=str(root / "runtime.json"),
        runtime_generation="generation-1",
        lock_device=lock_device,
        lock_inode=lock_inode,
    )
    native._write_runtime_lease(runtime)
    lock.release()
    original_read = native.os.read
    grew = False

    def growing_read(fd, size):
        nonlocal grew
        chunk = original_read(fd, size)
        if not grew:
            grew = True
            with (root / "runtime.json").open("ab") as handle:
                handle.write(b" ")
        return chunk

    monkeypatch.setattr(native.os, "read", growing_read)
    monkeypatch.setattr(
        psutil,
        "Process",
        Mock(side_effect=psutil.NoSuchProcess(999999)),
    )
    monkeypatch.setattr(native, "_processes_owning_data_dir", lambda *_args: [])

    with pytest.raises(native.NativeProfileError) as raised:
        native.NativeProfileSupervisor.for_profile(home).cleanup()

    assert raised.value.code == "native_runtime_lease_unsafe"
    assert grew is True
    assert (root / "runtime.json").exists()


def test_native_supervisor_store_is_excluded_from_backups():
    from hermes_cli.backup import _EXCLUDED_DIRS

    assert "browser-supervisor" in _EXCLUDED_DIRS


def test_live_pid_is_checked_against_fixed_google_requirement(monkeypatch):
    import hermes_cli.native_real_profile as native

    security_check = Mock(return_value=True)
    monkeypatch.setattr(native, "_security_check_guest_requirement", security_check)

    native._validate_live_process_signature(4321)

    security_check.assert_called_once_with(4321, native.STABLE_CHROME_REQUIREMENT)


def test_live_pid_requirement_failure_is_fail_closed(monkeypatch):
    import hermes_cli.native_real_profile as native

    monkeypatch.setattr(
        native,
        "_security_check_guest_requirement",
        Mock(return_value=False),
    )

    with pytest.raises(native.NativeProfileError) as exc:
        native._validate_live_process_signature(4321)

    assert exc.value.code == "native_live_signature_invalid"


def test_native_consent_revocation_requests_typed_snapshot_cleanup(monkeypatch):
    import hermes_cli.native_real_profile as native
    import tools.browser_use_cli as browser_use

    config = dict(_NATIVE_CONFIG, use_real_profile=False)
    supervisor = Mock()
    supervisor.acquire.side_effect = native.NativeProfileError(
        "native_consent_required", "consent required"
    )
    monkeypatch.setattr(
        native.NativeProfileSupervisor,
        "for_profile",
        classmethod(lambda cls, home=None: supervisor),
    )
    monkeypatch.setattr(browser_use, "_read_browser_cfg", lambda **_kw: config)
    monkeypatch.setattr(browser_use, "_find_cli", lambda: ["browser-use"])
    monkeypatch.setattr(browser_use, "_blocked_url_in_code", lambda _code: None)

    payload = _tool_payload(browser_use.browser_exec("print('x')"))

    assert "consent" in payload["error"]
    supervisor.cleanup.assert_called_once_with(delete_snapshot=True)


def test_acquire_installs_token_before_runtime_resolution(monkeypatch, tmp_path):
    import hermes_cli.native_real_profile as native

    supervisor = native.NativeProfileSupervisor.for_profile(tmp_path / "home")
    home = supervisor.hermes_home

    def resolve(_config, _env, *, hermes_home):
        assert hermes_home == home
        assert len(native._client_leases[home]) == 1
        native._runtimes[home] = SimpleNamespace(
            cdp_url="http://127.0.0.1:43123",
            runtime_generation="generation-a",
        )
        return "http://127.0.0.1:43123"

    monkeypatch.setattr(native, "_resolve_native_profile_cdp", resolve)

    client = supervisor.acquire(dict(_NATIVE_CONFIG), {})

    assert client.runtime_namespace == "generation-a"
    assert client.token in native._client_leases[home]
    native._client_leases.pop(home, None)
    native._runtimes.pop(home, None)


def test_failed_acquire_removes_exact_token_and_restores_cleanup(monkeypatch, tmp_path):
    import hermes_cli.native_real_profile as native

    supervisor = native.NativeProfileSupervisor.for_profile(tmp_path / "home")
    home = supervisor.hermes_home
    runtime = SimpleNamespace(runtime_generation="generation-b")
    native._runtimes[home] = runtime
    scheduled = Mock()
    monkeypatch.setattr(native, "_schedule_native_profile_cleanup", scheduled)
    monkeypatch.setattr(
        native,
        "_resolve_native_profile_cdp",
        Mock(side_effect=native.NativeProfileError("native_busy", "busy")),
    )

    with pytest.raises(native.NativeProfileError):
        supervisor.acquire(dict(_NATIVE_CONFIG), {})

    assert home not in native._client_leases
    scheduled.assert_called_once_with(
        120.0, hermes_home=home, runtime_generation="generation-b"
    )
    native._runtimes.pop(home, None)


def test_release_arms_cleanup_for_the_exact_runtime_generation(monkeypatch, tmp_path):
    import hermes_cli.native_real_profile as native

    supervisor = native.NativeProfileSupervisor.for_profile(tmp_path / "home")
    home = supervisor.hermes_home
    client = native.NativeProfileClient(
        hermes_home=home,
        token="token-c",
        cdp_url="http://127.0.0.1:43123",
        runtime_namespace="generation-c",
        inactivity_delay=17.0,
    )
    native._client_leases[home] = {client.token}
    scheduled = Mock()
    monkeypatch.setattr(native, "_schedule_native_profile_cleanup", scheduled)

    supervisor.release(client)

    assert home not in native._client_leases
    scheduled.assert_called_once_with(
        17.0, hermes_home=home, runtime_generation="generation-c"
    )


@pytest.mark.parametrize(
    "contents", ["browser: [", "not-a-mapping\n", "browser: true\n"]
)
def test_browser_exec_config_read_fails_closed_on_invalid_yaml_root(
    monkeypatch, tmp_path, contents
):
    import hermes_cli.config as config
    import tools.browser_use_cli as browser_use

    config_path = tmp_path / "config.yaml"
    config_path.write_text(contents, encoding="utf-8")
    monkeypatch.setattr(config, "get_config_path", lambda: config_path)

    with pytest.raises(Exception):
        browser_use._read_browser_cfg(fail_closed=True)


@pytest.mark.parametrize("listener_owners", [None, (9876,)])
def test_missing_root_retains_runtime_until_recorded_listener_absence_is_proven(
    monkeypatch, listener_owners
):
    import psutil

    import hermes_cli.native_real_profile as native

    runtime = cast(
        native._NativeRuntime,
        SimpleNamespace(
            process=SimpleNamespace(pid=999999),
            snapshot_path="/tmp/hermes-native-snapshot",
            cdp_port=43123,
        ),
    )
    monkeypatch.setattr(
        psutil,
        "Process",
        Mock(side_effect=psutil.NoSuchProcess(999999)),
    )
    monkeypatch.setattr(native, "_processes_owning_data_dir", lambda *_args: [])
    monkeypatch.setattr(
        native,
        "_query_lsof_listener_owners",
        lambda _port: listener_owners,
    )

    with pytest.raises(native.NativeProfileError) as raised:
        native._close_recorded_runtime(runtime)

    assert raised.value.code == "native_cleanup_uncertain"


def test_cleanup_keeps_runtime_lease_when_missing_root_listener_is_ambiguous(
    monkeypatch, tmp_path
):
    import psutil

    import hermes_cli.native_real_profile as native

    home = tmp_path / "profile"
    home.mkdir()
    canonical_home, snapshot, _root = native._profile_scope(home)
    runtime = cast(
        native._NativeRuntime,
        SimpleNamespace(
            process=SimpleNamespace(pid=999999),
            snapshot_path=str(snapshot),
            cdp_port=43123,
            runtime_generation="generation-listener-ambiguous",
            lock=Mock(),
        ),
    )
    native._runtimes[canonical_home] = runtime
    monkeypatch.setattr(
        psutil,
        "Process",
        Mock(side_effect=psutil.NoSuchProcess(999999)),
    )
    monkeypatch.setattr(native, "_processes_owning_data_dir", lambda *_args: [])
    monkeypatch.setattr(native, "_query_lsof_listener_owners", lambda _port: None)
    remove_runtime_lease = Mock()
    monkeypatch.setattr(native, "_remove_runtime_lease", remove_runtime_lease)

    try:
        with pytest.raises(native.NativeProfileError) as raised:
            native._cleanup_native_profile(hermes_home=home)

        assert raised.value.code == "native_cleanup_uncertain"
        assert native._runtimes[canonical_home] is runtime
        remove_runtime_lease.assert_not_called()
    finally:
        native._runtimes.pop(canonical_home, None)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), 0, -1])
def test_native_startup_timeout_rejects_nonfinite_or_nonpositive_values(value):
    import hermes_cli.native_real_profile as native

    with pytest.raises(native.NativeProfileError) as raised:
        native._native_startup_timeout(value)

    assert raised.value.code == "native_command_timeout_invalid"


def test_native_startup_timeout_has_a_finite_upper_bound():
    import hermes_cli.native_real_profile as native

    assert native._native_startup_timeout(10**100) == 120.0


def test_disabling_native_toggle_cleans_existing_snapshot_before_other_routes(
    monkeypatch, tmp_path
):
    import hermes_constants
    import hermes_cli.native_real_profile as native
    import tools.browser_use_cli as browser_use

    state_root = tmp_path / "browser-supervisor" / "native-real-profile"
    (state_root / "snapshot").mkdir(parents=True)
    supervisor = Mock()
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        native.NativeProfileSupervisor,
        "for_profile",
        classmethod(lambda cls, home=None: supervisor),
    )

    error = browser_use._cleanup_revoked_native_profile(
        dict(_NATIVE_CONFIG, real_profile_macos_native=False)
    )

    assert error is not None
    assert "Native real-profile mode is off" in error
    supervisor.cleanup.assert_called_once_with(delete_snapshot=True)


def test_missing_browser_use_cli_prevents_legacy_real_profile_side_effects(monkeypatch):
    import tools.browser_use_cli as browser_use

    config = dict(
        _NATIVE_CONFIG,
        real_profile_macos_native=False,
        use_real_profile=True,
    )
    resolve = Mock(side_effect=AssertionError("resolver must not run without CLI"))
    monkeypatch.setattr(browser_use, "_read_browser_cfg", lambda **_kw: config)
    monkeypatch.setattr(browser_use, "_find_cli", lambda: None)
    monkeypatch.setattr(browser_use, "_resolve_real_profile_cdp", resolve)
    monkeypatch.setattr(browser_use, "_blocked_url_in_code", lambda _code: None)

    payload = _tool_payload(browser_use.browser_exec("print('x')"))

    assert "browser-use CLI not found" in payload["error"]
    resolve.assert_not_called()
