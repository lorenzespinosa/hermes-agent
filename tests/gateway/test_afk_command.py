"""``/afk`` — the gateway availability command.

Semantics pinned here:

* bare ``/afk`` and ``/afk on [reason]`` engage,
* ``/afk off`` / ``back`` / ``return`` clear,
* ``/afk status`` reports the exact durable state,
* anything else prints usage and mutates NOTHING,
* every reply says out loud that AFK does not widen approvals or authorize
  consequential work,
* the command is dispatchable mid-run (the operator walks away *while* the
  agent is working — that is the whole point),
* engaging never touches the cached session-context system prompt.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent import afk
from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import (
    SessionContext,
    SessionSource,
    build_session_context_prompt,
)
from hermes_cli.commands import (
    COMMANDS,
    GATEWAY_KNOWN_COMMANDS,
    gateway_help_lines,
    resolve_command,
)


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def runner():
    return object.__new__(GatewayRunner)


class _FakeAfkEvent:
    """Just enough MessageEvent surface for the slash handler."""

    def __init__(self, args: str = ""):
        self._args = args
        self.text = f"/afk {args}".strip()

    def get_command(self):
        return "afk"

    def get_command_args(self):
        return self._args


async def _run(runner, args=""):
    return await runner._handle_afk_command(_FakeAfkEvent(args))


def _assert_no_authority_disclaimer(reply: str):
    low = reply.lower()
    assert "approval" in low, reply
    assert "authoriz" in low, reply


# ── registry / discoverability ──────────────────────────────────────────────


class TestRegistry:
    def test_afk_is_a_canonical_gateway_command(self):
        cmd = resolve_command("afk")
        assert cmd is not None and cmd.name == "afk"
        assert "afk" in GATEWAY_KNOWN_COMMANDS

    def test_afk_dispatches_while_the_agent_is_busy(self):
        assert resolve_command("afk").busy_policy == "dispatch"

    def test_afk_is_shared_by_cli_tui_and_gateway(self):
        cmd = resolve_command("afk")
        assert cmd.gateway_only is False
        assert cmd.cli_only is False
        assert "/afk" in COMMANDS

    def test_afk_requires_explicit_gateway_admin_authorization(self):
        assert resolve_command("afk").admin_only is True

    def test_afk_shows_up_in_gateway_help(self):
        assert any(line.startswith("`/afk") for line in gateway_help_lines())

    def test_afk_advertises_its_syntax(self):
        hint = resolve_command("afk").args_hint
        assert "on" in hint and "off" in hint and "status" in hint

    def test_afk_is_reachable_on_every_gateway_surface(self):
        """Slack routes it through `/hermes afk` (and `!afk` in threads);
        Telegram gets it in the bot-command menu."""
        from hermes_cli.commands import slack_subcommand_map, telegram_bot_commands

        assert "afk" in slack_subcommand_map()
        assert any(name == "afk" for name, _desc in telegram_bot_commands())


# ── engaging ────────────────────────────────────────────────────────────────


class TestEngage:
    @pytest.mark.asyncio
    async def test_bare_afk_engages(self, hermes_home, runner):
        reply = await _run(runner)
        assert afk.is_afk() is True
        assert afk.get_state()["reason"] is None
        _assert_no_authority_disclaimer(reply)

    @pytest.mark.asyncio
    async def test_afk_on_engages(self, hermes_home, runner):
        reply = await _run(runner, "on")
        assert afk.is_afk() is True
        assert afk.get_state()["reason"] is None
        _assert_no_authority_disclaimer(reply)

    @pytest.mark.asyncio
    async def test_afk_on_with_reason_stores_the_reason(self, hermes_home, runner):
        reply = await _run(runner, "on picking up the kids")
        assert afk.get_state()["reason"] == "picking up the kids"
        assert "picking up the kids" in reply

    @pytest.mark.asyncio
    async def test_engage_is_case_and_whitespace_tolerant(self, hermes_home, runner):
        await _run(runner, "  ON   lunch  ")
        assert afk.get_state()["reason"] == "lunch"

    @pytest.mark.asyncio
    async def test_repeat_bare_afk_reports_instead_of_resetting(
        self, hermes_home, runner
    ):
        await _run(runner, "on lunch")
        first = afk.get_state()

        reply = await _run(runner)

        assert afk.get_state() == first, "bare /afk clobbered an existing AFK"
        assert "already" in reply.lower()
        _assert_no_authority_disclaimer(reply)

    @pytest.mark.asyncio
    async def test_afk_on_replaces_an_existing_reason(self, hermes_home, runner):
        await _run(runner, "on lunch")
        await _run(runner, "on dentist")
        assert afk.get_state()["reason"] == "dentist"

    @pytest.mark.asyncio
    async def test_reply_reports_only_a_durable_state(
        self, hermes_home, runner, monkeypatch
    ):
        """A write that cannot be read back must not produce a success reply."""
        monkeypatch.setattr(afk, "_atomic_replace_json", lambda *a, **k: None)

        reply = await _run(runner, "on lunch")

        assert afk.is_afk() is False
        low = reply.lower()
        assert "couldn't" in low or "could not" in low or "failed" in low
        assert "nothing changed" in low


class TestDirectCli:
    def _cli(self):
        pytest.importorskip(
            "prompt_toolkit",
            reason="full CLI optional dependencies are not installed",
        )
        from cli import HermesCLI

        cli = HermesCLI.__new__(HermesCLI)
        cli.session_id = "cli-afk-test"
        cli._pending_resume_sessions = None
        cli._console_print = MagicMock()
        return cli

    def test_cli_afk_on_and_off_use_the_shared_command_path(self, hermes_home):
        cli = self._cli()
        with patch("hermes_cli.plugins.fire_pre_command_hook"):
            assert cli.process_command("/afk on lunch") is True
        assert afk.get_state()["reason"] == "lunch"
        assert "AFK recorded" in cli._console_print.call_args.args[0]

        with patch("hermes_cli.plugins.fire_pre_command_hook"):
            assert cli.process_command("/afk off") is True
        assert afk.is_afk() is False
        assert "AFK cleared" in cli._console_print.call_args.args[0]

    def test_cli_afk_keeps_the_closed_grammar(self, hermes_home):
        cli = self._cli()
        with patch("hermes_cli.plugins.fire_pre_command_hook"):
            assert cli.process_command("/afk maybe") is True
        assert afk.is_afk() is False
        assert "Usage:" in cli._console_print.call_args.args[0]


# ── clearing ────────────────────────────────────────────────────────────────


class TestClear:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("verb", ["off", "back", "return"])
    async def test_clearing_verbs(self, hermes_home, runner, verb):
        await _run(runner, "on lunch")
        reply = await _run(runner, verb)
        assert afk.is_afk() is False
        assert afk.get_state() is None
        _assert_no_authority_disclaimer(reply)

    @pytest.mark.asyncio
    async def test_clearing_verbs_are_case_insensitive(self, hermes_home, runner):
        await _run(runner, "on lunch")
        await _run(runner, "OFF")
        assert afk.is_afk() is False

    @pytest.mark.asyncio
    async def test_clearing_when_available_says_so(self, hermes_home, runner):
        reply = await _run(runner, "off")
        assert afk.is_afk() is False
        assert not afk.state_path().exists()
        assert "weren't" in reply.lower() or "not" in reply.lower()

    @pytest.mark.asyncio
    async def test_clearing_recovers_from_a_corrupt_state_file(
        self, hermes_home, runner
    ):
        afk.state_path().write_text("{corrupt", encoding="utf-8")
        await _run(runner, "off")
        assert afk.is_afk() is False


# ── status ──────────────────────────────────────────────────────────────────


class TestStatus:
    @pytest.mark.asyncio
    async def test_status_while_available(self, hermes_home, runner):
        reply = await _run(runner, "status")
        assert afk.get_state() is None, "/afk status mutated state"
        assert "afk" in reply.lower()
        _assert_no_authority_disclaimer(reply)

    @pytest.mark.asyncio
    async def test_status_reports_the_exact_durable_state(self, hermes_home, runner):
        state = afk.engage(reason="school run")
        reply = await _run(runner, "status")
        assert state["engaged_at"] in reply
        assert "school run" in reply
        assert afk.get_state() == state, "/afk status mutated state"

    @pytest.mark.asyncio
    async def test_status_survives_a_corrupt_state_file(self, hermes_home, runner):
        afk.state_path().write_text("{corrupt", encoding="utf-8")
        reply = await _run(runner, "status")
        assert "afk" in reply.lower()
        assert afk.is_afk() is True, "/afk status cleared an unreadable state"

    @pytest.mark.asyncio
    async def test_status_admits_when_it_cannot_read(
        self, hermes_home, runner, monkeypatch
    ):
        """Claiming "available" because the read blew up would be a lie."""

        def _boom():
            raise OSError("HERMES_HOME is unreachable")

        monkeypatch.setattr(afk, "get_state", _boom)
        reply = await _run(runner, "status")
        low = reply.lower()
        assert "unknown" in low
        assert "available" not in low


# ── unknown syntax ──────────────────────────────────────────────────────────


class TestUnknownSyntax:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "args", ["lunch", "maybe on", "onn", "toggle", "off please", "status now"]
    )
    async def test_unknown_syntax_prints_usage_and_does_not_engage(
        self, hermes_home, runner, args
    ):
        reply = await _run(runner, args)
        assert afk.is_afk() is False
        assert not afk.state_path().exists()
        assert "usage" in reply.lower()
        assert "/afk on" in reply

    @pytest.mark.asyncio
    async def test_unknown_syntax_does_not_clear_an_engaged_state(
        self, hermes_home, runner
    ):
        state = afk.engage(reason="lunch")
        reply = await _run(runner, "sorta")
        assert afk.get_state() == state
        assert "usage" in reply.lower()


# ── mid-run dispatch (Guard 2) ──────────────────────────────────────────────


class TestMidRunDispatch:
    @pytest.mark.asyncio
    async def test_afk_engages_while_an_agent_is_running(self, hermes_home, runner):
        cmd_def = resolve_command("afk")
        reply = await runner._dispatch_busy_slash_command(
            _FakeAfkEvent("on stepping out"), cmd_def, "session-key", None
        )
        assert afk.get_state()["reason"] == "stepping out"
        assert "can't run" not in (reply or "").lower()
        _assert_no_authority_disclaimer(reply)

    @pytest.mark.asyncio
    async def test_afk_clears_while_an_agent_is_running(self, hermes_home, runner):
        afk.engage(reason="lunch")
        cmd_def = resolve_command("afk")
        await runner._dispatch_busy_slash_command(
            _FakeAfkEvent("back"), cmd_def, "session-key", None
        )
        assert afk.is_afk() is False


# ── per-turn context injection ──────────────────────────────────────────────


def _session_context():
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C123",
        chat_type="channel",
        user_id="U123",
        user_name="operator",
    )
    return SessionContext(
        source=source, connected_platforms=[Platform.SLACK], home_channels={}
    )


class TestPromptCacheInvariants:
    def test_afk_never_enters_the_session_context_system_prompt(self, hermes_home):
        """The cached prompt prefix must be byte-identical either way —
        otherwise every /afk toggle rebuilds the agent and re-keys the
        provider prompt cache."""
        context = _session_context()
        available = build_session_context_prompt(context)

        afk.engage(reason="school run")
        note = afk.turn_context_note()
        assert note, "precondition: the operator is marked away"
        while_afk = build_session_context_prompt(context)

        assert while_afk == available
        assert note not in while_afk
        assert "school run" not in while_afk


# ── profile isolation, end to end through the command ───────────────────────


class TestOneRecordForEverySession:
    @pytest.mark.asyncio
    async def test_afk_set_in_one_profile_is_visible_in_another(
        self, tmp_path, monkeypatch, runner
    ):
        """Availability is a fact about the human, not about a profile."""
        root = tmp_path / "root"
        root.mkdir()
        work = root / "profiles" / "work"
        work.mkdir(parents=True)

        monkeypatch.setenv("HERMES_HOME", str(root))
        await _run(runner, "on standup")

        monkeypatch.setenv("HERMES_HOME", str(work))
        assert afk.turn_context_note() is not None
        status = await _run(runner, "status")
        assert "afk" in status.lower()
        assert "not afk" not in status.lower()

    @pytest.mark.asyncio
    async def test_a_profile_can_clear_the_shared_afk(
        self, tmp_path, monkeypatch, runner
    ):
        root = tmp_path / "root"
        root.mkdir()
        work = root / "profiles" / "work"
        work.mkdir(parents=True)

        monkeypatch.setenv("HERMES_HOME", str(work))
        await _run(runner, "on standup")

        monkeypatch.setenv("HERMES_HOME", str(root))
        await _run(runner, "off")
        assert afk.is_afk() is False

        monkeypatch.setenv("HERMES_HOME", str(work))
        assert afk.turn_context_note() is None
