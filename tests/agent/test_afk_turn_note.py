"""AFK availability reaches the model API-only — never as durable content.

The AFK note is read centrally by ``build_turn_context`` and travels through
the current user message's API-only content path.  Every conversation surface
enters through that prologue, so CLI, API, TUI, and messaging turns all see the
same machine-global state without surface-specific staging.  The API copy
carries the note while stored ``content`` stays clean, so

* no system message is added and the cached prompt prefix is untouched,
* no past message is rewritten,
* a multimodal turn receives an ephemeral API text part without mutating the
  durable list,
* each turn reads current state, so coming back leaves no stale note to replay.

Fake-agent shape mirrors ``tests/agent/test_gateway_turn_sidecar.py``.
"""

from __future__ import annotations

import copy
import types
from unittest.mock import patch

import pytest

from agent import afk
from agent.turn_context import build_turn_context, compose_user_api_content


@pytest.fixture
def hermes_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


class _FakeTodoStore:
    def has_items(self):
        return True


class _FakeGuardrails:
    def reset_for_turn(self):
        pass


class _FakeAgent:
    """Minimal stand-in covering only what the turn prologue touches."""

    def __init__(self):
        self.session_id = "sess-afk"
        self.model = "test/model"
        self.provider = "openrouter"
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_key = "sk-x"
        self.api_mode = "chat_completions"
        self.platform = "slack"
        self.quiet_mode = True
        self.max_iterations = 90
        self.tools = []
        self.valid_tool_names = set()
        self._skip_mcp_refresh = True
        self.compression_enabled = False
        self.context_compressor = types.SimpleNamespace(
            protect_first_n=2, protect_last_n=2
        )
        self._cached_system_prompt = "SYSTEM"
        self._memory_store = None
        self._memory_manager = None
        self._memory_nudge_interval = 0
        self._turns_since_memory = 0
        self._user_turn_count = 0
        self._todo_store = _FakeTodoStore()
        self._tool_guardrails = _FakeGuardrails()
        self._compression_warning = None
        self._interrupt_requested = False
        self._memory_write_origin = "assistant_tool"
        self._stream_context_scrubber = None
        self._stream_think_scrubber = None

    def _ensure_db_session(self):
        pass

    def _restore_primary_runtime(self):
        pass

    def _cleanup_dead_connections(self):
        return False

    def _emit_status(self, _msg):
        pass

    def _replay_compression_warning(self):
        pass

    def _hydrate_todo_store(self, *_a, **_k):
        pass

    def _safe_print(self, *_a, **_k):
        pass

    def _persist_session(self, messages, _history=None):
        pass


def _build(agent, **overrides):
    kwargs = dict(
        agent=agent,
        user_message="what's the status?",
        system_message=None,
        conversation_history=None,
        task_id=None,
        stream_callback=None,
        persist_user_message=None,
        restore_or_build_system_prompt=lambda *a, **k: None,
        install_safe_stdio=lambda: None,
        sanitize_surrogates=lambda s: s,
        summarize_user_message_for_log=lambda s: str(s),
        set_session_context=lambda _sid: None,
        set_current_write_origin=lambda _o: None,
        ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *a, **k: None),
    )
    kwargs.update(overrides)
    return build_turn_context(**kwargs)


@pytest.fixture(autouse=True)
def _stub_runtime_main():
    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None):
        yield


PRIOR_HISTORY = [
    {"role": "user", "content": "earlier question"},
    {"role": "assistant", "content": "earlier answer"},
]

MULTIMODAL = [
    {"type": "text", "text": "look at this"},
    {"type": "image_url", "image_url": {"url": "https://x/img.png"}},
]


class TestAvailableAddsNothing:
    def test_no_sidecar_when_the_operator_is_available(self, hermes_root):
        agent = _FakeAgent()
        with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
            ctx = _build(agent)
        msg = ctx.messages[ctx.current_turn_user_idx]
        assert "api_content" not in msg
        assert msg["content"] == "what's the status?"


class TestAfkNoteDelivery:
    @pytest.mark.parametrize("platform", ["cli", "api", "tui", "slack"])
    def test_every_conversation_surface_gets_the_current_status(
        self, hermes_root, platform
    ):
        afk.engage(reason="school run")
        agent = _FakeAgent()
        agent.platform = platform
        note = afk.turn_context_note()
        assert note

        with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
            ctx = _build(agent)

        msg = ctx.messages[ctx.current_turn_user_idx]
        assert msg["role"] == "user"
        assert msg["api_content"] == "what's the status?\n\n" + note
        assert msg["api_content"] == compose_user_api_content(
            "what's the status?",
            ctx.ext_prefetch_cache,
            ctx.plugin_user_context,
            ctx.afk_user_context,
        )
        # Durable content is untouched — the note is a delivery detail.
        assert msg["content"] == "what's the status?"

    def test_free_text_reason_never_reaches_the_api_copy(self, hermes_root):
        afk.engage(reason="picking up the kids from soccer")
        agent = _FakeAgent()
        with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
            ctx = _build(agent)
        wire = ctx.messages[ctx.current_turn_user_idx]["api_content"]
        assert "soccer" not in wire
        assert "picking up" not in wire

    def test_no_system_message_and_roles_still_alternate(self, hermes_root):
        afk.engage(reason="lunch")
        agent = _FakeAgent()

        with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
            ctx = _build(agent, conversation_history=copy.deepcopy(PRIOR_HISTORY))

        roles = [m.get("role") for m in ctx.messages]
        assert "system" not in roles
        for earlier, later in zip(roles, roles[1:]):
            assert earlier != later, f"role alternation broken: {roles}"

    def test_past_conversation_is_never_rewritten(self, hermes_root):
        afk.engage(reason="lunch")
        agent = _FakeAgent()
        history = copy.deepcopy(PRIOR_HISTORY)

        with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
            ctx = _build(agent, conversation_history=history)

        for original, delivered in zip(PRIOR_HISTORY, ctx.messages):
            assert delivered["role"] == original["role"]
            assert delivered["content"] == original["content"]
            assert "api_content" not in delivered

    def test_cached_system_prompt_is_not_rebuilt(self, hermes_root):
        afk.engage(reason="lunch")
        agent = _FakeAgent()
        before = agent._cached_system_prompt

        with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
            _build(agent)

        assert agent._cached_system_prompt == before

    def test_note_appends_after_plugin_context(self, hermes_root):
        afk.engage()
        agent = _FakeAgent()
        note = afk.turn_context_note()
        with patch(
            "hermes_cli.plugins.invoke_hook",
            return_value=[{"context": "PLUGIN-CTX"}],
        ):
            ctx = _build(agent)
        msg = ctx.messages[ctx.current_turn_user_idx]
        assert msg["api_content"] == "what's the status?\n\nPLUGIN-CTX\n\n" + note


class TestMultimodalStaysDurableClean:
    def test_multimodal_turn_gets_ephemeral_api_note_only(self, hermes_root):
        """An image turn gets current AFK context without growing a permanent
        AFK paragraph in its stored content."""
        afk.engage(reason="lunch")
        agent = _FakeAgent()
        note = afk.turn_context_note()
        content = copy.deepcopy(MULTIMODAL)

        with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
            ctx = _build(agent, user_message=content)

        msg = ctx.messages[ctx.current_turn_user_idx]
        assert msg["content"] == MULTIMODAL, "durable multimodal content mutated"
        assert all(
            note not in str(part.get("text", "")) for part in msg["content"]
        )
        wire = msg["api_content"]
        assert wire is not msg["content"]
        assert wire[:-1] == MULTIMODAL
        assert wire[-1] == {"type": "text", "text": note}
        wire[0]["text"] = "mutated API copy"
        assert msg["content"] == MULTIMODAL

    def test_note_is_explicitly_receipt_time_history(self, hermes_root):
        afk.engage(reason="private schedule")

        note = afk.turn_context_note()

        assert note is not None
        assert "Availability at receipt time" in note
        assert "when this user message was received" in note
        assert "not the operator's current status" in note
        assert "private schedule" not in note


class TestOneShot:
    def test_coming_back_leaves_no_stale_note(self, hermes_root):
        afk.engage(reason="lunch")
        agent = _FakeAgent()
        with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
            _build(agent)

        afk.clear()
        with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
            ctx = _build(agent)
        assert "api_content" not in ctx.messages[ctx.current_turn_user_idx]

    def test_a_cached_agent_reads_afk_fresh_on_every_turn(self, hermes_root):
        afk.engage()
        agent = _FakeAgent()
        with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
            _build(agent)

        with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
            ctx = _build(agent)
        assert afk.turn_context_note() in ctx.messages[ctx.current_turn_user_idx][
            "api_content"
        ]

    def test_afk_note_does_not_disturb_the_shared_gateway_notes_lane(
        self, hermes_root
    ):
        """The auto-reset / voice-channel channel keeps working unchanged."""
        afk.clear()
        agent = _FakeAgent()
        agent._gateway_turn_context_notes = "[System note: session reset]"
        with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
            ctx = _build(agent)
        msg = ctx.messages[ctx.current_turn_user_idx]
        assert msg["api_content"].endswith("[System note: session reset]")
