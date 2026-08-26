"""``!afk`` works in Slack threads through the existing bang-prefix rewrite.

Slack refuses native slash commands inside threads, so the Slack adapter
rewrites a leading ``!cmd`` to ``/cmd`` whenever the first token resolves to a
known gateway command. Registering ``afk`` in the central command registry is
therefore the entire integration — these tests prove that end of the wire
actually holds, and that casual ``!word`` chatter still reaches the agent
untouched.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Mock slack-bolt / slack-sdk if they are not installed (same shape as
# tests/gateway/test_slack_mention.py).
# ---------------------------------------------------------------------------

def _ensure_slack_mock():
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return

    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock

    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock

    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        (
            "slack_bolt.adapter.socket_mode.async_handler",
            slack_bolt.adapter.socket_mode.async_handler,
        ),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)

    sys.modules.setdefault("aiohttp", MagicMock())


_ensure_slack_mock()

from plugins.platforms.slack.adapter import (  # noqa: E402
    _rewrite_known_bang_command,
)


@pytest.mark.parametrize(
    "typed,expected",
    [
        ("!afk", "/afk"),
        ("!afk on", "/afk on"),
        ("!afk on picking up the kids", "/afk on picking up the kids"),
        ("!afk off", "/afk off"),
        ("!afk status", "/afk status"),
        ("!AFK", "/AFK"),
    ],
)
def test_bang_afk_is_rewritten_to_the_slash_form(typed, expected):
    assert _rewrite_known_bang_command(typed) == expected


def test_bang_afk_behind_a_mention_is_rewritten():
    """``@Hermes !afk`` reaches the rewrite with the mention already stripped."""
    assert _rewrite_known_bang_command("!afk on lunch".strip()) == "/afk on lunch"


@pytest.mark.parametrize(
    "chatter",
    [
        "!afkeyboard",
        "!nice work",
        "!afk-ish",
        "!brb",
        "!!afk",
    ],
)
def test_unknown_bang_words_pass_through_untouched(chatter):
    assert _rewrite_known_bang_command(chatter) == chatter


def test_plain_text_mentioning_afk_is_untouched():
    assert _rewrite_known_bang_command("going afk for a bit") == "going afk for a bit"
