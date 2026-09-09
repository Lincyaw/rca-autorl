"""Send one distilled conversation at a served checkpoint and check what comes back.

Replays a session from the SFT corpus up to a chosen assistant turn and asks the
served model to produce that turn, then reports whether the turn parsed as a
tool call and, for a `submit_result`, whether it validates against the bound
schema. This is teacher forcing against training data: it checks that a
checkpoint is loadable, aligned with its chat template, and emits the contract,
not that it can diagnose anything.

    ./scripts/serve_sft.sh &
    python -m scripts.probe_sft            # last turn of session 0
    python -m scripts.probe_sft --turn 0   # its first turn instead
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, cast

# The distillation's terminal submission runs to ~2000 tokens of evidence-carrying
# nodes, and a truncated one is indistinguishable from a model that emitted
# nothing: sglang returns finish_reason "length" with an empty tool_calls list.
# 6000 covers the longest observed submission with room for its reasoning.
DEFAULT_MAX_TOKENS = 6000


def load_session(path: Path, index: int) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i == index:
                return cast(dict[str, Any], json.loads(line))
    raise SystemExit(f"{path} has no session at index {index}")


def post(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return cast(dict[str, Any], json.load(response))
    except urllib.error.URLError as error:
        raise SystemExit(f"no server at {url} ({error}) — start ./scripts/serve_sft.sh") from error


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=Path, default=Path("data/sft/rca_sessions.jsonl"))
    parser.add_argument("--session", type=int, default=0, help="line of the sessions file")
    parser.add_argument(
        "--turn",
        type=int,
        default=-1,
        help="which assistant turn to ask for; -1 (default) is the terminal submission",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30111")
    parser.add_argument("--model", default="rca-sft")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args(argv)

    session = load_session(args.sessions, args.session)
    messages = session["messages"]
    # The exporter writes tool schemas bare; the OpenAI dialect wraps them.
    tools = [{"type": "function", "function": tool} for tool in session.get("tools", [])]

    assistant_turns = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
    if not assistant_turns:
        raise SystemExit(f"session {session.get('sample_id')} has no assistant turn")
    cut = assistant_turns[args.turn]

    response = post(
        f"{args.base_url}/v1/chat/completions",
        {
            "model": args.model,
            "messages": messages[:cut],
            "tools": tools,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
        args.timeout,
    )

    choice = response["choices"][0]
    message = choice["message"]
    calls = message.get("tool_calls") or []
    expected = [c["function"]["name"] for c in (messages[cut].get("tool_calls") or [])]

    print(f"case          {session.get('sample_id')} (train data — this is not held out)")
    print(f"context       {cut} messages, {response['usage']['prompt_tokens']} tokens")
    print(f"finish        {choice['finish_reason']} after {response['usage']['completion_tokens']}")
    print(f"teacher called {expected}")
    print(f"model called   {[c['function']['name'] for c in calls]}")

    if choice["finish_reason"] == "length":
        print("\nTRUNCATED — raise --max-tokens; a cut-off tool call parses as no call at all")
        return 1
    if not calls:
        content = message.get("content") or ""
        print(f"\nNO TOOL CALL. content head: {content[:200]!r}")
        if "<tool_call>" in content:
            print("The call is in content as text — serve with --tool-call-parser qwen.")
        return 1

    payload = json.loads(calls[0]["function"]["arguments"])
    print(f"arg keys      {sorted(payload)}")

    if calls[0]["function"]["name"] == "submit_result":
        from fpg import ModelRCAOutput

        submission = ModelRCAOutput.model_validate(payload)
        truth = json.loads(messages[cut]["tool_calls"][0]["function"]["arguments"])
        print(
            f"schema valid  nodes={len(submission.nodes)} edges={len(submission.edges)} "
            f"root_causes={payload.get('root_causes')}"
        )
        print(
            f"teacher       nodes={len(truth['nodes'])} edges={len(truth['edges'])} "
            f"root_causes={truth.get('root_causes')}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
