# AgentM Searcher SGLang Serving Notes

This note records the serving pitfalls found while testing the AgentM
`verifier_v2/searcher` scenario with a Qwen3.5 searcher SFT checkpoint.

## Recommended Serving Shape

Use the searcher checkpoint with SGLang's Qwen XML tool-call parser, but do not
enable a reasoning parser when the chat template defaults to no-think mode.

```bash
CKPT=/path/to/searcher/checkpoint

setsid env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  -u ALL_PROXY -u all_proxy \
  CUDA_VISIBLE_DEVICES=6 HF_ENDPOINT=https://hf-mirror.com \
  .venv/bin/python3 -u -m sglang.launch_server \
    --model-path "$CKPT" \
    --host 127.0.0.1 \
    --port 30000 \
    --trust-remote-code \
    --dtype bfloat16 \
    --context-length 16384 \
    --mem-fraction-static 0.38 \
    --max-running-requests 1 \
    --disable-cuda-graph \
    --tool-call-parser qwen3_coder \
    --log-level info \
  > .runs/sglang-searcher-server.log 2>&1 < /dev/null &
```

Notes:

- `setsid` keeps the server alive after the launching shell exits.
- Unset proxy env vars for local OpenAI/SGLang requests; otherwise httpx/requests
  may try to use SOCKS without `socksio`.
- Keep `--tool-call-parser qwen3_coder` for Qwen XML tool calls.
- Do not pass `--reasoning-parser qwen3` after making no-think the default.

## Chat Template Pitfall

The Qwen3.5 template originally rendered the assistant generation prompt as an
open thinking block unless `enable_thinking=false` was explicitly passed:

```jinja
<|im_start|>assistant
<think>
```

The searcher SFT data is closer to the no-think distribution: an empty closed
thinking block followed by XML tool calls. For serving this checkpoint, patch the
checkpoint `chat_template.jinja` so no-think is the default:

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is true %}
        {{- '<think>\n' }}
    {%- else %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
```

With the original open-think prompt, complex searcher prompts often spend the
generation budget on reasoning or emit XML inside a thinking region. AgentM may
then see empty tool calls or no parsed tool calls.

## EOS Consistency

Make the model config agree with the tokenizer:

- `<|im_end|>` is the tokenizer EOS token.
- `<|im_end|>` id is `248046`.
- `<|endoftext|>` id is `248044` and should be used as pad, not EOS.

Patch both config files in the served checkpoint:

```json
// generation_config.json
{
  "eos_token_id": 248046,
  "pad_token_id": 248044
}
```

```json
// config.json
{
  "eos_token_id": 248046,
  "pad_token_id": 248044,
  "text_config": {
    "eos_token_id": 248046,
    "pad_token_id": 248044
  }
}
```

If EOS stays on `248044`, generation may not stop at `<|im_end|>`, which makes
the model more likely to repeat tool calls until `max_tokens`.

## Reasoning Parser Interaction

After no-think is the default, the closing `</think>` is part of the prompt, not
newly generated text. SGLang's `--reasoning-parser qwen3` only sees generated
tokens, so it may classify generated `<tool_call>...</tool_call>` text as
`reasoning_content` instead of handing it to the tool parser.

Symptom:

- OpenAI response has `message.reasoning_content` containing full XML tool calls.
- `message.tool_calls` is `null`.
- `finish_reason` may be `length`.

Fix:

- Remove `--reasoning-parser qwen3`.
- Keep `--tool-call-parser qwen3_coder`.

## Parallel Tool Calls

Parallel tool calls are compatible with this setup. Do not disable them unless a
specific eval harness requires one tool per turn.

Observed behavior:

- With no-think default and no reasoning parser, complex searcher prompts produce
  multiple parsed `query_sql` tool calls with non-empty `sql` arguments.
- If `max_output_tokens` is too low, the model can start another XML tool call at
  the end of the response and get truncated, causing one trailing empty tool call.

Mitigations:

- Increase `max_output_tokens` for searcher rollouts.
- Add prompt/SFT guidance such as "emit at most N SQL queries per turn" if the
  model over-generates.
- Treat an empty trailing tool call after many valid calls as a truncation issue,
  not a tokenizer/parser failure.

## AgentM Environment

For the local `verifier_v2/searcher` scenario, include AgentM and scenario source
roots on `PYTHONPATH`:

```bash
export PYTHONPATH=/home/fay/AgentM:/home/fay/AgentM/src:\
/home/fay/AgentM/contrib/scenarios/rca/src:\
/home/fay/AgentM/contrib/extensions/llmharness/src
export AGENTM_PROJECT_ROOT=/home/fay/AgentM
export AGENTM_HOME=/home/fay/rca-autorl/.runs/agentm-home
```

An example local AgentM model profile:

```toml
default_model = "searcher_sglang"

[models.searcher_sglang]
provider = "openai"
name = "openai"
model = "/path/to/searcher/checkpoint"
base_url = "http://127.0.0.1:30000/v1"
api_key = "EMPTY"
context_window = 16384
max_output_tokens = 8192
```

Run a searcher smoke:

```bash
cd /home/fay/AgentM
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  -u ALL_PROXY -u all_proxy \
  agentm \
    --scenario verifier_v2/searcher \
    --provider openai \
    --model searcher_sglang \
    --cwd "$CASE_DIR" \
    --set duckdb_sql.data_dir="$CASE_DIR" \
    --max-tool-calls 20 \
    --max-turns 2 \
    --prompt "$(cat "$PROMPT_FILE")"
```

## Quick Validation

Validate tokenizer rendering:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
rendered = tok.apply_chat_template(
    [{"role": "user", "content": "hi"}],
    tokenize=False,
    add_generation_prompt=True,
)
assert rendered.endswith("<think>\n\n</think>\n\n")
assert tok.eos_token == "<|im_end|>"
assert tok.eos_token_id == 248046
```

Validate SGLang tool parsing:

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:30000/v1")
model = client.models.list().data[0].id
tool = {
    "type": "function",
    "function": {
        "name": "query_sql",
        "description": "Run DuckDB SQL",
        "parameters": {
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"],
        },
    },
}
resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Call query_sql with SELECT 1 AS x;"}],
    tools=[tool],
    tool_choice={"type": "function", "function": {"name": "query_sql"}},
    temperature=0,
    max_tokens=256,
)
assert resp.choices[0].message.tool_calls
assert "sql" in resp.choices[0].message.tool_calls[0].function.arguments
```
