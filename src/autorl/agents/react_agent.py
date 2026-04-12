from __future__ import annotations

import datetime
import json
import time

import json5
from pydantic import BaseModel
from qwen_agent.agents.fncall_agent import FnCallAgent
from transformers import PreTrainedTokenizer

from areal.experimental.openai import ArealOpenAI

try:
    from areal.utils import logging
except Exception:  # pragma: no cover
    import logging  # type: ignore

from autorl.rewards import get_reward_fn
from autorl.tool_env.client import ToolEnvClient

from .prompt import SYSTEM_PROMPT

logger = logging.getLogger("Tongyi-DeepResearch react agent")


def today_date() -> str:
    return datetime.date.today().strftime("%Y-%m-%d")


def parse_judge_result(raw_response: str) -> float:
    import ast

    parsed = None
    for parse_fn in (json.loads, ast.literal_eval):
        try:
            parsed = parse_fn(raw_response.split("```json")[-1].split("```")[0].strip())
            break
        except Exception:
            logger.warning("Error parsing judge result with %s.", parse_fn)

    if parsed is None and '"judgement": "incorrect"' in raw_response:
        parsed = {"judgement": "incorrect"}
    if parsed is None and '"judgement": "correct"' in raw_response:
        parsed = {"judgement": "correct"}
    if parsed is None:
        logger.warning("Unknown judge result. Raw response: %s", raw_response)
        parsed = {"judgement": "unknown"}

    return float(parsed.get("judgement") == "correct")


class MultiTurnReactAgent(FnCallAgent):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_tokens_per_turn: int = 10000,
        max_llm_calls_per_run: int = 100,
        max_total_tokens: int = 32768,
        judge_client: ArealOpenAI | None = None,
        tool_client: ToolEnvClient | None = None,
        tool_env_base_url: str | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_llm_calls_per_run = max_llm_calls_per_run
        self.max_total_tokens = max_total_tokens
        self.max_total_tokens_before_finishing = int(max_total_tokens * 0.8)
        self.judge_client = judge_client
        self.tool_client = tool_client or ToolEnvClient(base_url=tool_env_base_url)

    def count_tokens(self, messages):
        message_strs = []
        for msg in messages:
            if isinstance(msg, BaseModel):
                msg = msg.model_dump()
                assert "role" in msg and "content" in msg
            message_strs.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
        message_strs.append("<|im_start|>assistant\n")
        prompt_token_ids = self.tokenizer.encode("".join(message_strs))
        return len(prompt_token_ids)

    async def call_server(
        self, client: ArealOpenAI, messages: list[dict], max_attempts: int = 100
    ):
        attempts = 0
        while attempts < max_attempts:
            try:
                completion = await client.chat.completions.create(
                    messages=messages,
                    temperature=1.0,
                    stop=["\n<tool_response>", "<tool_response>"],
                    max_completion_tokens=self.max_tokens_per_turn,
                )
                message = completion.choices[0].message
                assert message, "Error: LLM response is empty."
                return completion, message
            except RuntimeError as err:
                attempts += 1
                logger.warning(
                    "RuntimeError during call_server attempt %s/%s: %s",
                    attempts,
                    max_attempts,
                    err,
                )
        raise RuntimeError(f"Failed to get response from LLM after {max_attempts} attempts.")

    async def run_agent(
        self,
        data,
        client: ArealOpenAI,
        save_path: str | None = None,
    ):
        start_time = time.time()
        question = data["question"]
        answer = data["answer"]

        system_prompt = SYSTEM_PROMPT + today_date()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        stats = {"turns": 0, "num_search": 0, "num_access": 0}
        num_llm_calls_available = self.max_llm_calls_per_run
        completions = []
        content = ""

        while num_llm_calls_available > 0:
            if time.time() - start_time > 150 * 60:
                return {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": "No answer found after 2h30mins",
                    "termination": "No answer found after 2h30mins",
                    "completions": completions,
                    "stats": stats,
                }

            stats["turns"] += 1
            num_llm_calls_available -= 1
            completion, message = await self.call_server(client, messages)
            content = message.content or ""
            completions.append(completion)
            messages.append(message)

            if "<tool_call>" in content and "</tool_call>" in content:
                tool_call = content.split("<tool_call>")[1].split("</tool_call>")[0]
                try:
                    tool_call = json5.loads(tool_call)
                    tool_name = tool_call["name"]
                    tool_args = tool_call.get("arguments", {})
                    result = await self.custom_call_tool(tool_name, tool_args)
                    if tool_name == "search":
                        stats["num_search"] += 1
                    elif tool_name == "visit":
                        stats["num_access"] += 1
                except Exception as err:
                    result = (
                        f"Error: {err} Tool call must be a valid json containing "
                        '"name" and "arguments".'
                    )
                messages.append({"role": "user", "content": f"<tool_response>\n{result}\n</tool_response>"})

            if "<answer>" in content and "</answer>" in content:
                break

            if num_llm_calls_available <= 0 and "<answer>" not in content:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Sorry, llm calls exceed the limit. Stop tool calls and provide final "
                            "answer in format <think>...</think>\\n<answer>...</answer>."
                        ),
                    }
                )

            token_count = self.count_tokens(messages)
            if token_count > self.max_total_tokens_before_finishing:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You reached max context length. Stop tool calls and provide final answer "
                            "in format <think>...</think>\\n<answer>...</answer>."
                        ),
                    }
                )
                completion, message = await self.call_server(client, messages)
                completions.append(completion)
                content = message.content or ""
                messages.append(message)
                break

        if "<answer>" in content and "</answer>" in content:
            prediction = content.split("<answer>")[1].split("</answer>")[0]
            termination = "answer"
        else:
            prediction = content or "No answer found."
            termination = (
                "exceed available llm calls"
                if num_llm_calls_available == 0
                else "answer not found"
            )

        result = {
            "question": question,
            "answer": answer,
            "messages": [m.model_dump() if isinstance(m, BaseModel) else m for m in messages],
            "prediction": prediction,
            "termination": termination,
            "completions": completions,
            "stats": stats,
        }
        if save_path:
            to_dump = dict(result)
            to_dump.pop("completions", None)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(to_dump, f, ensure_ascii=False, indent=2)
        return result

    async def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        tool_output = await self.tool_client.execute(tool_name=tool_name, arguments=tool_args)
        if not tool_output.get("ok"):
            return f"Error: {tool_output.get('error', 'tool execution failed')}"
        return str(tool_output.get("result", ""))

    async def calc_reward_with_llm_judge(self, result: dict[str, str]) -> float:
        fallback_reward = get_reward_fn("agent_default")
        if self.judge_client is None:
            return float(
                fallback_reward(
                    completions=str(result.get("prediction", "")),
                    answer=result.get("answer"),
                )
            )

        judge_prompt = (
            "You are an evaluation assistant. Determine whether pred_answer is equivalent "
            "to ground truth answer. Output JSON with rationale and judgement (correct/incorrect).\n\n"
            f"question: {result.get('question')}\n"
            f"ground truth answers: {result.get('answer')}\n"
            f"pred_answer: {str(result.get('prediction', ''))[:200]}\n\n"
            "Your output:"
        )
        try:
            judge_completion = await self.judge_client.chat.completions.create(
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=1.0,
                max_completion_tokens=8192,
                store=False,
            )
            judge_response = judge_completion.choices[0].message.content or ""
            return parse_judge_result(judge_response)
        except Exception as err:
            logger.warning("Error in calling LLM judge: %s", err)
            return float(
                fallback_reward(
                    completions=str(result.get("prediction", "")),
                    answer=result.get("answer"),
                )
            )

    async def make_trajectory(self, data: dict[str, str], client: ArealOpenAI) -> dict:
        result = await self.run_agent(data, client)
        reward = await self.calc_reward_with_llm_judge(result)
        completions = result["completions"]
        if completions:
            client.set_reward(completions[-1].id, reward)
        return result.get("stats", {})
