"""
Shared LLM generation call, used by every orchestrator node (decompose,
direct-answer, synthesize) and by grounding.py's faithfulness check, so the
provider-switching logic lives in exactly one place.

LLM_PROVIDER=ollama runs fully local against Ollama's OpenAI-compatible
endpoint (no API key needed); anything else calls real Claude via the
Anthropic API.

@traceable here is what makes the prompt actually sent, the model, and
token usage show up per-call in LangSmith -- LangGraph's own tracing covers
node-to-node flow automatically, but calls made with the raw OpenAI/
Anthropic SDKs (not LangChain's chat model wrappers) aren't traced unless
explicitly wrapped like this.
"""

import os
import re

from anthropic import Anthropic
from langsmith import get_current_run_tree, traceable
from openai import OpenAI

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")
OLLAMA_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "llama3.1:8b")
# See embed_store.py's OLLAMA_BASE_URL comment -- same reasoning applies here.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")


def extract_json_array(text: str) -> str:
    """Pull the first [...] block out of an LLM response that may wrap its
    JSON in prose or markdown fences despite being asked not to."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    return match.group(0) if match else "[]"


def _record_usage(model: str, usage) -> None:
    run = get_current_run_tree()
    if run is None:
        return
    run.metadata["provider"] = LLM_PROVIDER
    run.metadata["model"] = model
    if usage is not None:
        run.metadata["usage"] = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)


@traceable(name="llm_generate", run_type="llm")
def generate(system_prompt: str, user_content: str, max_tokens: int = 500) -> str:
    if LLM_PROVIDER == "ollama":
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        _record_usage(OLLAMA_MODEL, response.usage)
        return response.choices[0].message.content

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    _record_usage("claude-sonnet-4-6", response.usage)
    return "".join(block.text for block in response.content if block.type == "text")
