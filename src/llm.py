"""
Shared LLM generation call. Pulled out of query.py in Week 4 because the
LangGraph orchestrator needs the same provider-switching generation logic in
multiple nodes (decompose, direct-answer, synthesize), not just the one
place query.py used it before.

LLM_PROVIDER=ollama runs fully local against Ollama's OpenAI-compatible
endpoint (no API key needed). Set back to "anthropic" to use real Claude once
real keys are in place.
"""

import os
import re

from anthropic import Anthropic
from openai import OpenAI

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")
OLLAMA_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "llama3.1:8b")


def extract_json_array(text: str) -> str:
    """Pull the first [...] block out of an LLM response that may wrap its
    JSON in prose or markdown fences despite being asked not to."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    return match.group(0) if match else "[]"


def generate(system_prompt: str, user_content: str, max_tokens: int = 500) -> str:
    if LLM_PROVIDER == "ollama":
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        return response.choices[0].message.content

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    return "".join(block.text for block in response.content if block.type == "text")
