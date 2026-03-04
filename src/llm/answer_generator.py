from __future__ import annotations

from typing import Dict, List, Tuple

from src.config.prompts import SYSTEM_PROMPT, build_qa_prompt
from src.llm.guardrails import enforce_no_hallucination
from src.llm.lmstudio_client import LMStudioClient


def format_context(results: List[Dict]) -> str:
    blocks = []
    for item in results:
        citation = f"[{item['doc_id']}:p{item.get('page', '?')}:{item['chunk_id']}]"
        blocks.append(f"{citation}\n{item['text']}")
    return "\n\n".join(blocks)


def generate_answer(
    *,
    client: LMStudioClient,
    chat_model: str,
    question: str,
    retrieved_chunks: List[Dict],
    conversation_history: List[Dict[str, str]],
) -> Tuple[str, Dict]:
    context = format_context(retrieved_chunks)
    user_prompt = build_qa_prompt(question=question, context_block=context)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_prompt})

    response = client.chat(model=chat_model, messages=messages)
    choice = response["choices"][0]["message"]["content"]
    answer = enforce_no_hallucination(choice)
    usage = response.get("usage", {})
    return answer, usage
