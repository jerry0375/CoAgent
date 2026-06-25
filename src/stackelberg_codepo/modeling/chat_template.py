from __future__ import annotations


def merge_system_into_user(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    system_text = "\n".join(m.get("content", "") for m in messages if m.get("role") == "system").strip()
    merged = [dict(m) for m in messages if m.get("role") != "system"]
    if not system_text:
        return merged
    if merged and merged[0].get("role") == "user":
        merged[0]["content"] = system_text + "\n\n" + merged[0].get("content", "")
    else:
        merged.insert(0, {"role": "user", "content": system_text})
    return merged


def safe_apply_chat_template(tokenizer, messages: list[dict[str, str]], *, add_generation_prompt: bool = True) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    except Exception as exc:
        if "System role not supported" not in str(exc):
            raise
        return tokenizer.apply_chat_template(merge_system_into_user(messages), tokenize=False, add_generation_prompt=add_generation_prompt)
