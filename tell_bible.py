#!/usr/bin/env python3
"""
tell_bible.py — Local Ollama helper for retreat notes and Bible-focused Q&A.

Requires Ollama running locally (default http://127.0.0.1:11434).

Examples:
  python tell_bible.py notes_160526.md --mode analyze
  python tell_bible.py -n notes.md --mode psalms
  python tell_bible.py --mode verses --text "Read Psalm 23 and Revelation 2:5"
  python tell_bible.py --interactive
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


DEFAULT_BASE = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3")


SYSTEM_BIBLE = """You are a careful, respectful assistant for Christian Bible study.
Use mainstream Protestant/Catholic/Orthodox consensus where traditions differ; note disagreements briefly when relevant.
Do not invent Scripture wording—paraphrase or quote approximately unless the user supplies the text.
Be concise unless the user asks for depth."""


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def ollama_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.3,
    json_format: bool = False,
    timeout: int = 600,
) -> str:
    url = f"{base_url.rstrip('/')}/api/chat"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_format:
        payload["format"] = "json"

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {url}. Is `ollama serve` running? ({e.reason})"
        ) from e

    data = json.loads(raw)
    msg = data.get("message") or {}
    content = msg.get("content")
    if not content:
        raise RuntimeError(f"Unexpected Ollama response: {data!r}")
    return content.strip()


def prompt_for_mode(mode: str, user_text: str) -> tuple[str, str, bool]:
    """Return (system, user, use_json_format)."""
    system = SYSTEM_BIBLE
    use_json = False

    if mode == "analyze":
        user = f"""Analyze the following retreat or personal notes from a Christian perspective.

Focus on:
- Themes (repentance, grace, community, Scripture touchpoints, etc.)
- Implicit or explicit Scripture references; name book/chapter/verse when identifiable
- One or two practical reflection questions for the writer

Notes:
---
{user_text}
---
"""
    elif mode == "psalms":
        user = f"""From the text below, list every Psalm that is clearly referenced or strongly alluded to.
If a range is given (e.g. "Psalm 6–8"), expand to individual numbers.
If none, say so briefly.

Text:
---
{user_text}
---
"""
    elif mode == "verses":
        use_json = True
        user = f"""Detect Bible verse references and near-references in this text (e.g. "Rev 2:5", "Revelation Chapter 2:05", "Psalm 6", "John 3:16").

Return JSON with this exact shape:
{{
  "references": [
    {{
      "raw_span": "exact substring from input if possible",
      "book": "canonical English name e.g. Revelation",
      "chapter": <integer or null if only a book is named>,
      "verse_start": <integer or null>,
      "verse_end": <integer or null, same as start if single verse>,
      "digits": "BOOK_CHAPTER_VERSE as a compact string e.g. REV_2_5 or PSA_6_null for psalm-only"
    }}
  ],
  "notes": "brief caveats e.g. ambiguous numbering"
}}

Text:
---
{user_text}
---
"""
    elif mode == "references":
        use_json = True
        user = f"""Same as verse detection but emphasize numeric structure for study tools.

Return JSON:
{{
  "items": [
    {{
      "citation_guess": "normalized like Revelation 2:5",
      "book": "...",
      "chapter": 2,
      "verse": 5,
      "bible_digits": "2:5 under chapter 2 of the book (explain book separately)",
      "psalm_number": <integer or null if not a Psalm>
    }}
  ]
}}

Text:
---
{user_text}
---
"""
    elif mode == "explain":
        user = f"""Explain the main Christian theological ideas connected to this passage or phrase.
If it is not clearly Scriptural, say what biblical themes might relate.

Text:
---
{user_text}
---
"""
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return system, user, use_json


def run_mode(
    mode: str,
    text: str,
    *,
    base_url: str,
    model: str,
    temperature: float,
) -> str:
    system, user, use_json = prompt_for_mode(mode, text)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    out = ollama_chat(
        base_url, model, messages, temperature=temperature, json_format=use_json
    )
    if use_json:
        try:
            parsed = json.loads(out)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            return (
                out + "\n\n(warning: model did not return valid JSON; showing raw text)"
            )
    return out


def interactive_loop(base_url: str, model: str, temperature: float) -> None:
    print(
        "Interactive Bible mode — type a question and press Enter (paste multi-line if your terminal allows).\n"
        "Commands: /exit or /quit — leave; /model NAME — switch model for this session.\n"
    )
    current_model = model

    while True:
        try:
            line = input("bible> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        user_text = line.strip()
        if user_text in {"/exit", "/quit"}:
            break

        if user_text.startswith("/model "):
            current_model = user_text.split(maxsplit=1)[1].strip() or current_model
            print(f"(model → {current_model})")
            continue

        if not user_text:
            continue

        messages = [
            {"role": "system", "content": SYSTEM_BIBLE},
            {
                "role": "user",
                "content": user_text
                + "\n\nGive Scripture-grounded context; cite references as Book Chapter:Verse when you can.",
            },
        ]
        try:
            reply = ollama_chat(
                base_url,
                current_model,
                messages,
                temperature=temperature,
            )
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            continue
        print(reply)
        print()


def detect_notes_path(args: argparse.Namespace) -> str | None:
    return args.notes or args.notes_file


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Analyze notes and Bible text via local Ollama."
    )
    p.add_argument(
        "notes_file",
        nargs="?",
        metavar="FILE",
        help="Notes file (optional; same as -n/--notes)",
    )
    p.add_argument(
        "-n",
        "--notes",
        metavar="FILE",
        help="Path to a markdown or text notes file (overrides positional FILE)",
    )
    p.add_argument(
        "--text",
        metavar="STRING",
        help="Inline text instead of --notes (for quick verse/psalm checks)",
    )
    p.add_argument(
        "--mode",
        choices=("analyze", "psalms", "verses", "references", "explain"),
        default="analyze",
        help="analyze: thematic notes review | psalms: list Psalm numbers | "
        "verses: structured verse detection (JSON) | references: numeric refs | "
        "explain: theology/context for snippet",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="REPL: ask follow-up questions about the Bible (ignores --mode for generation style)",
    )
    p.add_argument(
        "--base-url",
        default=DEFAULT_BASE,
        help=f"Ollama base URL (default {DEFAULT_BASE})",
    )
    p.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"Model name (default {DEFAULT_MODEL}; env OLLAMA_MODEL overrides)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default 0.3)",
    )

    args = p.parse_args(argv)

    if args.interactive:
        interactive_loop(args.base_url, args.model, args.temperature)
        return 0

    path = detect_notes_path(args)
    if args.text and path:
        print("Use either --text or --notes, not both.", file=sys.stderr)
        return 2

    if args.text:
        body = args.text
    elif path:
        if not os.path.isfile(path):
            print(f"Not a file: {path}", file=sys.stderr)
            return 2
        body = read_text(path)
    else:
        p.print_help()
        print(
            "\nProvide --notes FILE, --text '...', or --interactive.",
            file=sys.stderr,
        )
        return 2

    try:
        result = run_mode(
            args.mode,
            body,
            base_url=args.base_url,
            model=args.model,
            temperature=args.temperature,
        )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
