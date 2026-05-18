#!/usr/bin/env python3
"""
analyze_file.py — Enrich retreat notes with Scripture via Ollama Cloud.

Reads a notes file, calls Ollama Cloud, and writes an expanded markdown file
under data/ (Psalm references, verse quotes, words of Christ with citations).

Requires OLLAMA_API_KEY in .env at the project root (see .env.example).

Examples:
  python analyze_file.py notes_180526.md --quotables
  python analyze_file.py -i notes_170526.md --quotables --model gpt-oss:120b
  python analyze_file.py notes.md --analyze
  python analyze_file.py notes.md --psalms --stdout
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_ENV_FILE = ROOT / ".env"
DEFAULT_OUTPUT_DIR = ROOT / "data"
DEFAULT_CLOUD_BASE = "https://ollama.com"
DEFAULT_MODEL = "gpt-oss:120b"

SYSTEM_BIBLE = """You are a careful, respectful assistant for Christian Bible study and retreat notes.
Use standard book names and citation format: Book Chapter:Verse (e.g. John 12:35, Psalm 25:4-5).
For words of Christ, cite the Gospel reference (e.g. John 8:12).
Quote Scripture faithfully; if unsure of exact wording, paraphrase briefly and mark [paraphrase].
Do not invent references—only cite passages you are confident exist."""


def load_env_file(path: Path) -> dict[str, str]:
    """Load KEY=VALUE pairs from a .env file (no quotes required)."""
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            out[key] = value
    return out


def apply_env(env_path: Path) -> None:
    """Merge .env into os.environ without overwriting existing vars."""
    for key, value in load_env_file(env_path).items():
        os.environ.setdefault(key, value)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def ollama_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    api_key: str | None = None,
    temperature: float = 0.3,
    json_format: bool = False,
    timeout: int = 900,
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

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach Ollama at {url} ({e.reason})") from e

    data = json.loads(raw)
    msg = data.get("message") or {}
    content = msg.get("content")
    if not content:
        raise RuntimeError(f"Unexpected Ollama response: {data!r}")
    return content.strip()


def strip_markdown_fence(text: str) -> str:
    """Remove optional ```markdown ... ``` wrapper from model output."""
    m = re.match(r"^```(?:markdown|md)?\s*\n(.*)\n```\s*$", text.strip(), re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def output_path_for(
    input_path: Path,
    output_dir: Path,
    *,
    suffix: str,
    extension: str | None = None,
) -> Path:
    ext = extension if extension is not None else input_path.suffix
    stem = input_path.stem
    return output_dir / f"{stem}{suffix}{ext}"


def prompt_quotables(notes: str, source_name: str) -> tuple[str, str]:
    system = SYSTEM_BIBLE
    user = f"""Expand the retreat notes below into a single markdown document.

Rules:
1. Keep every original bullet and heading verbatim (same order and wording).
2. Immediately after each bullet (or paragraph) that touches faith, prayer, Scripture,
   repentance, grace, commandments, or spiritual practice, add a block:

   **Scripture & reflection**
   - Psalm: quote or paraphrase with reference (e.g. Psalm 25:4-5) when a Psalm fits.
   - Verse: one or more Bible verses with full reference (Book Chapter:Verse).
   - Christ: if the note echoes Jesus' teaching, add his words with Gospel reference
     (e.g. John 12:35).
3. Use proper reference codes on every quote (Book C:V or C:V-V).
4. If a note already cites a passage (e.g. "John 12:34"), supply the standard text
   for that reference and related cross-references.
5. Skip enrichment only for empty bullets or pure section headers.
6. Add a short title line at the top: `# Enriched notes — {source_name}`
   and a line: `> Generated with analyze_file.py --quotables`
7. Return markdown only—no preamble or explanation outside the document.

Source file: {source_name}

---
{notes}
---
"""
    return system, user


def prompt_analyze(notes: str) -> tuple[str, str]:
    user = f"""Analyze the following retreat notes from a Christian perspective.

Focus on themes, implicit Scripture, and two practical reflection questions.

Notes:
---
{notes}
---
"""
    return SYSTEM_BIBLE, user


def prompt_psalms(notes: str) -> tuple[str, str]:
    user = f"""List every Psalm clearly referenced or strongly alluded to in this text.
Expand ranges to individual numbers. If none, say so briefly.

Text:
---
{notes}
---
"""
    return SYSTEM_BIBLE, user


def run_generation(
    mode: str,
    notes: str,
    source_name: str,
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    temperature: float,
) -> str:
    use_json = False
    if mode == "quotables":
        system, user = prompt_quotables(notes, source_name)
    elif mode == "analyze":
        system, user = prompt_analyze(notes)
    elif mode == "psalms":
        system, user = prompt_psalms(notes)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    out = ollama_chat(
        base_url,
        model,
        messages,
        api_key=api_key,
        temperature=temperature,
        json_format=use_json,
    )
    if mode == "quotables":
        return strip_markdown_fence(out)
    return out


def resolve_api_key(explicit: str | None) -> str:
    key = explicit or os.environ.get("OLLAMA_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OLLAMA_API_KEY is not set. Add it to .env in the project root "
            f"({DEFAULT_ENV_FILE}) or export it in your shell."
        )
    return key


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze and enrich retreat notes using Ollama Cloud.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "input_file",
        nargs="?",
        metavar="FILE",
        help="Notes markdown or text file",
    )
    p.add_argument(
        "-i",
        "--input",
        dest="input_opt",
        metavar="FILE",
        help="Input notes file (same as positional FILE)",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--quotables",
        action="store_true",
        help="Enrich notes with Psalms, verses, and sayings of Christ (writes to data/)",
    )
    mode.add_argument(
        "--analyze",
        action="store_true",
        help="Thematic Christian analysis (stdout or --output)",
    )
    mode.add_argument(
        "--psalms",
        action="store_true",
        help="List Psalm references in the notes",
    )
    p.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Output path (default: data/<stem>_enriched<ext> for --quotables)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated files (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--suffix",
        default="_enriched",
        help="Filename suffix before extension (default: _enriched)",
    )
    p.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help=f"Path to .env with OLLAMA_API_KEY (default: {DEFAULT_ENV_FILE})",
    )
    p.add_argument(
        "--base-url",
        default=os.environ.get("OLLAMA_HOST", DEFAULT_CLOUD_BASE),
        help=f"Ollama API base URL (default: cloud {DEFAULT_CLOUD_BASE})",
    )
    p.add_argument(
        "--model",
        "-m",
        default=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL),
        help=f"Model name (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--api-key",
        help="Ollama API key (overrides OLLAMA_API_KEY from .env)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.25,
        help="Sampling temperature (default: 0.25)",
    )
    p.add_argument(
        "--stdout",
        action="store_true",
        help="Print result to stdout instead of writing a file",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it exists",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be sent/written without calling the API",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)

    apply_env(args.env_file)

    input_path = Path(args.input_opt or args.input_file or "")
    if not input_path:
        p.print_help()
        print("\nProvide an input FILE path.", file=sys.stderr)
        return 2

    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        print(f"Not a file: {input_path}", file=sys.stderr)
        return 2

    if args.quotables:
        mode = "quotables"
    elif args.analyze:
        mode = "analyze"
    elif args.psalms:
        mode = "psalms"
    else:
        mode = "quotables"

    notes = read_text(input_path)
    source_name = input_path.name

    if args.dry_run:
        print(f"mode={mode} input={input_path} model={args.model} base={args.base_url}")
        if mode == "quotables" and not args.stdout:
            out_default = output_path_for(
                input_path, args.output_dir.resolve(), suffix=args.suffix
            )
            print(f"would write: {out_default}")
        print(f"notes length: {len(notes)} chars, {len(notes.splitlines())} lines")
        return 0

    try:
        api_key = resolve_api_key(args.api_key)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    try:
        result = run_generation(
            mode,
            notes,
            source_name,
            base_url=args.base_url.rstrip("/"),
            model=args.model,
            api_key=api_key,
            temperature=args.temperature,
        )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    if args.stdout:
        print(result)
        return 0

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    elif mode == "quotables":
        out_path = output_path_for(
            input_path,
            args.output_dir.resolve(),
            suffix=args.suffix,
        )
    else:
        out_path = args.output_dir.resolve() / f"{input_path.stem}_{mode}.txt"

    if out_path.exists() and not args.force:
        print(
            f"Output exists: {out_path} (use --force to overwrite)",
            file=sys.stderr,
        )
        return 2

    header = (
        f"<!-- source: {source_name} | mode: {mode} | "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ')} -->\n\n"
    )
    write_text(out_path, header + result + "\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
