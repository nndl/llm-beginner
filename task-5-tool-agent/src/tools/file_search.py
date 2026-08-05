"""Search filenames and text under the task workspace only."""
from pathlib import Path

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "file_search",
        "description": "Find filenames or content matching a literal pattern within a local directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Filename glob or literal content"},
                "dir": {"type": "string", "description": "Directory inside this task workspace"},
            },
            "required": ["pattern", "dir"],
            "additionalProperties": False,
        },
    },
}

_ROOT = Path(__file__).resolve().parents[2]


def run(args: dict) -> str:
    pattern, directory = args.get("pattern"), args.get("dir")
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("'pattern' must be a non-empty string")
    if not isinstance(directory, str):
        raise ValueError("'dir' must be a string")
    target = Path(directory).resolve()
    if target != _ROOT and _ROOT not in target.parents:
        raise ValueError("Directory must be inside the task workspace")
    if not target.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    matches = []
    for path in sorted(target.rglob("*")):
        if not path.is_file() or path.stat().st_size > 200_000:
            continue
        relative = path.relative_to(_ROOT)
        text = path.read_text(encoding="utf-8", errors="replace")
        filename_match = path.match(pattern) or pattern.lower() in path.name.lower()
        content_match = pattern.lower() in text.lower()
        if filename_match or content_match:
            snippet = text[:500].replace("\x00", " ").strip()
            matches.append(f"{relative}\n{snippet}")
    return "\n\n".join(matches) if matches else "No matches found."