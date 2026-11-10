"""Text formats shared by the corpus tools, the arms and the calibrator."""

from __future__ import annotations

from ..ids import canonical_json


def format_read(path: str, start_line: int, end_line: int, body: str) -> str:
    """The text of a corpus_read result (and of a retrieved RAG chunk): a "[path:start-end]"
    header line, then the lines verbatim. The header is also the ref a claim cites."""
    return f"[{path}:{start_line}-{end_line}]\n{body}"


def tool_result_json(text: str, tool_use_id: str = "toolu_calibration") -> str:
    """The canonical JSON of the tool_result block carrying `text`, as a request sends it."""
    block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": [{"type": "text", "text": text}]}
    return canonical_json(block).decode("utf-8")
