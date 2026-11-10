"""A thin, transport-free MCP-shaped stub over the ctx tools.

Decision: M2 ships this stub instead of a real MCP server.
- M2 has a no-network and minimal-dependency rule; a real MCP server adds a transport, a process
  lifecycle and an `mcp` SDK dependency.
- What MCP buys (any client can call the ctx tools) only matters once the harness is external,
  in M4.
- The stub proves the handler layer is transport-agnostic: it runs the identical `dispatch`
  the Anthropic tool path runs, so both produce the same text for the same input.
- It costs about 40 lines.
"""

from __future__ import annotations

from .tools import CTX_TOOLS, dispatch


def list_tools() -> list[dict]:
    """The ctx tool definitions (name, description, input_schema), as MCP's tools/list returns."""
    return [dict(tool) for tool in CTX_TOOLS]


def call_tool(session, name: str, arguments: dict, *, turn_id: str = "mcp") -> dict:
    """Run a ctx tool like MCP's tools/call: {"content": [{"type": "text", "text": ...}], "isError": bool}."""
    outcome = dispatch(session, name, arguments, turn_id=turn_id)
    return {"content": [{"type": "text", "text": outcome.text}], "isError": outcome.is_error}
