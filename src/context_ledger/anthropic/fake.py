"""FakeAnthropicServer: a deterministic stand-in for the Messages API, for tests and the demo.

It is a pure function of its scripted responses, the request bodies it receives and its own
cache state. There is no clock: the cache TTL is counted in calls (`ttl_calls`).

Prompt-cache simulation. The request is linearized in cache order (tools, system, messages);
the prefix up to each `cache_control` breakpoint is hashed (markers stripped, so a breakpoint
moving does not change a prefix). cache_read_input_tokens is the size of the longest
breakpoint prefix stored before and still live; cache_creation_input_tokens is the rest up to
the last breakpoint, which is then stored; input_tokens is everything after the last
breakpoint. Prefixes shorter than the model's min_cache_tokens are not cached. Tokens are
approx_tokens of each block's canonical JSON: a simulator, not Anthropic's tokenizer.

No network: `as_sync_transport` and `as_async_transport` return httpx2.MockTransport handlers
for the real SDK; `as_fake_client` returns a plain object with `.messages.create(**kw)` and
`.messages.count_tokens(**kw)`.

M3 additions, all off unless asked for, so M2 runs are unchanged:
- `POST /v1/messages/count_tokens` (and `count_tokens` on the fake client) answers
  {"input_tokens": n}, deterministically: the simulated prompt tokens plus COUNT_OVERHEAD.
- `emulate=True` mirrors the 2026-10-31 model table: the thinking type must be one the model
  accepts (Sonnet 5.5 rejects "disabled", Haiku 5.5 accepts it), else a 400; with tools present
  the 286-token tool-use system prompt is billed as input; Sonnet's between_tools mode returns
  a thinking block before each tool use; a thinking block this server never issued is refused
  (an invalid signature); usage reports output_tokens_details.thinking_tokens.
- A scripted response {"__error__": status, "error": {...}} is answered with that HTTP error;
  a script that raises FakeTimeout simulates a timeout.
- `responder`, if given, answers every request after the script runs out; `on_request(path,
  body)` is called before each request is handled.

BUILD-V3FIX, always on: a request (or count_tokens body) with an empty or whitespace-only text
block is answered, as the API answers it, with a 400 invalid_request_error in the API's words
(request.empty_text_problems: "messages.6.content.0.text: cache_control cannot be set for empty
text blocks" for one with a breakpoint, "messages: text content blocks must be non-empty"
otherwise); it uses no scripted response. Other invalid requests are harness bugs and fail an
assertion, as before.

BUILD-V4FIX, always on: thinking blocks are bound to the conversation they were created in, as
the API has bound them since 2026-11-02 (before, in the v3 Sonnet grid of 2026-11-01, it took
them back after rewritten content). Every thinking or redacted_thinking block a response carries
(scripted or emulated) is recorded with the request's messages before it (cache_control
stripped). A request (or count_tokens body) that sends such a block back altered, or after
content that differs from what preceded it when it was created, is answered with the API's 400
in its words, e.g. "messages.3.content.0: Invalid `signature` in `thinking` block. The block is
bound to a different conversation. Remove the block, or set ... Content before this block was
rewritten since it was created, first at `messages.2.content.3`." (positions are this request's).
A block left out is accepted: the API's own 400 says to remove it. This replaces M3's rule that
a thinking block issued with a tool use must come back with it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from types import SimpleNamespace

from ..ids import canonical_json, sha256_hex
from ..pricing import PricingTable
from ..window import approx_tokens
from .request import THINKING_TYPES, block_tokens, empty_text_problems, linearize, request_problems, strip_cache_control

Script = Callable[[dict], dict]
COUNT_OVERHEAD = 3
TOOL_SYSTEM_PROMPT_TOKENS = 286
ACCEPTED_THINKING = {"claude-haiku-5-5": {"disabled", "adaptive"}, "claude-sonnet-5-5": {"between_tools", "adaptive"}}


class FakeTimeout(Exception):
    """Raised by a script to simulate a request that timed out (no response at all)."""


class FakeAPIStatusError(Exception):
    """What the fake client raises for an HTTP error, shaped like the SDK's APIStatusError."""

    def __init__(self, status_code: int, body: dict) -> None:
        error = body.get("error") or {}
        super().__init__(f"Error code: {status_code} - {body}")
        self.status_code, self.body = status_code, body
        self.type = error.get("type")
        self.request_id = f"req_fake_{status_code}"


class _HTTPError(Exception):
    def __init__(self, status: int, body: dict) -> None:
        super().__init__(status)
        self.status, self.body = status, body


def _error(status: int, kind: str, message: str) -> _HTTPError:
    return _HTTPError(status, {"type": "error", "error": {"type": kind, "message": message}})


def _refuse_empty_text(body: dict) -> None:
    """The API's 400 for an empty text block (the first one it finds), before anything else."""
    refused = empty_text_problems(body)
    if refused:
        raise _error(400, "invalid_request_error", refused[0])


BINDING_HELP = ('Remove the block, or set `thinking.block_binding.prefix_mismatch_behavior` to "drop_block". That setting requires the '
                "`thinking-binding-controls-2026-08-01` value in the `anthropic-beta` header.")  # fmt: skip


def _thinking_id(block: dict) -> str:
    """What identifies a thinking block: its signature (a redacted one: its data)."""
    return str(block.get("signature") if block["type"] == "thinking" else block.get("data"))


def _prefix(messages: list[dict], m: int, b: int) -> list[tuple[int, int, str, bytes]]:
    """Every message block before messages[m].content[b], as (message, block, role, canonical
    JSON without cache_control)."""
    out = []
    for n, message in enumerate(messages[: m + 1]):
        for k, block in enumerate(message["content"] if n < m else message["content"][:b]):
            out.append((n, k, message["role"], canonical_json(strip_cache_control(block))))
    return out


class FakeAnthropicServer:
    def __init__(
        self,
        script: Sequence[Script] = (),
        *,
        pricing: PricingTable,
        ttl_calls: int = 1000,
        responder: Script | None = None,
        emulate: bool = False,
        on_request: Callable[[str, dict], None] | None = None,
    ) -> None:
        self.script = list(script)
        self.pricing = pricing
        self.ttl_calls = ttl_calls
        self.responder, self.emulate, self.on_request = responder, emulate, on_request
        self.requests: list[dict] = []
        self.counted: list[dict] = []
        self.errors: list[int] = []
        self.authorized = 0  # requests that carried an Authorization header (its value is never kept)
        self._cache: dict[str, int] = {}  # prefix hash -> last call index at which it is live
        self._issued: dict[str, list[tuple[bytes, list]]] = {}  # thinking id -> (block, the messages before it), per issue

    def handle(self, body: dict) -> dict:
        """Answer one Messages request with the next scripted response, usage simulated."""
        if self.on_request is not None:
            self.on_request("/v1/messages", body)
        _refuse_empty_text(body)
        self._check_thinking(body)
        problems = request_problems(body)
        assert not problems, f"invalid request: {problems}"
        index = len(self.requests)
        assert index < len(self.script) or self.responder is not None, "the scripted conversation is exhausted"
        if self.emulate:
            self._check_model(body)
        self.requests.append(json.loads(json.dumps(body)))
        script = self.script[index] if index < len(self.script) else self.responder
        response = dict(script(body))
        if "__error__" in response:
            raise _HTTPError(response["__error__"], {"type": "error", "error": response.get("error") or {"type": "api_error", "message": "scripted"}})
        content = response.setdefault("content", [])
        if self.emulate:
            content = response["content"] = self._add_thinking(body, content, index)
        self._bind(body, content)
        usage = self._cache_usage(body, index)
        usage["output_tokens"] = approx_tokens(canonical_json(content).decode("utf-8"))
        if self.emulate:
            if body.get("tools"):
                usage["input_tokens"] += TOOL_SYSTEM_PROMPT_TOKENS
            thinking = [b for b in content if b.get("type") in ("thinking", "redacted_thinking")]
            usage["output_tokens_details"] = {"thinking_tokens": approx_tokens(canonical_json(thinking).decode("utf-8")) if thinking else 0}
        response.setdefault("usage", usage)
        response.setdefault("id", f"msg_fake_{index:04d}")
        response.setdefault("type", "message")
        response.setdefault("role", "assistant")
        response.setdefault("model", body["model"])
        response.setdefault("stop_reason", "end_turn")
        response.setdefault("stop_sequence", None)
        return response

    def count_tokens(self, body: dict) -> dict:
        """The free count_tokens endpoint: simulated prompt tokens (every block, plus the tool
        system prompt when emulating) plus COUNT_OVERHEAD. No caching logic, as documented."""
        if self.on_request is not None:
            self.on_request("/v1/messages/count_tokens", body)
        assert "max_tokens" not in body, "count_tokens takes the request body without max_tokens"
        _refuse_empty_text(body)
        self._check_thinking(body)
        if self.emulate:
            self._check_model(body)
        self.counted.append(json.loads(json.dumps(body)))
        total = sum(block_tokens(block) for _, block in linearize(body))
        if self.emulate and body.get("tools"):
            total += TOOL_SYSTEM_PROMPT_TOKENS
        return {"input_tokens": total + COUNT_OVERHEAD}

    def _check_model(self, body: dict) -> None:
        model = body.get("model")
        if model not in ACCEPTED_THINKING:
            raise _error(404, "not_found_error", f"model: {model}")
        thinking = body.get("thinking")
        if thinking is not None and thinking.get("type") not in ACCEPTED_THINKING[model]:
            raise _error(400, "invalid_request_error", f"thinking.type {thinking.get('type')!r} is not supported by {model}")

    def _bind(self, body: dict, content: list) -> None:
        """Record each thinking block of a response with the messages before it, as the next
        request renders them: in a new assistant message, or after the request's last message if
        that one is the assistant's (the response continues it)."""
        messages = list(body.get("messages", []))
        if messages and messages[-1]["role"] == "assistant":
            offset = len(messages[-1]["content"])
            messages[-1] = {"role": "assistant", "content": [*messages[-1]["content"], *content]}
        else:
            offset = 0
            messages.append({"role": "assistant", "content": list(content)})
        for i, block in enumerate(content):
            if block.get("type") in THINKING_TYPES:
                issued = (canonical_json(strip_cache_control(block)), _prefix(messages, len(messages) - 1, offset + i))
                self._issued.setdefault(_thinking_id(block), []).append(issued)

    def _check_thinking(self, body: dict) -> None:
        """The API's 400 for the first thinking block sent back altered, never issued (when
        emulating), or after content rewritten since it was created (module docstring)."""
        messages = body.get("messages", [])
        for m, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            for b, block in enumerate(message["content"]):
                if block.get("type") not in THINKING_TYPES:
                    continue
                field = "signature" if block["type"] == "thinking" else "data"
                at = f"messages.{m}.content.{b}: Invalid `{field}` in `{block['type']}` block."
                issued = [p for blob, p in self._issued.get(_thinking_id(block), []) if blob == canonical_json(strip_cache_control(block))]
                if not issued:
                    if self._issued.get(_thinking_id(block)) or self.emulate:
                        raise _error(400, "invalid_request_error", at)
                    continue
                here = _prefix(messages, m, b)
                if here in issued:
                    continue
                created = issued[-1]
                first = next((j for j in range(min(len(here), len(created))) if here[j] != created[j]), min(len(here), len(created)))
                where = f"messages.{here[first][0]}.content.{here[first][1]}" if first < len(here) else f"messages.{m}.content.{b}"
                raise _error(400, "invalid_request_error", f"{at} The block is bound to a different conversation. {BINDING_HELP} "
                                                           f"Content before this block was rewritten since it was created, first at `{where}`.")  # fmt: skip

    def _add_thinking(self, body: dict, content: list, index: int) -> list:
        """Sonnet's between_tools mode: a progress-update thinking block before the tool uses."""
        if (body.get("thinking") or {}).get("type") != "between_tools" or not any(b.get("type") == "tool_use" for b in content):
            return content
        block = {"type": "thinking", "thinking": f"Progress update {index}: calling a tool.", "signature": f"sig_fake_{index:04d}"}
        return [block, *content]

    def _cache_usage(self, body: dict, index: int) -> dict:
        blocks = linearize(body)
        prefix, total, digest = [], 0, ""
        hashes = []
        for where, block in blocks:
            total += block_tokens(block)
            prefix.append(total)
            digest = sha256_hex((digest + where).encode() + canonical_json(strip_cache_control(block)))
            hashes.append(digest)
        minimum = self.pricing.price(body["model"]).min_cache_tokens
        marks = [k for k, (_, block) in enumerate(blocks) if "cache_control" in block and prefix[k] >= minimum]
        hits = [k for k in marks if self._cache.get(hashes[k], -1) >= index]
        read_to = max(hits, default=-1)
        read = prefix[read_to] if read_to >= 0 else 0
        writes = {"5m": 0, "1h": 0}
        previous = read
        for k in marks:
            if k > read_to:
                writes[blocks[k][1]["cache_control"].get("ttl", "5m")] += prefix[k] - previous
                previous = prefix[k]
        for k in marks:  # store new prefixes and refresh the ones read
            self._cache[hashes[k]] = index + self.ttl_calls
        created = writes["5m"] + writes["1h"]
        return {
            "input_tokens": total - read - created,
            "cache_creation_input_tokens": created,
            "cache_read_input_tokens": read,
            "cache_creation": {"ephemeral_5m_input_tokens": writes["5m"], "ephemeral_1h_input_tokens": writes["1h"]},
        }

    def as_sync_transport(self):
        import httpx2

        def handler(request: httpx2.Request) -> httpx2.Response:
            return self._http(request, httpx2)

        return httpx2.MockTransport(handler)

    def as_async_transport(self):
        import httpx2

        async def handler(request: httpx2.Request) -> httpx2.Response:
            return self._http(request, httpx2)

        return httpx2.MockTransport(handler)

    def _http(self, request, httpx2):
        path = request.url.path
        assert request.method == "POST" and path in ("/v1/messages", "/v1/messages/count_tokens"), f"unexpected {request.method} {path}"
        self.authorized += "authorization" in request.headers
        body = json.loads(request.content)
        try:
            return httpx2.Response(200, json=self.count_tokens(body) if path.endswith("count_tokens") else self.handle(body))
        except _HTTPError as exc:
            self.errors.append(exc.status)
            return httpx2.Response(exc.status, json=exc.body, headers={"request-id": f"req_fake_{exc.status}"})
        except FakeTimeout:
            raise httpx2.ReadTimeout("simulated timeout", request=request) from None

    def as_fake_client(self) -> SimpleNamespace:
        """A plain object whose `.messages.create(**params)` returns response dicts and whose
        `.messages.count_tokens(**params)` returns an object with `input_tokens` (no SDK).
        HTTP errors raise FakeAPIStatusError; a scripted timeout raises FakeTimeout."""

        def create(**params):
            try:
                return self.handle(params)
            except _HTTPError as exc:
                self.errors.append(exc.status)
                raise FakeAPIStatusError(exc.status, exc.body) from None

        def count_tokens(**params):
            try:
                return SimpleNamespace(**self.count_tokens(params))
            except _HTTPError as exc:
                raise FakeAPIStatusError(exc.status, exc.body) from None

        return SimpleNamespace(messages=SimpleNamespace(create=create, count_tokens=count_tokens))
