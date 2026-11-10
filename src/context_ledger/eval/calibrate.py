"""Token calibration: fit calib-v1 against the API's count_tokens (A.2).

What it does, in order:

1. Sample. A seeded, stratified sample of 64 texts from the corpus source files: 16 `.py`
   chunks, 16 prose chunks (`.rst`, `.txt`), 16 JSON tool-result renderings of real
   corpus_read outputs and 16 mixed texts (a python, a prose and a JSON piece together, class
   "other"). Sizes are log-uniform from 200 bytes to 24 KB. 4 texts per class are held out.
2. Count. Each text is counted as `messages=[{"role": "user", "content": text}]`, no system and
   no tools. The overhead of a 1-byte message ("x"), counted 3 times, minimum taken, is
   subtracted.
3. Fit. Per class, ordinary least squares tokens = a_c * bytes + b_c on the training texts,
   exactly in Fractions; the upper envelope a_c_hi is the 95th percentile (nearest rank) of
   (tokens - b_c) / bytes, never below a_c.
4. Report. bytes/4 error per class and overall (mean absolute, signed mean and max absolute
   percent error) on every text, and calib-v1's point estimate on the held-out texts.
5. Output. `calibration/<model>-<YYYYMMDD>.json` holds sample hashes, sizes, counts,
   coefficients and errors. It never holds sample text or anything secret.

Counting goes through the `TokenCounter` interface: `FakeTokenCounter` (a known synthetic
rule) for tests and offline runs, `CountTokensCounter` (the free count_tokens endpoint, through
a GuardedClient) for the live run only. A calibration made with the fake counter is labelled
"counter": "fake" and is refused wherever a real one is needed (tokens.load_calibration).
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Protocol

from ..envelope import Envelope, envelope_path, prompt_tokens, tool_use_text
from ..ids import canonical_json, sha256_hex
from ..live.redact import safe_message
from ..tokens import BYTES4_V0, CALIBRATION_DIR, CLASSES, CalibEstimator, ClassFit, decimal_str
from .formats import format_read, tool_result_json

PER_CLASS = 16
HELD_OUT = 4
MIN_BYTES, MAX_BYTES = 200, 24 * 1024
PYTHON_EXT, PROSE_EXT = (".py",), (".rst", ".txt")


class CalibrationError(Exception):
    """The corpus cannot supply a sample, or a count failed."""


class TokenCounter(Protocol):
    """Counts the input tokens of one user message holding `text` (no system, no tools)."""

    name: str

    def count_message(self, text: str) -> int: ...


def synthetic_rule(rules: dict[str, tuple[Fraction, Fraction]], classify: Callable[[str], str]) -> Callable[[str], int]:
    """tokens = ceil(a_c * bytes + b_c) with c = classify(text): a known rule to recover."""

    def rule(text: str) -> int:
        a, b = rules[classify(text)]
        return math.ceil(a * len(text.encode("utf-8")) + b)

    return rule


DEFAULT_RULES = {
    "python": (Fraction(10, 33), Fraction(7)),  # 1 token per 3.3 bytes, plus 7
    "prose": (Fraction(10, 42), Fraction(3)),
    "json": (Fraction(10, 30), Fraction(5)),
    "other": (Fraction(10, 36), Fraction(4)),
}


def guess_class(text: str) -> str:
    """A crude classifier for the default fake rule (the real counter needs none)."""
    stripped = text.lstrip()
    if stripped.startswith('{"content"'):
        return "json"
    if "\ndef " in text or "\nclass " in text or "import " in text[:200]:
        return "python"
    return "prose"


class FakeTokenCounter:
    """A deterministic stand-in for count_tokens: `overhead` plus `rule(text)` tokens."""

    name = "fake"

    def __init__(self, rule: Callable[[str], int] | None = None, overhead: int = 8) -> None:
        self.rule = rule or synthetic_rule(DEFAULT_RULES, guess_class)
        self.overhead = overhead
        self.calls = 0

    def count_message(self, text: str) -> int:
        self.calls += 1
        return self.overhead + self.rule(text)


class CountTokensCounter:
    """The live counter: the free `messages.count_tokens` endpoint, through the GuardedClient
    the live factory built. Used only by `calibrate --live`."""

    name = "count_tokens"

    def __init__(self, client, model: str) -> None:
        self.client, self.model = client, model

    def count_message(self, text: str) -> int:
        result = self.client.messages.count_tokens(model=self.model, messages=[{"role": "user", "content": text}])
        tokens = result["input_tokens"] if isinstance(result, dict) else result.input_tokens
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
            raise CalibrationError("count_tokens returned no usable input_tokens")
        return tokens


@dataclass(frozen=True)
class Sample:
    cls: str
    text: str
    origin: str  # "<path>:<line>" of the chunk start (mixed: the three origins)
    held_out: bool

    @property
    def size(self) -> int:
        return len(self.text.encode("utf-8"))


def _log_uniform(rng: random.Random) -> int:
    return int(round(MIN_BYTES * (MAX_BYTES / MIN_BYTES) ** rng.random()))


def _chunk(rng: random.Random, files: Sequence[tuple[str, str]], size: int) -> tuple[str, str, int, int]:
    """A run of whole lines of about `size` bytes from a random file: (path, body, start, end),
    1-based inclusive lines. Lines are added until the size is reached, then the text is cut at
    the last character that fits."""
    big = [f for f in files if len(f[1].encode("utf-8")) >= size] or [max(files, key=lambda f: len(f[1]))]
    path, text = big[rng.randrange(len(big))]
    lines = text.split("\n")
    start = rng.randrange(len(lines))
    body, end = "", start
    while end < len(lines) and len(body.encode("utf-8")) < size:
        body += lines[end] + "\n"
        end += 1
    if len(body.encode("utf-8")) < size and start > 0:  # ran off the end: extend backwards
        while start > 0 and len(body.encode("utf-8")) < size:
            start -= 1
            body = lines[start] + "\n" + body
    body = body.encode("utf-8")[:size].decode("utf-8", errors="ignore")
    return path, body, start + 1, start + max(1, body.count("\n"))


def draw_sample(files: Sequence[tuple[str, str]], seed: int) -> list[Sample]:
    """The stratified sample of 64 texts (16 per class, 4 per class held out), from (path,
    text) pairs of corpus source files. Deterministic in `seed` and the file list."""
    python = sorted(f for f in files if f[0].endswith(PYTHON_EXT))
    prose = sorted(f for f in files if f[0].endswith(PROSE_EXT))
    if not python or not prose:
        raise CalibrationError("the corpus needs .py files and .rst or .txt files to calibrate")
    rng = random.Random(f"calibrate:{seed}")
    out: list[Sample] = []
    for cls in CLASSES:
        held = set(rng.sample(range(PER_CLASS), HELD_OUT))
        for k in range(PER_CLASS):
            size = _log_uniform(rng)
            if cls in ("python", "prose"):
                path, body, start, _ = _chunk(rng, python if cls == "python" else prose, size)
                text, origin = body, f"{path}:{start}"
            elif cls == "json":
                path, body, start, end = _chunk(rng, python if rng.random() < 0.5 else prose, max(1, size * 17 // 20))
                text, origin = tool_result_json(format_read(path, start, end, body.rstrip("\n"))), f"{path}:{start}"
            else:
                parts = []
                for pool in (python, prose, python):
                    path, body, start, end = _chunk(rng, pool, max(1, size // 3))
                    parts.append((path, start, body, end))
                (p1, s1, b1, _), (p2, s2, b2, _), (p3, s3, b3, e3) = parts
                text = b1 + "\n" + b2 + "\n" + tool_result_json(format_read(p3, s3, e3, b3.rstrip("\n")))
                origin = f"{p1}:{s1} {p2}:{s2} {p3}:{s3}"
            out.append(Sample(cls, text, origin, k in held))
    return out


def ols(points: Sequence[tuple[int, int]]) -> tuple[Fraction, Fraction]:
    """Exact least squares y = a x + b over (x, y) points (at least two distinct x)."""
    n = len(points)
    mx = Fraction(sum(x for x, _ in points), n)
    my = Fraction(sum(y for _, y in points), n)
    sxx = sum((x - mx) ** 2 for x, _ in points)
    if sxx == 0:
        raise CalibrationError("least squares needs at least two distinct sizes")
    a = sum((x - mx) * (y - my) for x, y in points) / sxx
    return a, my - a * mx


def percentile(values: Sequence[Fraction], q_percent: int) -> Fraction:
    """Nearest-rank percentile: the ceil(q/100 * n)-th smallest value (1-based)."""
    ordered = sorted(values)
    rank = max(1, -((-q_percent * len(ordered)) // 100))
    return ordered[rank - 1]


def error_stats(pairs: Sequence[tuple[int, int]]) -> dict[str, Fraction | int]:
    """Percent errors of (estimate, actual) pairs with actual > 0: mean absolute (mape), signed
    mean (bias, positive = overestimate) and max absolute (max_ape), as exact Fractions."""
    errors = [Fraction(100 * (est - act), act) for est, act in pairs if act > 0]
    if not errors:
        return {"n": 0, "mape": Fraction(0), "bias": Fraction(0), "max_ape": Fraction(0)}
    return {
        "n": len(errors),
        "mape": sum(abs(e) for e in errors) / len(errors),
        "bias": sum(errors) / len(errors),
        "max_ape": max(abs(e) for e in errors),
    }


def _show(stats: dict) -> dict:
    return {k: (v if k == "n" else decimal_str(v, 4)) for k, v in stats.items()}


def fit_class(points: Sequence[tuple[int, int]]) -> ClassFit:
    """OLS fit and 95th-percentile envelope, rounded to the 6-decimal strings that get stored."""
    a, b = ols(points)
    a = Fraction(decimal_str(a))
    b = Fraction(decimal_str(b))
    if a <= 0:
        raise CalibrationError("the fitted tokens-per-byte slope is not positive")
    envelope = percentile([(Fraction(y) - b) / x for x, y in points], 95)
    a_hi = max(Fraction(decimal_str(envelope)), a)
    return ClassFit(a, b, a_hi)


def calibrate(sample: Sequence[Sample], counter: TokenCounter, *, model: str, seed: int, created: str | None = None) -> dict:
    """Count every sample, fit calib-v1 per class and report errors. Returns the calibration
    file contents (no text: hashes, sizes and counts only)."""
    overhead = min(counter.count_message("x") for _ in range(3))
    counted = [(s, counter.count_message(s.text) - overhead) for s in sample]
    coefficients: dict[str, ClassFit] = {}
    for cls in CLASSES:
        train = [(s.size, t) for s, t in counted if s.cls == cls and not s.held_out]
        coefficients[cls] = fit_class(train)
    estimator = CalibEstimator(model, coefficients, "0" * 64)
    bytes4 = {cls: error_stats([(BYTES4_V0.count_bytes(s.size), t) for s, t in counted if s.cls == cls]) for cls in CLASSES}
    bytes4["overall"] = error_stats([(BYTES4_V0.count_bytes(s.size), t) for s, t in counted])
    held = [(s, t) for s, t in counted if s.held_out]
    calib = {cls: error_stats([(estimator.estimate_bytes(s.size, cls), t) for s, t in held if s.cls == cls]) for cls in CLASSES}
    calib["overall"] = error_stats([(estimator.estimate_bytes(s.size, s.cls), t) for s, t in held])
    bytes4_held = error_stats([(BYTES4_V0.count_bytes(s.size), t) for s, t in held])
    covered = sum(estimator.count_bytes(s.size, s.cls) >= t for s, t in held)
    return {
        "version": "calib-v1",
        "model": model,
        "counter": counter.name,
        "created": created or _dt.date.today().isoformat(),
        "seed": seed,
        "overhead_tokens": overhead,
        "samples": [
            {"cls": s.cls, "bytes": s.size, "sha256": sha256_hex(s.text.encode("utf-8")), "tokens": t, "held_out": s.held_out}
            for s, t in counted
        ],
        "coefficients": {cls: coefficients[cls].to_dict() for cls in CLASSES},
        "errors": {
            "bytes4_all": {k: _show(v) for k, v in bytes4.items()},
            "bytes4_held_out": _show(bytes4_held),
            "calib_v1_held_out": {k: _show(v) for k, v in calib.items()},
            "envelope_covers_held_out": f"{covered}/{len(held)}",
        },
    }


def calibration_path(model: str, created: str, directory: str | Path | None = None) -> Path:
    """calibration/<model>-<YYYYMMDD>.json."""
    return Path(directory if directory is not None else CALIBRATION_DIR) / f"{model}-{created.replace('-', '')}.json"


def write_calibration(result: dict, directory: str | Path | None = None) -> Path:
    """Write the calibration file (canonical-JSON-safe, indented for review) and return its path."""
    canonical_json(result)  # ints and strings only
    path = calibration_path(result["model"], result["created"], directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def render_report(result: dict) -> str:
    """A short plain-text summary of the fit and its errors, for the CLI."""
    lines = [f"calibration of {result['model']} with the {result['counter']} counter, overhead {result['overhead_tokens']} tokens"]
    for cls in CLASSES:
        c = result["coefficients"][cls]
        b4 = result["errors"]["bytes4_all"][cls]
        cv = result["errors"]["calib_v1_held_out"][cls]
        lines.append(
            f"  {cls:<7} tokens = {c['a']} x bytes + {c['b']} (envelope {c['a_hi']}); "
            f"bytes/4 error {b4['mape']}% (bias {b4['bias']}%, max {b4['max_ape']}%); calib-v1 held-out {cv['mape']}%"
        )
    overall = result["errors"]
    lines.append(f"  overall bytes/4 error {overall['bytes4_all']['overall']['mape']}%, calib-v1 held-out {overall['calib_v1_held_out']['overall']['mape']}%")
    lines.append(f"  the envelope covers {overall['envelope_covers_held_out']} held-out texts")
    return "\n".join(lines)


# Envelope calibration (BUILD-V2 item 4): the structural tokens of messages and content blocks.

ENVELOPE_K = range(0, 9)
_TOOL = {"name": "t", "description": "x", "input_schema": {"type": "object", "properties": {}}}
_X = {"type": "text", "text": "x"}
_THINKING = {"type": "thinking", "thinking": "x", "signature": "c2lnbmF0dXJl"}


class RequestCounter(Protocol):
    """Counts the input tokens of a request body (system, messages, tools, thinking)."""

    name: str

    def count_request(self, body: dict) -> int: ...


class FakeEnvelopeCounter:
    """A deterministic stand-in for count_tokens on whole requests: `envelope` and `content`
    (an estimator) applied by envelope.prompt_tokens, plus `tool_tokens(tools)` for the tool
    definitions. Fitting against it must recover `envelope` exactly."""

    name = "fake"

    def __init__(self, envelope, content, tool_tokens: Callable[[list], int] | None = None) -> None:
        self.envelope, self.content = envelope, content
        self.tool_tokens = tool_tokens or (lambda tools: 300 + len(canonical_json(tools)) // 4 if tools else 0)
        self.bodies: list[dict] = []

    def count_request(self, body: dict) -> int:
        self.bodies.append(json.loads(json.dumps(body)))
        return prompt_tokens(body, self.content, self.envelope) + self.tool_tokens(body.get("tools") or [])


class CountTokensRequestCounter:
    """The live counter: `messages.count_tokens(model=..., **body)` through the GuardedClient
    the live factory built (free). Used only by `calibrate --live --envelope`."""

    name = "count_tokens"

    def __init__(self, client, model: str) -> None:
        self.client, self.model = client, model

    def count_request(self, body: dict) -> int:
        result = self.client.messages.count_tokens(model=self.model, **body)
        tokens = result["input_tokens"] if isinstance(result, dict) else result.input_tokens
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
            raise CalibrationError("count_tokens returned no usable input_tokens")
        return tokens


def _tool_use(n: int) -> dict:
    return {"type": "tool_use", "id": f"toolu_env_{n:03d}", "name": "t", "input": {"q": "x"}}


def _tool_result(n: int) -> dict:
    return {"type": "tool_result", "tool_use_id": f"toolu_env_{n:03d}", "content": [dict(_X)]}


def envelope_requests(thinking: dict | None = None) -> list[tuple[str, int, dict]]:
    """The synthetic requests, (kind, k, body) for k = 0..8 of each kind, all with one minimal
    tool definition (tool_use blocks need `tools`) and the model's thinking config:
        text      one user message with 1 + k text blocks "x"
        message   2k + 1 alternating messages of one text block "x" each
        tool      user "x"; assistant "x" + k tool_use blocks; user k tool_results + "x"
        thinking  user "x"; assistant k thinking blocks + "x"; user "x"
    """
    out = []
    for k in ENVELOPE_K:
        user = lambda *blocks: {"role": "user", "content": [dict(b) for b in blocks]}  # noqa: E731
        assistant = lambda *blocks: {"role": "assistant", "content": [dict(b) for b in blocks]}  # noqa: E731
        bodies = {
            "text": [user(*[_X] * (1 + k))],
            "message": [(user if m % 2 == 0 else assistant)(_X) for m in range(2 * k + 1)],
            "tool": [user(_X), assistant(_X, *[_tool_use(n) for n in range(k)]), user(*[_tool_result(n) for n in range(k)], _X)],
            "thinking": [user(_X), assistant(*[_THINKING] * k, _X), user(_X)],
        }
        for kind, messages in bodies.items():
            body = {"messages": messages, "tools": [dict(_TOOL)]}
            if thinking is not None:
                body["thinking"] = dict(thinking)
            out.append((kind, k, body))
    return out


def arm_tools() -> dict[str, list[dict]]:
    """Each arm's tool definitions, sorted by name as the arms send them."""
    from ..tools import CTX_TOOLS
    from .tools import CORPUS_TOOLS

    ordered = lambda tools: sorted((json.loads(json.dumps(t)) for t in tools), key=lambda t: t["name"])  # noqa: E731
    return {"B3": [], "B4": ordered(CORPUS_TOOLS), "L": ordered(CTX_TOOLS + CORPUS_TOOLS)}


def _round(value: Fraction) -> int:
    """Half-up to an integer, clamped at 0 (a structural cost is never negative)."""
    return max(0, int(decimal_str(value, 0)))


def calibrate_envelope(counter: RequestCounter, content, *, model: str, thinking: dict | None = None, created: str | None = None) -> dict:
    """Count the synthetic requests, fit each kind's slope over k by least squares, derive the
    envelope constants (envelope.py) net of `content` (the calib-v2 estimator of the model's
    calib-v1 calibration), and measure each arm's tool definitions. Returns the file contents:
    hashes, counts, fits and constants, never any text. A kind whose requests the counter
    rejects (thinking blocks may be) is recorded as unmeasured with constant 0."""
    c_other = content.count("x", "other")
    c_json = content.count("x", "json")
    requests = envelope_requests(thinking)
    rows, errors, slopes, fits = [], {}, {}, {}
    for kind, k, body in requests:
        if kind in errors:
            continue
        try:
            tokens = counter.count_request(body)
        except Exception as exc:  # noqa: BLE001 - recorded in the file, never swallowed silently
            if kind != "thinking":
                raise CalibrationError(f"count_tokens failed on the {kind} requests: {safe_message(exc)}") from None
            errors[kind] = safe_message(exc)
            continue
        rows.append({"kind": kind, "k": k, "sha256": sha256_hex(canonical_json(body)), "tokens": tokens})
    rows = [r for r in rows if r["kind"] not in errors]
    for kind in ("text", "message", "tool", "thinking"):
        points = [(r["k"], r["tokens"]) for r in rows if r["kind"] == kind]
        if kind in errors or len(points) < 2:
            continue
        a, b = ols(points)
        slopes[kind] = a
        fits[kind] = {"slope": decimal_str(a), "intercept": decimal_str(b)}
    minimal = {"messages": [{"role": "user", "content": [dict(_X)]}]}
    if thinking is not None:
        minimal["thinking"] = dict(thinking)
    no_tools = counter.count_request(minimal)
    with_tool = counter.count_request({**minimal, "tools": [dict(_TOOL)]})
    text = slopes["text"] - c_other
    message = slopes["message"] / 2 - slopes["text"]
    tool_use = slopes["tool"] - content.count(tool_use_text(_tool_use(0)), "json") - text - c_json
    raw = {
        "base": Fraction(no_tools) - message - text - c_other,
        "message": message,
        "text": text,
        "tool_use": tool_use,
        "tool_result": Fraction(0),
        "thinking": slopes["thinking"] - c_other if "thinking" in slopes else Fraction(0),
    }
    envelope = Envelope(**{key: _round(value) for key, value in raw.items()})
    minimal_tool = with_tool - no_tools
    bodies = {(kind, k): body for kind, k, body in requests}
    residual = max(abs(prompt_tokens(bodies[(r["kind"], r["k"])], content, envelope) + minimal_tool - r["tokens"]) for r in rows)
    definitions = {}
    for arm, tools in arm_tools().items():
        tokens = counter.count_request({**minimal, "tools": tools}) - no_tools if tools else 0
        definitions[arm] = {"tokens": tokens, "tools_sha256": sha256_hex(canonical_json(tools))}
    return {
        "version": "envelope-v1",
        "model": model,
        "counter": counter.name,
        "created": created or _dt.date.today().isoformat(),
        "content_estimator": content.payload(),
        "calibration_sha256": content.calibration_sha256,
        "thinking": thinking,
        "requests": rows,
        "baseline": {"no_tools": no_tools, "minimal_tool": minimal_tool},
        "fits": fits,
        "raw": {key: decimal_str(value, 4) for key, value in raw.items()},
        "envelope": envelope.to_dict(),
        "note": "tool_use and tool_result always come in pairs, so only their sum is identifiable; it is attributed to tool_use",
        "unmeasured": sorted(errors),
        "errors": errors,
        "max_abs_residual": residual,
        "tool_definitions": definitions,
    }


def write_envelope(result: dict, directory: str | Path | None = None) -> Path:
    """Write calibration/<model>-envelope-<YYYYMMDD>.json; refuses to overwrite any file."""
    canonical_json(result)
    path = envelope_path(result["model"], result["created"], directory)
    if path.exists():
        raise CalibrationError(f"{path} exists; calibration files are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def render_envelope_report(result: dict) -> str:
    e = result["envelope"]
    lines = [f"envelope of {result['model']} with the {result['counter']} counter (content: {result['content_estimator']['estimator']})"]
    lines.append("  " + ", ".join(f"{k} {e[k]}" for k in ("base", "message", "text", "tool_use", "tool_result", "thinking")))
    lines.append(f"  fits: " + "; ".join(f"{k} slope {v['slope']}" for k, v in sorted(result["fits"].items())))
    if result["unmeasured"]:
        lines.append(f"  unmeasured (constant 0): {', '.join(result['unmeasured'])}")
    lines.append(f"  max abs residual {result['max_abs_residual']} tokens over {len(result['requests'])} requests")
    lines.append("  tool definitions: " + ", ".join(f"{arm} {d['tokens']}" for arm, d in sorted(result["tool_definitions"].items())))
    return "\n".join(lines)
