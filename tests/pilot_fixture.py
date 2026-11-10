"""A tiny synthetic CPython-like source tarball for the pilot tests (no real CPython needed).

About a dozen files under Python-3.15.0/: library modules with upper-case constants, literal
parameter defaults, direct raises and rare function names; a test module; reST docs that
mention some of the functions; and a NEWS file. The tarball is built in-test with `tarfile`;
its sha256 is the test-only override the builder checks against.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

TOPICS = [
    ("spool", "Spooler", "drain"),
    ("ledgerbook", "Bookkeeper", "tally"),
    ("weather", "Forecaster", "predict"),
    ("garden", "Gardener", "prune"),
    ("harbor", "Harbormaster", "dock"),
    ("orchard", "Picker", "harvest"),
]
EXCEPTIONS = ["ValueError", "KeyError", "TypeError", "RuntimeError", "LookupError", "OverflowError"]


def module_text(n: int) -> str:
    name, cls, verb = TOPICS[n]
    up = name.upper()
    funcs = []
    for k in range(5):
        exc = EXCEPTIONS[(n + k) % len(EXCEPTIONS)]
        funcs.append(f'''
def {verb}_{name}_step{k}(value, retries={3 + n + k}, label="{name}-{k}", scale={1.5 + k}):
    """Apply step {k} of the {name} pipeline to `value`, `retries` times."""
    if value is None:
        raise {exc}("{name} step {k} needs a value")
    if retries < 0:
        raise ValueError("retries must not be negative")
    total = value
    for _ in range(retries):
        total = total * scale + len(label)
    return total
''')
    return f'''"""The {name} module: helpers that {verb} things, for the pilot fixture."""

import os

{up}_LIMIT = {120 + 7 * n}
{up}_RATIO = {0.25 + n / 100}
{up}_LABEL = "{name}-default"
{up}_MODE = 'strict'
{"".join(funcs)}

class {cls}:
    """Keeps state for {name} operations."""

    def __init__(self, capacity={40 + n}, policy="fifo"):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.policy = policy

    def {verb}_all(self, items, limit={10 + n}):
        """{verb.capitalize()} every item, up to `limit` of them."""
        if len(items) > limit:
            raise OverflowError("too many items")
        return [item for item in items[:limit]]
'''


DOCS = {
    "Doc/library/spool.rst": "The spool module\n================\n\n" + "".join(
        f"Call :func:`drain_spool_step{k}` to run step {k}; it raises when the value is missing.\n\n" for k in range(5)
    ) * 3,
    "Doc/library/weather.rst": "The weather module\n==================\n\n" + "Forecasts use :func:`predict_weather_step2` and friends.\n\n" * 20,
    "Doc/whatsnew/3.15.rst": "What's new\n==========\n\n" + "Many modules gained new helpers in this release.\n" * 40,
}
OTHER = {
    "Lib/test/test_spool.py": "import unittest\n\nfrom spool import drain_spool_step0\n\n\nclass SpoolTest(unittest.TestCase):\n    def test_step(self):\n        self.assertEqual(drain_spool_step0(2, retries=1), 2 * 1.5 + 7)\n" * 3,
    "Misc/NEWS.txt": "".join(f"- gh-{1000 + k}: Fixed a bug in the harbor module ({k}).\n" for k in range(120)),
    "README.md": "This is the pilot fixture of Python 3.15.0.\n" * 30,
    "Modules/spoolmodule.c": "/* spool accelerator */\n#include \"Python.h\"\n\nstatic int spool_limit = 120;\n" * 10,
}


def fixture_files() -> dict[str, str]:
    files = {f"Lib/{TOPICS[n][0]}.py": module_text(n) for n in range(len(TOPICS))}
    return {**files, **DOCS, **OTHER}


def write_fixture_tarball(directory: Path, files: dict[str, str] | None = None) -> tuple[Path, str]:
    """Write Python-3.15.0.tgz with the fixture files (or `files`) into `directory`; (path, sha256)."""
    path = Path(directory) / "Python-3.15.0.tgz"
    with tarfile.open(path, "w:gz") as tar:
        for name, text in sorted((files or fixture_files()).items()):
            data = text.encode("utf-8")
            info = tarfile.TarInfo(f"Python-3.15.0/{name}")
            info.size, info.mtime = len(data), 0
            tar.addfile(info, io.BytesIO(data))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def fleet_dock() -> str:
    """A gold module with multi-hop questions: defaults and constants naming constants of the
    same module, of fleet/consts.py (relative imports) and of harbor.py (an absolute import);
    and a function whose name another module also defines (a confusable subject)."""
    funcs = "".join(
        f'''
def berth_ship_{k}(name, slot={20 + k}, tag="berth-{k}"):
    """Berth ship `name` in `slot` number {k}."""
    if not name:
        raise KeyError("ship {k} needs a name")
    return f"{{tag}}:{{name}}:{{slot}}"
'''
        for k in range(4)
    )
    return f'''"""The dock module: mooring helpers, for the v2 pilot fixture."""

import harbor
from . import consts
from .consts import DOCK_LIMIT, PORT_NAME

LOCAL_SPEED = 14
SIGNAL_CODE = "dock-signal"
BERTH_LIMIT = DOCK_LIMIT
HOME_PORT = consts.PORT_NAME
HARBOR_CAP = harbor.HARBOR_LIMIT
CRUISE_SPEED = LOCAL_SPEED


def moor_vessel(name, limit=DOCK_LIMIT, port=PORT_NAME, speed=LOCAL_SPEED):
    """Moor `name` at `port`, at most `limit` vessels, arriving at `speed` knots."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    return (name, limit, port, speed)


def tide_wait(hours, ratio=consts.TIDE_RATIO, cap=harbor.HARBOR_LIMIT):
    """Hours to wait for the tide, scaled by `ratio`, at most `cap`."""
    if hours < 0:
        raise ValueError("hours must not be negative")
    return min(hours * ratio, cap)


def drain_spool_step0(value, retries=9, label="dock-0", scale=2.5):
    """The dock's own step 0: a confusable name (spool.py defines one too)."""
    if value is None:
        raise TypeError("dock step 0 needs a value")
    return value * scale + retries + len(label)
{funcs}

class Dockmaster:
    """Keeps the dock's books."""

    def __init__(self, berths=6, mode="tidal"):
        if berths <= 0:
            raise ValueError("berths must be positive")
        self.berths = berths
        self.mode = mode

    def assign_berth(self, ship, preferred=LOCAL_SPEED):
        """Assign `ship` a berth, `preferred` first."""
        if ship is None:
            raise LookupError("no ship")
        return preferred
'''


FLEET = {
    "Lib/fleet/__init__.py": '"""The fleet package."""\n\nFLEET_SIZE = 12\n',
    "Lib/fleet/consts.py": '"""Fleet-wide constants."""\n\nDOCK_LIMIT = 48\nPORT_NAME = "north-port"\nTIDE_RATIO = 0.75\n',
}


def v2_fixture_files() -> dict[str, str]:
    """fixture_files plus the fleet package (Lib/fleet/dock.py is a gold module with multi-hop
    questions) and a second definition of drain_spool_step0 outside the gold files."""
    files = fixture_files()
    files.update(FLEET)
    files["Lib/fleet/dock.py"] = fleet_dock()
    files["Tools/scripts/spool_tool.py"] = '"""A tool script."""\n\n\ndef drain_spool_step0(value):\n    return value\n\n\ndef predict_weather_step1(x):\n    return x\n'
    return files


TOKEN_NAMES = [
    "ENDMARKER", "NAME", "NUMBER", "STRING", "NEWLINE", "INDENT", "DEDENT", "LPAR", "RPAR", "LSQB", "RSQB", "COLON", "COMMA", "SEMI",
    "PLUS", "MINUS", "STAR", "SLASH", "VBAR", "AMPER", "LESS", "GREATER", "EQUAL", "DOT", "PERCENT", "LBRACE", "RBRACE", "EQEQUAL",
    "NOTEQUAL", "LESSEQUAL", "GREATEREQUAL", "TILDE", "CIRCUMFLEX", "LEFTSHIFT", "RIGHTSHIFT", "DOUBLESTAR", "PLUSEQUAL", "MINEQUAL",
    "STAREQUAL", "SLASHEQUAL", "PERCENTEQUAL", "AMPEREQUAL", "VBAREQUAL", "CIRCUMFLEXEQUAL", "LEFTSHIFTEQUAL", "RIGHTSHIFTEQUAL",
    "DOUBLESTAREQUAL", "DOUBLESLASH", "DOUBLESLASHEQUAL", "AT", "ATEQUAL", "RARROW", "ELLIPSIS", "COLONEQUAL", "EXCLAMATION", "OP",
]  # fmt: skip


def token_table() -> str:
    """Lib/tokentab.py, shaped like CPython's Lib/token.py: 56 one-line constants, so one
    corpus_read in L is dozens of tiny spans (the read that collapsed pilot v2b's L, BUILD-V2c)."""
    head = '"""Token constants."""\n# Auto-generated\n\n__all__ = [' + ", ".join(repr(n) for n in TOKEN_NAMES[:8]) + "]\n\n"
    return head + "".join(f"{name} = {k}\n" for k, name in enumerate(TOKEN_NAMES)) + "\n\ndef ISTERMINAL(x):\n    return x < 250\n"
