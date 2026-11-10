"""Core purity: the core is stdlib-only, and nothing imports the SDK at import time."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "context_ledger"


def top_level_imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


# The live factory builds the real SDK client, so it imports the SDK (lazily, see below).
SDK_MODULES = {PACKAGE / "live" / "factory.py"}
# analyze-v3 draws main_figure.png only if matplotlib is importable; it is no dependency (lazily, see below).
OPTIONAL_IMPORTS = {PACKAGE / "eval" / "analyze_v3.py": {"matplotlib"}}


def module_level_imports(path: Path) -> set[str]:
    return {
        alias.name.split(".")[0]
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")])
    }


def test_core_modules_import_only_the_standard_library():
    core = [p for p in PACKAGE.rglob("*.py") if "anthropic" not in p.relative_to(PACKAGE).parts and p not in SDK_MODULES]
    assert len(core) > 15
    for path in core:
        outside = top_level_imports(path) - set(sys.stdlib_module_names) - {"context_ledger"} - OPTIONAL_IMPORTS.get(path, set())
        assert not outside, f"{path.relative_to(PACKAGE)} imports {outside}"


def test_optional_packages_are_imported_only_lazily():
    for path, packages in OPTIONAL_IMPORTS.items():
        assert top_level_imports(path) & packages and not module_level_imports(path) & packages, path.name


def test_anthropic_subpackage_imports_the_sdk_only_lazily():
    for path in [*(PACKAGE / "anthropic").glob("*.py"), *SDK_MODULES]:
        module_level = {
            alias.name.split(".")[0]
            for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")])
        }
        assert not module_level & {"anthropic", "httpx2", "httpx"}, path.name


def test_everything_imports_without_the_sdk():
    """In a fresh interpreter where `anthropic` and `httpx2` cannot be imported."""
    code = (
        "import sys; sys.modules['anthropic'] = None; sys.modules['httpx2'] = None\n"
        "import context_ledger, context_ledger.session, context_ledger.invariants, context_ledger.cost\n"
        "import context_ledger.tools, context_ledger.prompts, context_ledger.cli\n"
        "import context_ledger.anthropic, context_ledger.anthropic.client, context_ledger.anthropic.fake\n"
        "import context_ledger.live.factory, context_ledger.live.keys, context_ledger.live.gate, context_ledger.live.redact\n"
        "print('ok')\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=PACKAGE.parent, timeout=60)
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr
