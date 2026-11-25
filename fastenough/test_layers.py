"""The layers of the package: tasks know nothing of design or research, and design knows nothing of research.

At a dataset release fastenough/research is removed; this test is what keeps the rest working without it
(PLAN.md section 9). It reads the import lines of every file and runs nothing.
Run: python -m pytest -p no:cacheprovider -q fastenough/test_layers.py   (PYTHONPATH = repo root)
"""
import ast
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
MAY_NOT_IMPORT = {"tasks": ("fastenough.design", "fastenough.research"), "design": ("fastenough.research",)}


def imported(path: Path) -> set:
    """Every module name a file imports, written out in full."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


@pytest.mark.parametrize("layer", sorted(MAY_NOT_IMPORT))
def test_a_layer_does_not_import_the_layers_above_it(layer):
    files = sorted((HERE / layer).rglob("*.py"))
    assert files, f"no files found under {layer}"
    bad = [(str(p.relative_to(HERE)), n) for p in files for n in sorted(imported(p)) if n.startswith(MAY_NOT_IMPORT[layer])]
    assert not bad, bad
