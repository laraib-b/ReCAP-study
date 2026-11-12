"""Static checks that marimo notebooks follow the one-definition-per-name rule."""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.py"))


def _is_cell(node: ast.AST) -> bool:
    return isinstance(node, ast.FunctionDef) and any(
        isinstance(d, ast.Attribute) and d.attr == "cell" for d in node.decorator_list
    )


def _assigned_names(func: ast.FunctionDef) -> set[str]:
    names: set[str] = set()

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            return
        if isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        for child in ast.iter_child_nodes(node):
            visit(child)

    for stmt in func.body:
        visit(stmt)
    return {n for n in names if not n.startswith("_")}


def check_notebook(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cells = [n for n in ast.walk(tree) if _is_cell(n)]
    owners: dict[str, list[int]] = defaultdict(list)
    for index, cell in enumerate(cells):
        for name in _assigned_names(cell):
            owners[name].append(index)

    errors = [
        f"{path.name}: '{name}' defined in cells {idx}"
        for name, idx in sorted(owners.items())
        if len(idx) > 1
    ]
    for index, cell in enumerate(cells):
        for arg in cell.args.args:
            if arg.arg not in owners:
                errors.append(f"{path.name}: cell {index} uses '{arg.arg}' but no cell defines it")
    return errors


def test_marimo_notebooks_have_unique_definitions() -> None:
    errors = [e for nb in NOTEBOOKS if "marimo" in nb.read_text(encoding="utf-8") for e in check_notebook(nb)]
    assert not errors, "\n".join(errors)


if __name__ == "__main__":
    found = [e for nb in NOTEBOOKS if "marimo" in nb.read_text(encoding="utf-8") for e in check_notebook(nb)]
    print("\n".join(found) if found else "OK: no duplicate definitions")
    raise SystemExit(1 if found else 0)
