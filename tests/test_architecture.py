"""Guard the import directions that keep components independently replaceable."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "eo_visual_retrieval"
ROOT_MODULE = "eo_visual_retrieval"


def _imported_modules(source: Path) -> Iterator[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


def _package_modules(relative: str) -> list[Path]:
    return sorted((PACKAGE / relative).rglob("*.py"))


def test_encoders_do_not_depend_on_the_benchmark_package() -> None:
    """A representation must not import the benchmark that first happened to use it.

    Dataset identity lives in ``datasets/``; ``benchmarks/`` composes it into one
    specific experiment. Reversing that would make every new encoder inherit
    EuroSAT-specific code.
    """

    offenders = {
        module.relative_to(PACKAGE).as_posix(): sorted(
            name
            for name in _imported_modules(module)
            if name.startswith(f"{ROOT_MODULE}.benchmarks")
        )
        for module in _package_modules("embeddings")
    }
    assert {path: names for path, names in offenders.items() if names} == {}


def test_leaf_modules_have_no_intra_package_dependencies() -> None:
    """``hashing`` and ``vectors`` are shared by everything, so they import nothing."""

    for name in ("hashing.py", "vectors.py"):
        imported = [
            module
            for module in _imported_modules(PACKAGE / name)
            if module.startswith(ROOT_MODULE)
        ]
        assert imported == [], f"{name} must stay a leaf, but imports {imported}"


def test_dataset_identity_module_depends_only_on_leaves() -> None:
    allowed = {f"{ROOT_MODULE}.hashing", f"{ROOT_MODULE}.models"}
    imported = {
        module
        for module in _imported_modules(PACKAGE / "datasets" / "eurosat.py")
        if module.startswith(ROOT_MODULE)
    }
    assert imported <= allowed, f"unexpected dependencies: {sorted(imported - allowed)}"


def test_digest_normalization_and_distance_have_exactly_one_implementation() -> None:
    """Provenance, normalization, and great-circle distance drift silently when re-implemented."""

    definitions: dict[str, list[str]] = {
        "file_sha256": [],
        "file_md5": [],
        "l2_normalize": [],
        "nearest_distances_m": [],
    }
    for module in _package_modules("."):
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in definitions:
                definitions[node.name].append(module.relative_to(PACKAGE).as_posix())

    assert definitions["file_sha256"] == ["hashing.py"]
    assert definitions["file_md5"] == ["hashing.py"]
    assert definitions["l2_normalize"] == ["vectors.py"]
    assert definitions["nearest_distances_m"] == ["benchmarks/coverage.py"]
