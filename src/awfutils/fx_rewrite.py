"""Small declarative wrappers around PyTorch FX subgraph rewriting.

The Inductor graph-pass interface is private and version-sensitive. Consumers
should test this module against the exact PyTorch version they pin.
"""

from __future__ import annotations

import hashlib
import inspect
import os
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor

# CustomGraphPass and get_hash_for_files are private PyTorch APIs. Keeping them
# isolated here makes version-sensitive integration failures direct and local.
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files
from torch.fx import Graph, GraphModule, Node, symbolic_trace
from torch.fx.subgraph_rewriter import replace_pattern_with_filters


def _tensor_value(node: Node) -> Tensor | None:
    """Read tensor metadata in the formats emitted by Dynamo and FX passes."""
    for key in ("val", "example_value"):
        value = node.meta.get(key)
        if isinstance(value, Tensor):
            return value
    return None


def _tensor_ndim(node: Node) -> int | None:
    value = _tensor_value(node)
    if value is not None:
        return value.ndim
    tensor_meta = node.meta.get("tensor_meta")
    return len(tensor_meta.shape) if tensor_meta is not None else None


def _tensor_shape(node: Node) -> tuple[Any, ...] | None:
    value = _tensor_value(node)
    if value is not None:
        return tuple(value.shape)
    tensor_meta = node.meta.get("tensor_meta")
    return tuple(tensor_meta.shape) if tensor_meta is not None else None


def _known_equal_dimension(left: Any, right: Any) -> bool:
    """Return true only when two dimensions are statically and safely equal."""
    if type(left) is not int or type(right) is not int:
        return False
    return left == right


def _source_file(subject: object) -> str | None:
    try:
        path = inspect.getsourcefile(subject)
    except TypeError:
        return None
    return os.path.realpath(path) if path is not None else None


class Rewrite:
    """Trace declarative find/replace functions and guard matches with FX metadata."""

    def __init__(
        self,
        *,
        find: Callable[..., Any],
        replace: Callable[..., Any],
        cond: Callable[..., bool],
        cache_key: str,
        behavior_files: Iterable[str] = (),
    ) -> None:
        find_names = tuple(inspect.signature(find).parameters)
        replace_names = tuple(inspect.signature(replace).parameters)
        condition_names = tuple(inspect.signature(cond).parameters)
        if replace_names != find_names or condition_names != find_names:
            raise TypeError("find, replace, and cond must have identical parameters")

        self._names = find_names
        self._find = symbolic_trace(find)
        self._replace = symbolic_trace(replace)
        self.cache_key = cache_key
        self.match_count = 0
        callable_files = (_source_file(function) for function in (find, replace, cond))
        self.behavior_files = tuple(
            sorted(
                {
                    os.path.realpath(path)
                    for path in (*behavior_files, *callable_files)
                    if path is not None
                }
            )
        )

        placeholders = {
            str(node.target): node
            for node in self._find.graph.nodes
            if node.op == "placeholder"
        }
        if tuple(placeholders) != self._names:
            raise RuntimeError(
                f"traced placeholders {tuple(placeholders)} do not match {self._names}"
            )

        def match_filter(match, _graph: Graph, _pattern: Graph) -> bool:
            bound = tuple(match.nodes_map[placeholders[name]] for name in self._names)
            if not all(isinstance(node, Node) for node in bound):
                return False
            return cond(*bound)

        self._match_filters = [match_filter]

    def apply(self, graph_module: GraphModule) -> int:
        matches = replace_pattern_with_filters(
            graph_module,
            self._find,
            self._replace,
            match_filters=self._match_filters,
        )
        count = len(matches)
        self.match_count += count
        return count


class DeclarativeRewritePass(CustomGraphPass):
    """Apply declarative rewrites as an Inductor pre-gradient graph pass."""

    def __init__(
        self,
        *rewrites: Rewrite,
        behavior_files: Iterable[str] = (),
    ) -> None:
        if not rewrites:
            raise ValueError("A declarative rewrite pass needs at least one rewrite")
        self.rewrites = rewrites
        self.match_count = 0
        self.invocation_count = 0
        source_files = {
            os.path.realpath(__file__),
            *(os.path.realpath(path) for path in behavior_files),
            *(path for rewrite in rewrites for path in rewrite.behavior_files),
        }
        subclass_file = _source_file(type(self))
        if subclass_file is not None:
            source_files.add(subclass_file)
        self.behavior_files = tuple(sorted(source_files))

    def __call__(self, graph: Graph) -> None:
        graph_module = graph.owning_module
        if graph_module is None:
            raise RuntimeError("Inductor pre-grad graph has no owning GraphModule")
        self.invocation_count += 1
        for rewrite in self.rewrites:
            self.match_count += rewrite.apply(graph_module)
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

    def uuid(self) -> bytes:
        file_hash = get_hash_for_files(self.behavior_files)
        keys = "\0".join(rewrite.cache_key for rewrite in self.rewrites).encode()
        return hashlib.sha256(file_hash + b"\0" + keys).digest()


def _test_diagonal_matmul(matrix, scale):
    return matrix @ torch.diag(scale)


def _test_column_scale(matrix, scale):
    return matrix * scale


def _test_rank_and_shape_condition(matrix, scale):
    matrix_value = matrix.meta.get("example_value")
    scale_value = scale.meta.get("example_value")
    return (
        isinstance(matrix_value, torch.Tensor)
        and isinstance(scale_value, torch.Tensor)
        and matrix_value.ndim == 2
        and scale_value.ndim == 1
        and matrix_value.shape[1] == scale_value.shape[0]
    )


def _test_rewrite(cache_key="test-example-value-rewrite"):
    return Rewrite(
        find=_test_diagonal_matmul,
        replace=_test_column_scale,
        cond=_test_rank_and_shape_condition,
        cache_key=cache_key,
    )


class _TestRewritePass(DeclarativeRewritePass):
    pass


def test_declarative_rewrite_binds_placeholders_and_reads_metadata():
    graph_module = symbolic_trace(_test_diagonal_matmul)
    placeholders = [
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    ]
    placeholders[0].meta["example_value"] = torch.randn(2, 3)
    placeholders[1].meta["example_value"] = torch.randn(3)

    rewrite = _test_rewrite()

    assert rewrite.apply(graph_module) == 1
    assert "diag" not in graph_module.code
    assert "matmul" not in graph_module.code
    matrix = torch.randn(2, 3)
    scale = torch.randn(3)
    torch.testing.assert_close(
        graph_module(matrix, scale),
        _test_column_scale(matrix, scale),
    )


def test_declarative_rewrite_condition_rejects_wrong_rank_or_shape():
    for matrix_value, scale_value in (
        (torch.randn(2, 4), torch.randn(3)),
        (torch.randn(2, 3, 4), torch.randn(4)),
        (torch.randn(2, 3), torch.randn(3, 3)),
    ):
        graph_module = symbolic_trace(_test_diagonal_matmul)
        placeholders = [
            node for node in graph_module.graph.nodes if node.op == "placeholder"
        ]
        placeholders[0].meta["example_value"] = matrix_value
        placeholders[1].meta["example_value"] = scale_value

        assert _test_rewrite().apply(graph_module) == 0
        assert "diag" in graph_module.code
        assert "matmul" in graph_module.code


def test_declarative_rewrite_pass_applies_rules_and_counts_invocations():
    graph_module = symbolic_trace(_test_diagonal_matmul)
    placeholders = [
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    ]
    placeholders[0].meta["example_value"] = torch.randn(2, 3)
    placeholders[1].meta["example_value"] = torch.randn(3)
    rewrite_pass = DeclarativeRewritePass(_test_rewrite())

    rewrite_pass(graph_module.graph)

    assert rewrite_pass.invocation_count == 1
    assert rewrite_pass.match_count == 1
    assert "diag" not in graph_module.code


def test_declarative_rewrite_pass_cache_tracks_behavior_and_rule_version():
    first = _TestRewritePass(_test_rewrite("first"))
    second = _TestRewritePass(_test_rewrite("second"))

    assert _source_file(type(first)) in first.behavior_files
    assert (
        _source_file(_test_rank_and_shape_condition) in first.rewrites[0].behavior_files
    )
    assert first.uuid() != second.uuid()
