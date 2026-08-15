"""PyTorch pytree helpers, including ``SimpleNamespace`` registration."""

from __future__ import annotations

import operator
from collections.abc import Iterator
from types import BuiltinFunctionType, FunctionType, SimpleNamespace

import torch
from prettyprinter import pformat
from torch import Tensor
from torch.utils import _pytree
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from .ndarray_str import ndarray_str
from .print_utils import class_name, fn_name


def _namespace_flatten(namespace: SimpleNamespace):
    names = list(vars(namespace))
    return [getattr(namespace, name) for name in names], names


def _namespace_unflatten(values, names: list[str]) -> SimpleNamespace:
    return SimpleNamespace(**dict(zip(names, values)))


def _namespace_flatten_with_keys(namespace: SimpleNamespace):
    values, names = _namespace_flatten(namespace)
    return [
        (_pytree.GetAttrKey(name), value) for name, value in zip(names, values)
    ], names


_pytree.register_pytree_node(
    SimpleNamespace,
    _namespace_flatten,
    _namespace_unflatten,
    serialized_type_name="types.SimpleNamespace",
    flatten_with_keys_fn=_namespace_flatten_with_keys,
)


def _path_name(path: tuple[_pytree.KeyEntry, ...]) -> str:
    parts = []
    for key in path:
        if (
            isinstance(key, _pytree.MappingKey)
            and isinstance(key.key, str)
            and key.key.isidentifier()
        ):
            parts.append(("." if parts else "") + key.key)
        else:
            parts.append(str(key))
    return "".join(parts).lstrip(".")


def _flatten_tensor_tree(tree: object):
    path_leaves, spec = _pytree.tree_flatten_with_path(tree)
    named_leaves = []
    for path, leaf in path_leaves:
        name = _path_name(path)
        if not isinstance(leaf, Tensor):
            raise TypeError(f"Leaf {name!r} is a {type(leaf).__name__}, not a Tensor")
        named_leaves.append((name, leaf))
    return named_leaves, spec


def named_tensor_leaves(tree: object) -> Iterator[tuple[str, Tensor]]:
    """Flatten a pytree, deriving leaf names from field and sequence paths."""
    yield from _flatten_tensor_tree(tree)[0]


def tensor_leaves(tree: object) -> Iterator[Tensor]:
    return (tensor for _, tensor in named_tensor_leaves(tree))


def flatten_tensor_tree(tree: object) -> tuple[list[tuple[str, Tensor]], str]:
    """Return named tensor leaves and a serialized specification for rebuilding them."""
    named_leaves, spec = _flatten_tensor_tree(tree)
    return named_leaves, _pytree.treespec_dumps(spec)


def unflatten_tensor_tree(serialized_spec: str, leaves: list[Tensor]) -> object:
    return _pytree.treespec_loads(serialized_spec).unflatten(leaves)


def test_named_tensor_tree_paths_and_serialization():
    tree = SimpleNamespace(
        vocabulary=torch.tensor([1.0]),
        layers=(SimpleNamespace(query=torch.tensor([[2.0]])),),
        projections={"output": torch.tensor([3.0])},
    )

    named_leaves, serialized_spec = flatten_tensor_tree(tree)

    assert [name for name, _ in named_leaves] == [
        "vocabulary",
        "layers[0].query",
        "projections.output",
    ]
    rebuilt = unflatten_tensor_tree(
        serialized_spec, [tensor for _, tensor in named_leaves]
    )
    assert isinstance(rebuilt, SimpleNamespace)
    assert isinstance(rebuilt.layers[0], SimpleNamespace)
    assert torch.equal(rebuilt.layers[0].query, tree.layers[0].query)


def test_named_tensor_tree_rejects_non_tensor_leaves():
    try:
        list(named_tensor_leaves(SimpleNamespace(weight=torch.tensor(1), label="bad")))
    except TypeError as error:
        assert str(error) == "Leaf 'label' is a str, not a Tensor"
    else:
        raise AssertionError("named_tensor_leaves accepted a non-Tensor leaf")


def _testing_vals():
    val = (torch.tensor(1.123), torch.randn(3, 5))
    A = pt_rand_like(val)
    return val, A


def pt_flatmap(f, *trees):
    flats, _specs = zip(*map(tree_flatten, trees))
    # Assert specs all the same
    results = [f(*args) for args in zip(*flats)]
    return results


def pt_map(f, *trees):
    flats, specs = zip(*map(tree_flatten, trees))
    # Assert specs all the same
    results = [f(*args) for args in zip(*flats)]
    return tree_unflatten(results, specs[0])


def pt_sum(tree):
    leaves, _spec = tree_flatten(tree)
    return sum(l.sum() for l in leaves)


def pt_rand_like(x):
    return tree_map(torch.rand_like, x)


def pt_dot(A, B):
    """
    Return the dot product <A,B> for arbitrary pytrees A,B, producing a single scalar.
    """
    dots = pt_flatmap(lambda a, b: (a * b).sum(), A, B)
    return sum(dots)


def pt_maxabs(tree):
    leaves, _spec = tree_flatten(tree)
    return max(l.abs().max() for l in leaves)


def test_maxabs():
    val = (torch.tensor(1.123), torch.tensor([3, -5.2, 4.2]))
    torch.testing.assert_close(pt_maxabs(val).item(), 5.2)
    val = (torch.tensor(6.123), torch.tensor([3, -5.2, 4.2]))
    torch.testing.assert_close(pt_maxabs(val).item(), 6.123)


def test_dot():
    val, A = _testing_vals()
    torch.testing.assert_close(pt_dot(val, A), val[0] * A[0] + (val[1] * A[1]).sum())


def pt_mul(A, B):
    return pt_map(operator.mul, A, B)


def test_mul():
    val, A = _testing_vals()
    torch.testing.assert_close(pt_mul(val, A), (val[0] * A[0], val[1] * A[1]))


def pt_add(A, B):
    return pt_map(operator.add, A, B)


def test_add():
    val, A = _testing_vals()
    torch.testing.assert_close(pt_add(val, A), (val[0] + A[0], val[1] + A[1]))


def pt_sub(A, B):
    return pt_map(operator.sub, A, B)


def test_sub():
    val, A = _testing_vals()
    torch.testing.assert_close(pt_sub(val, A), (val[0] - A[0], val[1] - A[1]))


import numpy as np


def assert_equal_or_close(a, b, atol=1e-6):
    if hasattr(a, "__array__"):
        np.testing.assert_allclose(a, b, atol=atol)
    else:
        np.testing.assert_equal(a, b)


def pt_assert_close(A, B, atol=1e-6, verbose=False):
    """
    Assert that pytrees A and B are close.
    """
    pt_map(lambda a, b: assert_equal_or_close(a, b, atol=atol), A, B)


class PyTree:
    def __init__(self, *args):
        vals = tuple(a.val if isinstance(a, PyTree) else a for a in args)
        if len(args) == 1:
            self.val = vals[0]
        else:
            self.val = vals

    def __add__(self, b):
        if isinstance(b, PyTree):
            return PyTree(pt_add(self.val, b.val))
        else:
            return PyTree(pt_add(self.val, b))

    __radd__ = __add__

    def __sub__(self, b):
        if isinstance(b, PyTree):
            return PyTree(pt_sub(self.val, b.val))
        else:
            return PyTree(pt_sub(self.val, b))

    def __mul__(self, b):
        if isinstance(b, PyTree):
            return PyTree(pt_mul(self.val, b.val))
        else:
            # Assume RHS is something that can multiply any tensor
            return PyTree.map(lambda x: x * b, self.val)

    __rmul__ = __mul__

    def sum(self):
        return PyTree(pt_sum(self.val))

    @classmethod
    def rand_like(cls, arg):
        return PyTree(pt_rand_like(PyTree(arg).val))

    @classmethod
    def map(cls, f, *args):
        return PyTree(pt_map(f, PyTree(*args).val))

    @classmethod
    def dot(cls, a, b):
        return pt_dot(PyTree(a).val, PyTree(b).val)

    @classmethod
    def assert_close(cls, A, B, verbose=False):
        pt_assert_close(PyTree(A).val, PyTree(B).val, atol=1e-6)


def test_PyTree():
    val, A = _testing_vals()

    pval = PyTree.rand_like(val)
    val = pval.val
    pa = PyTree(A)

    PyTree.assert_close(pval + pa, PyTree(val[0] + A[0], val[1] + A[1]), verbose=True)
    PyTree.assert_close(pval + pa, PyTree(val[0] + A[0], val[1] + A[1]))
    PyTree.assert_close(pval + pa, PyTree(val[0] + A[0], val[1] + A[1]))
    PyTree.assert_close(1.23 * pa, PyTree(1.23 * A[0], 1.23 * A[1]))
    PyTree.assert_close(pa * 1.23, 1.23 * pa)


def _strval(x):
    """
    Convert x to string value, one line

    Any default renders that would include non-repeatable
    information (e.g. object id) are overridden here.
    """
    if isinstance(x, FunctionType):
        return "function " + fn_name(x)

    if isinstance(x, BuiltinFunctionType):
        return "builtin " + x.__name__

    if isinstance(x, type):
        return "class " + class_name(x)

    if isinstance(x, tuple):
        # This will happen when we are printing a dict key, and the key is a tuple
        # Assume entries are small
        return "(" + ", ".join(map(_strval, x)) + ")"

    if hasattr(x, "__array__"):
        return "Tensor " + ndarray_str(x)

    s = pformat(x).replace("\n", "\\n")
    return s[:40]


def printlines(x, tag="", strval=_strval):
    """
    Print a PyTree, with tag prefix, and compactly printing tensors
    Assumes that dict keys are "simple" in the sense that they render well in 40 chars.
    This is a generator of lines.
    """
    if isinstance(x, tuple):
        l = len(x)
        for i in range(l):
            yield from printlines(x[i], tag=tag + f"[{i}]", strval=strval)
    elif isinstance(x, list):
        yield (tag + "[")
        for i, v in enumerate(x):
            yield from printlines(v, tag=f"{tag}[{i}]", strval=strval)
        yield (tag + "]")
    elif isinstance(x, dict):
        for k in x:
            yield from printlines(x[k], tag=tag + f"[{_strval(k)}]", strval=strval)
    elif isinstance(x, SimpleNamespace):
        for k, v in x.__dict__.items():
            yield from printlines(v, tag=tag + f".{k!s}", strval=strval)
    elif isinstance(x, torch.nn.Module):
        for k, v in x.named_parameters():
            yield from printlines(v, tag + f".{k}", strval=strval)
    else:
        yield (tag + " = " + strval(x))


def pt_print_aux(x, tag="", printer=print, strval=_strval):
    """
    Print a PyTree, with tag prefix, and compactly printing tensors
    Assumes that dict keys are "simple" in the sense that they render well in 40 chars.
    """
    for line in printlines(x, tag=tag, strval=strval):
        printer(line)


def pt_print(*xs):
    for x in xs:
        pt_print_aux(x, tag="x:")


def test_PyTree_with_non_floats():
    val = (
        "tuple",
        [1, (torch.rand(2, 3), [1, 2, 3, 4], torch.tensor(3.4), torch.rand(12, 13)), 3],
        "fred",
    )

    # Just a crash check TODO: inspect output
    pt_print(val)

    # print(cpprint(pt_map(ndarray_str, val)))

    # # val is a tuple(list[int, tuple(array, str, array), int], str)
    # lens = somap(typestr, val)
    # assert str(lens) == "(['int', ('ndarray', 'str', 'ndarray'), 'int'], 'str')"

    # reduced = soreduce(lambda acc, val: acc + ":" + typestr(val), "S", val)
    # assert reduced == "S:int:ndarray:str:ndarray:int:str"

    # eqs = somap(np.array_equal, val, val)
    # assert soall(eqs)
