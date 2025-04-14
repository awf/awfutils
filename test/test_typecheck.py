from contextlib import nullcontext as does_not_raise
from functools import partial

import jax
import jax.numpy as jnp
import jaxtyping
import pytest
import torch
from beartype import beartype as typechecker
from jaxtyping import Array, jaxtyped
from torch import Tensor

from awfutils import typecheck

# Define type aliases - typecheck will use these
# as names, for much clearer error messages.
Int = jaxtyping.Int[Array, ""]
Float = jaxtyping.Float[Array, ""]
FloatN = jaxtyping.Float[Array, "N"]
FloatNxN = jaxtyping.Float[Array, "N N"]


def test_typecheck_1():
    @typecheck
    def foo(x: int, t: float) -> float:
        y: float = x * t
        assert isinstance(y, float), f"y : {type(y)} not of type float"
        z: int = x // 2
        assert isinstance(z, int), "z not of type int"
        return z * y

    # Use manual checks
    foo.__wrapped__(3, 4.2)

    # Ensure passes
    with does_not_raise():
        foo(3, 4.2)

    # Check that argument mismatches also raise
    with pytest.raises(TypeError, match="t not of type float"):
        foo(3, 5)

    @typecheck
    def foo1(x: int, t: int) -> float:
        y: float = x * t  # Expect to fail here
        z: int = x // 2
        return z * y

    with does_not_raise():
        foo1.__wrapped__(3, 5)

    with pytest.raises(TypeError, match="y not of type float"):
        foo1(3, 5)


def test_typecheck_callables():
    def is_float(v):
        return isinstance(v, float)

    @typecheck
    def foo(x: int, t: is_float) -> float:
        y: is_float = x * t
        assert isinstance(y, float), f"y : {type(y)} not of type float"
        z: int = x // 2
        assert isinstance(z, int), "z not of type int"
        return z * y

    # Use manual checks
    foo.__wrapped__(3, 4.2)

    # Ensure passes
    with does_not_raise():
        foo(3, 4.2)

    # Check that argument mismatches also raise
    with pytest.raises(TypeError, match="t does not satisfy is_float"):
        foo(3, 5)

    @typecheck
    def foo1(x: int, t: int) -> float:
        y: is_float = x * t  # Expect to fail here
        z: int = x // 2
        return z * y

    with does_not_raise():
        foo1.__wrapped__(3, 5)

    with pytest.raises(TypeError, match="y does not satisfy is_float"):
        foo1(3, 5)


def test_typecheck_torch():

    def is_square_tensor(x):
        return x.shape[0] == x.shape[1]

    @typecheck
    def foo(x: Tensor):
        z: is_square_tensor = x @ x.T  # check result is square
        return z

    foo(torch.ones(3, 4))

    def is_shape(*sh):
        # Return a function that checks the shape
        return lambda x: x.shape == sh

    is_shape(3, 4)(torch.ones(3, 4))

    @typecheck
    def floo(x: Tensor):
        L, D = x.shape  # Get shape of X
        LxD = is_shape(L, D)  # LxD(v) checks that v is LxD
        LxL = is_shape(L, L)  # LxL

        z: LxL = x @ x.T  # check result is square
        w: LxD = z @ x
        return w

    floo(torch.ones(3, 4))

    @typecheck
    def shouldfail(x: Tensor):
        L, D = x.shape  # Get shape of X
        LxD = is_shape(L, D)  # LxD(v) checks that v is LxD

        z: LxD = x @ x.T  # check result is square
        return z

    with pytest.raises(TypeError, match="z does not satisfy LxD"):
        # This should fail, as z is LxL
        shouldfail(torch.ones(3, 4))


z_in_global_scope = 9


def test_typecheck_scope():
    @typecheck
    def foo2(x: int, t: float = 4.2) -> float:
        return x * t * z_in_global_scope

    foo2.__wrapped__(3)

    with does_not_raise():
        foo2(3)

    z_in_outer_scope = 8

    @typecheck
    def foo2(x: int, t: float = 4.2) -> float:
        return x * t * z_in_outer_scope

    foo2.__wrapped__(3)

    with does_not_raise():
        foo2(3)


def test_typecheck_jax():
    @typecheck(show_src=True)
    def foo1(x: jnp.ndarray, t: jnp.ndarray) -> float:
        y: jnp.ndarray = x * t
        z: jnp.ndarray = y / 2
        return z

    print(f"{isinstance(3, jnp.ndarray)=}")

    float_array = jnp.ones((3, 5))

    with pytest.raises(TypeError, match="x not of type jnp.ndarray"):
        foo1(3, float_array)

    # Jitted, it will not raise, as the tracers are of type jnp.ndarray
    with does_not_raise():
        jax.jit(foo1)(3, float_array)


def test_typecheck_jaxtyping1():
    rng = jax.random.PRNGKey(42)
    vec_f32 = jax.random.uniform(rng, (11,))

    @jax.jit
    @typecheck(show_src=True)
    def foo1(x: Int, t: FloatN) -> FloatN:
        z: FloatN = x * t
        return z

    with does_not_raise():
        foo1(3, vec_f32)


def test_typecheck_jaxtyping2():
    rng = jax.random.PRNGKey(42)
    vec_f32 = jax.random.uniform(rng, (11,))

    # Raw jaxtyped - won't check the statement annotation
    @jaxtyped(typechecker=typechecker)
    def standardize(x: FloatN, eps=1e-5) -> FloatN:
        m: float = x.mean()
        xc: FloatNxN = x - m  # Wants to be NxN, won't be caught
        return xc / (x.std() + eps)

    with does_not_raise():
        t1 = standardize(vec_f32)

    # Typecheck with jaxtyping types - will raise
    @typecheck
    def standardize_tc(x: FloatN, eps=1e-5) -> FloatN:
        m: Float = x.mean()
        xc: FloatNxN = x - m  # Wants to be NxN, won't be caught
        return xc / (x.std() + eps)

    with pytest.raises(TypeError, match=r"xc not of type FloatNxN"):
        t1 = standardize_tc(vec_f32)
