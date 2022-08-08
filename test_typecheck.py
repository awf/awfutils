import pytest

from typecheck import typecheck
from functools import partial

typecheck_show_src = partial(typecheck, show_src=True)


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
    foo(3, 4.2)
    assert True, "foo did not raise, as expected"

    # Check that argument mismatches also raise
    with pytest.raises(TypeError, match="t not of type float"):
        foo(3, 5)

    @typecheck
    def foo1(x: int, t: int) -> float:
        y: float = x * t  # Expect to fail here
        z: int = x // 2
        return z * y

    foo1.__wrapped__(3, 5)
    assert True, "foo1 did not raise, as expected"

    with pytest.raises(TypeError, match="y not of type float"):
        foo1(3, 5)


z_in_global_scope = 9


def test_typecheck_scope():
    @typecheck
    def foo2(x: int, t: float = 4.2) -> float:
        return x * t * z_in_global_scope

    foo2.__wrapped__(3)

    foo2(3)
    assert True, "foo2 did not raise, as expected"

    z_in_outer_scope = 8

    @typecheck
    def foo2(x: int, t: float = 4.2) -> float:
        return x * t * z_in_outer_scope

    foo2.__wrapped__(3)

    foo2(3)
    assert True, "foo1 did not raise, as expected"


def test_typecheck_jax():
    try:
        import jax
    except:
        pytest.skip("No jax")

    import jax
    import jax.numpy as jnp

    @jax.jit
    @typecheck_show_src
    def foo1(x: jnp.ndarray, t: jnp.ndarray) -> float:
        y: jnp.ndarray = x * t
        z: jnp.ndarray = y / 2
        return z

    float_array = jnp.ones((3, 5))

    foo1(3, float_array)
    assert True, "foo1 did not raise"


def test_typecheck_jaxtyping():
    try:
        import jax
        import jaxtyping
    except:
        pytest.skip("No jax or jaxtyping")

    import jax
    from jaxtyping import f32, u, jaxtyped

    @jaxtyped
    def standardize(x: jaxtyping.f32["N"], eps=1e-5) -> f32["N"]:
        m: float = x.mean()
        xc: f32["N"] = x - m
        return xc / (x.std() + eps)

    rng = jax.random.PRNGKey(42)

    embeddings = jax.random.uniform(rng, (11,))
    t1 = standardize(embeddings)

    # embeddings = jax.random.uniform(rng, (11,13))
    # t1 = standardize(embeddings)

    # embeddings = jax.random.uniform(rng, (11, 13))
    # t1 = jax.vmap(standardize)(embeddings)
