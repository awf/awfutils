import pytest

from typecheck import typecheck


def test_typecheck_1():
    @typecheck
    def foo(x: int, t: float) -> float:
        y: float = x * t
        assert isinstance(y, float), f"y a {type(y)} not a float"
        z: int = x // 2
        assert isinstance(z, int), "z not a int"
        return z * y

    # Use manual checks
    foo.__wrapped__(3, 4.2)

    # Ensure passes
    foo(3, 4.2)
    assert True, "foo did not raise, as expected"

    @typecheck
    def foo1(x: int, t: int) -> float:
        y: float = x * t  # Expect to fail here
        z: int = x // 2
        return z * y

    foo1.__wrapped__(3, 5)
    assert True, "foo1 did not raise, as expected"

    with pytest.raises(AssertionError):
        foo1(3, 5)


def test_typecheck_jax():
    try:
        import jax
    except:
        pytest.skip("No jax")

    import jax
    import jax.numpy as jnp

    @jax.jit
    @typecheck
    def foo1(x: int, t: jnp.ndarray) -> float:
        y: int = x * t  # Expect to raise here
        z: jnp.ndarray = y / 2
        return z

    float_array = jnp.ones((3, 5))

    with pytest.raises(AssertionError):
        foo1(3, float_array)


def test_typecheck_jaxtyping():
    try:
        import jax
        import jaxtyping
    except:
        pytest.skip("No jaxtyping")

    import jax
    from jaxtyping import f32, u, jaxtyped

    @jaxtyped
    @typecheck
    def standardize(x: f32["N"], eps=1e-5) -> f32["N"]:
        return (x - x.mean()) / (x.std() + eps)

    rng = jax.random.PRNGKey(42)

    embeddings = jax.random.uniform(rng, (11,))
    t1: f32["N M"] = standardize(embeddings)

    # embeddings = jax.random.uniform(rng, (11,13))
    # t1 = standardize(embeddings)

    embeddings = jax.random.uniform(rng, (11, 13))
    t1 = jax.vmap(standardize)(embeddings)
