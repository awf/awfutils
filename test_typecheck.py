import pytest

from typecheck import typecheck
from functools import partial

from contextlib import nullcontext as does_not_raise

from icecream import ic

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
    try:
        import jax
    except:
        pytest.skip("No jax")

    import jax
    import jax.numpy as jnp

    @typecheck_show_src
    def foo1(x: jnp.ndarray, t: jnp.ndarray) -> float:
        y: jnp.ndarray = x * t
        z: jnp.ndarray = y / 2
        return z

    ic(isinstance(3, jnp.ndarray))

    float_array = jnp.ones((3, 5))

    with pytest.raises(TypeError, match="x not of type jnp.ndarray"):
        foo1(3, float_array)

    # Jitted, it will not raise, as the tracers are of type jnp.ndarray
    with does_not_raise():
        jax.jit(foo1)(3, float_array)


def test_typecheck_jaxtyping1():
    try:
        import jax
        import jaxtyping
    except:
        pytest.skip("No jax or jaxtyping")

    import jax
    from jaxtyping import f32, u, jaxtyped

    # int_t = jaxtyping.i[""] TODO

    rng = jax.random.PRNGKey(42)
    vec_f32 = jax.random.uniform(rng, (11,))

    @jax.jit
    @partial(typecheck, show_src=True, refers=(jaxtyping,))
    def foo1(x: jaxtyping.i[""], t: f32["N"]) -> f32["N"]:
        z: f32["N"] = x * t
        return z

    with does_not_raise():
        foo1(3, vec_f32)


def test_typecheck_jaxtyping2():
    try:
        import jax
        import jaxtyping
    except:
        pytest.skip("No jaxtyping")

    from jaxtyping import jaxtyped, f32

    rng = jax.random.PRNGKey(42)
    vec_f32 = jax.random.uniform(rng, (11,))

    # Raw jaxtyped - won't check the statement annotation
    @jaxtyped
    def standardize(x: jaxtyping.f32["N"], eps=1e-5) -> f32["N"]:
        m: float = x.mean()
        xc: f32["N N"] = x - m  # Wants to be NxN, won't be caught
        return xc / (x.std() + eps)

    with does_not_raise():
        t1 = standardize(vec_f32)

    # Typecheck with jaxtyping types - will raise
    @partial(typecheck, show_src=True)
    def standardize_tc(x: jaxtyping.f32["N"], eps=1e-5) -> f32["N"]:
        m: jaxtyping.f32[""] = x.mean()
        xc: f32["N N"] = x - m  # Wants to be NxN, won't be caught
        return xc / (x.std() + eps)

    with pytest.raises(TypeError, match=r"xc not of type f32\['N N'\]"):
        t1 = standardize_tc(vec_f32)

    # embeddings = jax.random.uniform(rng, (11,13))
    # t1 = standardize(embeddings)

    # embeddings = jax.random.uniform(rng, (11, 13))
    # t1 = jax.vmap(standardize)(embeddings)
