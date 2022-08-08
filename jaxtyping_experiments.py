import inspect
from prettyprinter import pprint

from functools import partial

import jax
import jax.numpy as jnp

import jaxtyping
from jaxtyping import f32, jaxtyped

int_t = jaxtyping.i[""]

from typecheck import typecheck

typecheck_show_src = partial(typecheck, show_src=True)


rng = jax.random.PRNGKey(42)
# Don't even bother threading rng
def rand(*args):
    return jax.random.uniform(rng, args)


vec_f32 = rand(11)
vec_i32 = jnp.arange(13)
mat_f32 = rand(7, 11)

###### Raw jaxtyped


@jax.jit
@jaxtyped
def foo1(x: int_t, t: f32["N"]):
    y = x + t
    assert isinstance(y, f32["N"]), f"y is a {y} of {type(y)}!"


foo1(3, vec_f32)


###### awfutils.typecheck, using jaxtyping isinstance


@jax.jit
@typecheck
def foo2(x: int_t, t: f32["N"]) -> f32["N"]:
    z: f32["N"] = x * t
    return z


foo2.__wrapped__.__wrapped__(3, vec_f32)
print("Wrapped worked fine, ignoring types")

foo2(3, vec_f32)
print("GOOD: foo2 worked fine, and should have")

try:
    foo2(3, 3.3)
    print("FAIL: foo2 worked fine, but should not have")
except TypeError as e:
    print("GOOD", e)

try:
    foo2(3.3, vec_f32)
    print("FAIL: foo2 worked fine, but should not have")
except TypeError as e:
    print("GOOD", e)


###### Now both together


@jax.jit
@jaxtyped
@typecheck
def foo3(A: f32["M N"], x: f32["N"], b: f32["M"]):
    y: f32["M"] = A @ x + b
    return y


foo3(rand(7, 11), rand(11), rand(7))
print("GOOD: foo3 worked fine, and should have")


@jax.jit
@jaxtyped
@typecheck_show_src
def foo4(A: f32["M N"], x: f32["N"], b: f32["M"]):
    y: f32["N"] = A @ x + b  # should fail here if M != N
    return y


foo4(rand(7, 7), rand(7), rand(7))
print("GOOD: foo3 worked fine, and should have")

try:
    foo4(rand(7, 11), rand(11), rand(7))
    print("FAIL: foo4 worked fine, but should not have")
except TypeError as e:
    print("GOOD", e)

exit(0)

# TODO : return type


def f_nested_imports():
    from jaxtyping import i, f32
    import jax.numpy as jnp

    z_in_outer_scope = 9.9

    @jax.jit
    @partial(typecheck, show_src=True, refers=(i, f32))
    def foo2(x: i, t: f32["N"]) -> float:
        z: int = x * t * z_in_outer_scope
        return z

    foo2.__wrapped__(3, vec_f32)
    print("Wrapped worked fine")

    foo2(3, vec_f32)


# f_nested_imports()


@jaxtyped
@typecheck
def standardize(x: f32["N"], eps=1e-5) -> f32["N"]:
    return (x - x.mean()) / (x.std() + eps)


t1: f32["N M"] = standardize(vec_f32)

# embeddings = jax.random.uniform(rng, (11,13))
# t1 = standardize(embeddings)

embeddings = jax.random.uniform(rng, (11, 13))
t1 = jax.vmap(standardize)(vec_f32)
