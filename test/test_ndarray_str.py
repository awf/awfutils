import pytest
import numpy as np
import torch
from types import SimpleNamespace

from awfutils import ndarray_str


def np_mk(dtype, *sz):
    return np.arange(np.prod(sz), dtype=dtype).reshape(sz)


def np_mx(dtype, vals):
    return np.array(vals, dtype=dtype)


def np_zeros(dtype, shape):
    return np.zeros(shape, dtype=dtype)


import jax.numpy as jnp


def jax_mk(dtype, *sz):
    return jnp.arange(np.prod(sz), dtype=dtype).reshape(sz)


def jax_mx(dtype, vals):
    return jnp.array(vals, dtype=dtype)


def jax_zeros(dtype, shape):
    return jnp.zeros(shape, dtype=dtype)


numpy_to_torch_dtype_dict = {
    np.bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}


def torch_mk(dtype, *sz):
    return torch.arange(np.prod(sz), dtype=numpy_to_torch_dtype_dict[dtype]).reshape(sz)


def torch_mx(dtype, vals):
    return torch.tensor(vals)


def torch_zeros(dtype, shape):
    return torch.zeros(shape, dtype=numpy_to_torch_dtype_dict[dtype])


platforms = [
    pytest.param(
        SimpleNamespace(mx=np_mx, mk=np_mk, zeros=np_zeros),
        id="np",
    ),
    pytest.param(
        SimpleNamespace(mx=torch_mx, mk=torch_mk, zeros=torch_zeros),
        id="torch",
    ),
    pytest.param(
        SimpleNamespace(mx=jax_mx, mk=jax_mk, zeros=jax_zeros),
        id="jax",
    ),
]


def go(x, target):
    act = ndarray_str(x)
    print(act)
    assert act == target


@pytest.mark.parametrize("p", platforms)
def test_ndarray_str(p):
    mx, mk = p.mx, p.mk
    go(
        mx(np.float32, [[1, 2, np.inf], [np.inf, np.nan, 1.1]]),
        "f32[2x3] [[1.000 2.000 inf], [inf nan 1.100]]",
    )
    go(
        mk(np.float32, 2, 3, 4),
        "f32[2x3x4] Percentiles{0.000 1.000 6.000 11.000 17.000 22.000 23.000}",
    )
    go(
        mk(np.float32, 2, 3),
        "f32[2x3] [[0.000 1.000 2.000], [3.000 4.000 5.000]]",
    )

    go(
        mx(np.float32, [1, 2, 3, 4, 5, 5, 4, 3, 2, 1, np.inf, np.inf, np.nan]),
        "f32[13] Percentiles{1.000 1.000 2.000 3.000 4.000 5.000 5.000} #inf=2 #nan=1",
    )
    go(
        mx(np.float32, [0.0, 0.0, np.nan]),
        "f32[3] [0.0 0.0 nan]",
    )


@pytest.mark.parametrize("p", platforms)
def test_zeros(p):
    go(p.zeros(np.float32, []), "f32[] [0.0]")
    go(p.zeros(np.float32, 0), "f32[0] []")
    go(p.zeros(np.float32, 1), "f32[1] [0.0]")
    go(p.zeros(np.float32, 100), "f32[100] Zeros")
    go(p.zeros(np.float16, (100, 100, 101)), "f16[100x100x101] Zeros")


@pytest.mark.parametrize("p", platforms)
def test_ints(p):
    go(p.mk(np.int32, 2, 3), "i32[2x3] [[0 1 2], [3 4 5]]")
    go(p.mk(np.int32, 21, 31), "i32[21x31] Percentiles{0 32 162 325 488 618 650}")
