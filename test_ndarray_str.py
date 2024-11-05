from ndarray_str import ndarray_str
import numpy as np


def mx(dtype, *sz):
    return np.arange(np.prod(sz), dtype=dtype).reshape(sz)


def go(x, target):
    act = ndarray_str(np.array(x))
    print(act)
    assert act == target


def test_ndarray_str():
    go([1, 2, np.inf, np.inf, np.nan], "f64[5] [1.000 2.000 inf inf nan]")
    go(
        mx(np.float64, 2, 3, 4),
        "f64[2x3x4] Percentiles{0.000 1.000 6.000 12.000 17.000 22.000 23.000}",
    )
    go(
        mx(np.float32, 2, 3),
        "f32[2x3] [0.000 1.000 2.000 3.000 4.000 5.000]",
    )

    go(
        [1, 2, 3, 4, 5, np.inf, np.inf, np.nan],
        "f64[8] Percentiles{1.000 1.000 2.000 3.000 4.000 5.000 5.000} #inf=2 #nan=1",
    )


def test_zeros():
    go(np.zeros([]), "f64[] Zeros")
    go(np.zeros(0), "f64[0] Zeros")
    go(np.zeros(1), "f64[1] Zeros")
    go(np.zeros(100), "f64[100] Zeros")
    go(np.zeros((100, 100, 101), dtype=np.float16), "f16[100x100x101] Zeros")


def test_ints():
    go(mx(np.int32, 2, 3), "i32[2x3] [0 1 2 3 4 5]")
    go(mx(np.int32, 20, 30), "i32[20x30] Percentiles{0 30 150 300 449 569 599}")
