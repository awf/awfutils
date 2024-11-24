import numpy as np

from awfutils import ndarray_str


def mx(dtype, *sz):
    return np.arange(np.prod(sz), dtype=dtype).reshape(sz)


def go(x, target):
    act = ndarray_str(np.array(x))
    print(act)
    assert act == target


def test_ndarray_str():
    go(
        [[1, 2, np.inf], [np.inf, np.nan, 1.1]],
        "f64[2x3] [[1.000 2.000 inf], [inf nan 1.100]]",
    )
    go(
        mx(np.float64, 2, 3, 4),
        "f64[2x3x4] Percentiles{0.000 1.000 6.000 12.000 17.000 22.000 23.000}",
    )
    go(
        mx(np.float32, 2, 3),
        "f32[2x3] [[0.000 1.000 2.000], [3.000 4.000 5.000]]",
    )

    go(
        [1, 2, 3, 4, 5, 5, 4, 3, 2, 1, np.inf, np.inf, np.nan],
        "f64[13] Percentiles{1.000 1.000 2.000 3.000 4.000 5.000 5.000} #inf=2 #nan=1",
    )
    go([0.0, 0.0, np.nan], "f64[3] [0.0 0.0 nan]")


def test_zeros():
    go(np.zeros([]), "f64[] [0.0]")
    go(np.zeros(0), "f64[0] []")
    go(np.zeros(1), "f64[1] [0.0]")
    go(np.zeros(100), "f64[100] Zeros")
    go(np.zeros((100, 100, 101), dtype=np.float16), "f16[100x100x101] Zeros")


def test_ints():
    go(mx(np.int32, 2, 3), "i32[2x3] [[0 1 2], [3 4 5]]")
    go(mx(np.int32, 20, 30), "i32[20x30] Percentiles{0 30 150 300 449 569 599}")
