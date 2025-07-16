from array_api_compat import array_namespace
import re


# Fixes for Array API
import array_api_compat


def _is_torch_array(xp, x):
    # String check for jax so we don't need to import it
    return xp.__name__ != "jax.numpy" and xp.is_torch_array(x)


def _is_floating(xp, x):
    if _is_torch_array(xp, x):
        return xp.is_floating_point(x)

    return xp.issubdtype(x.dtype, xp.floating)


def _quantile(values, quantiles):
    xp = array_namespace(values)

    if _is_torch_array(xp, values):
        dtype = values.dtype
        if dtype != xp.float32:
            values = xp.tensor(values, dtype=xp.float32)
        qs = xp.quantile(values, xp.tensor(quantiles), interpolation="nearest")
        return qs.to(dtype=dtype)
    else:
        return xp.quantile(values, xp.array(quantiles), interpolation="nearest").astype(
            values.dtype
        )


# End/Fixes for Array API


def ndarray_str(x, tiny: int = 10, large: int = 100_000_000):
    """
    Nicely print an ndarray on one line.

    Small tensors are printed in the format::

      f23[2x1x3] 10^6 x [[[0.065 1.741 0.013]], [[0.000 0.172 1.334]]]

    While larger tensors are printed as Percentiles:

      Tensor(22x11x33,f32)       10^-7 x Percentiles{0.002|0.493|2.470|4.958|7.490|9.434|9.996}
      ^type  ^shape   ^dtype     ^scale            0 (min)|   5%|  25%|  50%|  75%|  95%|100% (max)

    See the notebook <11-utils.ipynb> for more information.
    """

    if not hasattr(x, "__array__"):
        return repr(x)

    xp = array_namespace(x)

    shape_str = "x".join(map(str, x.shape))
    dtype_str = (
        f"{x.dtype}".replace("float", "f").replace("uint", "u").replace("int", "i")
    )
    dtype_str = re.sub(r"^torch\.", r"", dtype_str)
    type_str = f"{dtype_str}[{shape_str}]"

    notes = ""
    finite_vals = x[xp.isfinite(x)]
    all_finite = xp.size(finite_vals) == xp.size(x)
    display_all = xp.size(x) <= tiny

    def disp(a, fmt):
        if len(a.shape) > 1:
            return "[" + "], [".join(disp(e, fmt) for e in a) + "]"
        elif len(a.shape) > 0:
            return " ".join(fmt.format(v=v) for v in a)
        else:
            return fmt.format(v=a)

    if display_all:
        vals = x
        head, tail = "[", "]"
    else:
        if not all_finite:
            notes += f" #inf={xp.isinf(x).sum()} #nan={xp.isnan(x).sum()}"

        if xp.size(x) < large:
            quantiles = [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
            vals = _quantile(finite_vals, quantiles)
            head, tail = "Percentiles{", "}"
        else:
            # Too large to sort, just show min, median, max
            vals = xp.array(
                [finite_vals.min(), xp.median(finite_vals), finite_vals.max()],
                dtype=x.dtype,
            )
            head, tail = "MinMedMax{", "}"

    if _is_floating(xp, x):
        # scale down vals
        max = xp.abs(finite_vals).max() if xp.size(finite_vals) else 0
        if max > 0:
            logmax = xp.floor(xp.log10(max))
            if -2 <= logmax <= 3:
                logmax = 0
            max_scale = 10**-logmax
            max_scale_str = f"10^{int(logmax)} x " if logmax != 0 else ""
            vals_str = disp(vals * max_scale, "{v:.3f}")
            vals_str = max_scale_str + head + vals_str + tail
        else:
            max_scale = 1
            max_scale_str = ""
            if display_all:
                vals_str = head + disp(vals, "{v}") + tail
            else:
                vals_str = "Zeros"
    else:
        # Assume integer, print as integers
        vals_str = head + disp(vals, "{v:d}") + tail

    dtype_str = (
        f"{x.dtype}".replace("float", "f").replace("uint", "u").replace("int", "i")
    )

    return f"{type_str} {vals_str}{notes}"
