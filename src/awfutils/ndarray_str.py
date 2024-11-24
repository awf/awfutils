import numpy as np


def ndarray_str(x):
    """
    Nicely print a tensor on one line.

    Small tensors are printed in the format::

      Tensor(2x1x3,f32) 10^6 x [0.065 1.741 0.013 0.000 0.172 1.334]

    While larger tensors are printed as Percentiles:

      Tensor(22x11x33,f32)       10^-7 x Percentiles{0.002|0.493|2.470|4.958|7.490|9.434|9.996}
      ^type  ^shape   ^dtype     ^scale            0 (min)|   5%|  25%|  50%|  75%|  95%|100% (max)

    See the notebook <11-utils.ipynb> for more information.
    """
    if not hasattr(x, "__array__"):
        return repr(x)

    shape_str = "x".join(map(str, x.shape))
    dtype_str = (
        f"{x.dtype}".replace("float", "f").replace("uint", "u").replace("int", "i")
    )
    type_str = f"{dtype_str}[{shape_str}]"

    notes = ""
    finite_vals = x[np.isfinite(x)]
    all_finite = finite_vals.size == x.size
    display_all = x.size <= 10

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
            notes += f" #inf={np.isinf(x).sum()} #nan={np.isnan(x).sum()}"

        if x.size < 1e6:
            quantiles = [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
            vals = np.quantile(finite_vals, quantiles, method="nearest")
            head, tail = "Percentiles{", "}"
        else:
            # Too large to sort, just show min, median, max
            vals = finite_vals.min(), np.median(finite_vals), finite_vals.max()
            head, tail = "MinMedMax{", "}"

    if np.issubdtype(x.dtype, np.floating):
        # scale down vals
        max = np.abs(finite_vals).max() if finite_vals.size else 0
        if max > 0:
            logmax = np.floor(np.log10(max))
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
