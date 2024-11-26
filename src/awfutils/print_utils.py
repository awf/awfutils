def fn_name(f):
    """
    Get the name of a function, including the module it is defined in.
    """
    n = f.__name__
    if hasattr(f, "__module__") and f.__module__ != "_operator":
        if f.__module__ == "torch._ops.aten":
            n = f"aten.{n}"
        else:
            n = f"{f.__module__}.{n}"

    # if hasattr(f, "__code__"):
    #     n += f"[{f.__code__.co_filename}:{f.__code__.co_firstlineno}]"

    return n


def class_name(c):
    """
    Get the name of a class, including the module it is defined in.
    """
    n = c.__name__
    if hasattr(c, "__module__") and c.__module__ != "_operator":
        n = f"{c.__module__}.{n}"

    return n
