import functools

from awfutils import typecheck


def foo(x: int, y: float):
    z: int = x * y  # This should error, but doesn't
    w: float = z * 3.2
    return w


foo(3, 1.3)


@typecheck(show_src=True)
def foo(x: int, y: float):
    z: int = x * y  # Now it raises AssertionError: z not int
    w: float = z * 3.2
    return w


foo(3, 1.3)  # Error comes from this call
