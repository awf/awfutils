# AWFUTILS: AWF's Utilities

A few utils for general python programming

## Usage

This is very much code-in-progress.  When I use it, I typically just put it as a submodule under whatever else I'm building, so I can easily bugfix `awfutils` as I do other work:
```sh
$ git submodule add https://github.com/awf/awfutils
```

# Typecheck

I love run time type checkers, [particularly with JAX](https://github.com/google/jaxtyping/blob/main/FAQ.md#what-about-support-for-static-type-checkers-like-mypy-pyright-etc) but by default they (OK, beartype doesn't) don't check statement-level annotations like these:
```python
def foo(x : int, y : float):
  z : int = x * y # This should error, but doesn't
  w : float = z * 3.2
  return w

foo(3, 1.3)
```

With the awfutils `typecheck` decorator, they can...
```python
@typecheck
def foo(x : int, y : float):
  z : int = x * y # Now it raises TypeError: z not of type int
  w : float = z * 3.2
  return w

foo(3, 1.3) # Error comes from this call
```

This works by AST transformation, replacing the function foo above
with the function
```python
def foo_typecheck_wrap(x: int, y: float):
    assert isinstance(x, int), 'x not of type int'
    assert isinstance(y, float), 'y not of type float'
    z: int = x * y
    assert isinstance(z, int), 'z not of type int'
    w: float = z * 3.2
    assert isinstance(w, float), 'w not of type float'
    return w
```
Because it's AST transformation it is basically literally the above code, 
which you can see with the optional argument show_src=True
```python
@functools.partial(typecheck, show_src=True)
def foo(x : int, y : float):
  z : int = x * y # Now it does
  w : float = z * 3.2
  return w
```

# Arg

A distributed argument parser, like absl flags, but a little more convenient and less stringy.
If you want a config value anywhere in your program, just declare an Arg nearby (at top level), and use it:

```python
# my-random-file.py
from jaxutils.Arg import Arg

tau = Arg("mqh-tau", 0.01, "Scale factor for softmax in my_quick_hack")

def my_quick_hack(xs, qs):
    ...
    softmax(tau() * xs @ qs.T)
    ...
```

Now, even if  `my_quick_hack` is far down the call tree from main, you can quickly try some values of `tau` by just running
```sh
$ python main.py -tau 0.0001
```

More conventionally, `Arg` is also useful in `main`:
```python
def main():
    param1 = Arg("p1", default=34, help="Set first parameter")
    switch = Arg("s", False, "Turn on the switch")

    if switch(): # This is where arg parsing first happens. See notes.
        foo(param1())

    # To see which args have been set:
    print(Arg.config())
```

It's all a thin wrapper around `argparse`, so inherits a bunch of goodness from there, but avoids a lot of long-chain plumbing when `my_quick_hack` is far down the call tree from main.

And yes, these are global variables. This is absolutely reasonable, because the command line is a global resource.  You'll see that a little bit of namespacing has been illustrated above, where `tau` was given the flag `mqh-tau`.  Feel free to formalize that as much as you like.

Parsing happens the first time any of the Args is read, and is then cached.

There is a potential gotcha if you want to act on an arg during module load time, e.g. at the top level:
```python
jit_foo = Arg("jit-foo", False, "Run JIT on foo")
if jit_foo():   # Prefer jit_foo.peek() for load-time checks
    foo = jit(foo)
```
The call to `jit_foo()` will know only about arguments declared before that 
point, so a call to `--help` will produce too short a list.  
This is remedied later but is better avoided:
```python
jit_foo = Arg("jit-foo", False, "Run JIT on foo")
if jit_foo.peek(): # Just check for this arg in sys.argv
    foo = jit(foo)
```
