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
Because it _is_ AST transformation, it is basically literally the above code,
which you can see with the optional argument show_src=True
```python
@functools.partial(typecheck, show_src=True)
def foo(x : int, y : float):
  z : int = x * y # Now it does
  w : float = z * 3.2
  return w
```

# Tensor pretty-printing

It's handy to print a one-line summary of the contents of a tensor.

```python
a = np.random.rand(2, 1, 3) - 0.2
print(au.ndarray_str(a**6 * 1e7))
```
outputs
```
f64[2x1x3] 10^5 x [0.181 5.555 1.721 1.462 0.001 0.000]
```
Easy to read, even with only 3 significant figures (see the leading ``10^5x``).

For larger tensors, show percentiles:
```
f32[22x11x33] 10^-7 x Percentiles{0.002|0.493|2.470|4.958|7.490|9.434|9.996}
              ^scale            0 (min)|   5%|  25%|  50%|  75%|  95%|100% (max)
```

# MkSweep:  Simple sweeps via makefile

How do you specify a sweep?  In one sense, it's easy - you generally just want to
run a series of commands, with different command-line arguments:
```sh
python myrun.py --lr=0.00003
python myrun.py --lr=0.0001
python myrun.py --lr=0.0003
python myrun.py --lr=0.001
```
A few properties we might like are:

  * Interruptable: if a job fails, or if a machine fails, we can easily
    resume the sweep without re-running already-finished jobs.
    Similarly, if we change the sweep definition slightly, we don't want to rerun jobs
    that were in previous sweeps.

  * Flexibile: we can define complex combinations of configurations, 
    rather than simple grids.

  * Parallel: we can easily run jobs in parallel up to the resource limits 
    of available hardware

  * Portable: we can easily set up a sweep without installing a lot of infrastructure

These properties are reminiscent of those one might want in a large software build 
system, so `MkSweep` simply puts the series of commands into a classic `Makefile`,
which can be called in order to run them.  Each command is given an output directory 
which is a hash of its command line, so that re-running the same command will re-use the outputs.  The output's "done" marker is not updated until the command is successfully completed, so an interrupted command will always re-run until it completes successfully.

To specify which commands to run, we use a python script rather than any sort of YAML 
file, for reasons which, if not apparent immediately, should become reasonable when 
we see more complex examples.  For the above simple learning rate sweep, the definition is:
```py
from awfutils import MkSweep
with MkSweep("mytmp") as ms:
  for lr in (0.00003, 0.0001, 0.0003, 0.001):
      ms.add(f"python myrun.py --lr={lr}")
```
which, when run, creates a `Makefile` in folder `mytmp` which contains target definitions like
```makefile
mytmp/926fa3d0/done.txt: # if done.txt doesn't exist
	python myrun.py --lr=0.00003 >& mytmp/926fa3d0/log.txt
	touch $@ # Create the "done" file
```
so that
```sh
make -f mytmp/Makefile
```
will execute any undone commands, leaving outputs and logs in the subfolders of `mytmp`.

If we edit the sweep definition to include an extra `lr`, and a baseline run:
```py
from awfutils import MkSweep
with MkSweep("mytmp") as ms:
	ms.add(f"python myrun.py --no-lr") # "No LR" is some alternative option
  for lr in (0.00003, 0.0001, 0.0003, 0.001, 0.003):
      ms.add(f"python myrun.py --lr={lr}")
```
then re-running `make -f mytmp/Makefile` will just run the parts that have not been marked as done, which in this case would mean the new command for lr=0.003, and the new "No LR" command.

If we want to re-run everything (for example the code changed), then we can just remove the `mytmp` folders, or just make a new sweep folder e.g. `sweeps/run2`

### Why python, not YAML?
I have some problem with hyperparameters "alpha" and "beta", and I'm testing the idea 
that you should set beta to 1-alpha, while existing work sets it either to .99 or .999.

With a traditional sweeping infrastructure, configured via YAML files, it would be hard 
to encode the special rule that you don't need to run 1-alpha if it equals beta.
In python you can just write
```py
	for alpha in [1e-4, 3e-4, 1e-3]:
		for beta in set([0.99, 0.999, 1-alpha]): # deduplicate betas
			ms.add(f"python myrun.py --alpha={alpha} --beta={beta}")
```
Now in this case, the command hashing would not have run both commands anyway, 
but other constraints, e.g. $\alpha \le \beta \le \alpha^2$ are easily handled because the specification is all in Python.

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
$ python main.py -mqh-tau 0.0001
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


# PyTree Utils

Various smoothers for `torch.utils._pytree`.

Given a nest of lists and tuples, perform various useful funcions.

For example, given the object `val` as follows:
```py
val = ( # tuple
        [ # list
          1,
          (np.random.rand(2, 3), "b", np.random.rand(12, 13)),
          3
        ],
        "fred"
      )
```
Then `PyTree.map(foo, val)` will make these six calls to `foo`:
```py
foo(1),
foo(np.random.rand(2, 3))
foo("b")
foo(np.random.rand(12, 13))
foo(3)
foo("fred")
```

And given a numeric-only pytree, e.g.
```py
val = ( # tuple
        [ # list
          1,
          (np.random.rand(2, 3), np.random.rand(12, 13))
        ]
      )
```
Then arithmetic can be performed on `PyTree`s, e.g.
```
(PyTree(val) + val) * val == 2 * PyTree(val) * val
```
