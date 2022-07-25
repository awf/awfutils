# awfutils: AWF's Utilities

A few utils for general python programming

## Usage

This is very much code-in-progress.  When I use it, I typically just put it as a submodule under whatever else I'm building, so I can easily bugfix `awfutils` as I do other work:

```sh
$ git submodule add https://github.com/awf/awfutils
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

Parsing happens the first time any of the Args is read, and is then cached.  If this is before all Args have been declared, then the next call will miss the cache, and re-parse.  This is annoying when you want to act on an arg during module load time, e.g.:
```python
jit_foo = Arg("jit-foo", False, "Run JIT on foo")
if jit_foo():
    foo = jit(foo)
```
If the user has passed the `-help` argument to get help, this call will return only the entries for the args that have been added so far, probably very few.
For such situations, you can skip the cache, and just peek into `sys.argv` directly using `peek`:
```python
jit_foo = Arg("jit-foo", False, "Run JIT on foo")
if jit_foo.peek(): # Just check for this arg in sys.argv
    foo = jit(foo)
```
