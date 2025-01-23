import hashlib
from pathlib import Path


def dicttoflags(d):
    return " ".join([f"--{k}={v}" for k, v in d.items()])



import contextlib


class MkSweep(contextlib.AbstractContextManager):
    """ 
    Simple sweeps via makefile

    ```
      from awfutils import MkSweep
      with MkSweep("mytmp/sweeplogs") as ms:
        ms.add("python myrun.py --tvalue=7")
        for lr in (1e-4, 3e-4, 1e-3):
          for beta in set([0.99, 0.999, 1-lr])): 
            ms.add(f"python myrun.py --tvalue=7 --lr={lr} --beta={beta}")
    ```

    Will produce a makefile which contains the commands
    ```
      python myrun.py --tvalue=7
      python myrun.py --tvalue=7 --lr=1e-4 --beta=0.99
      python myrun.py --tvalue=7 --lr=1e-4 --beta=0.999
      python myrun.py --tvalue=7 --lr=1e-4 --beta=0.9999
      python myrun.py --tvalue=7 --lr=3e-4 --beta=0.99
      python myrun.py --tvalue=7 --lr=3e-4 --beta=0.999
      python myrun.py --tvalue=7 --lr=3e-4 --beta=0.9997
      python myrun.py --tvalue=7 --lr=1e-4 --beta=0.99
      python myrun.py --tvalue=7 --lr=1e-3 --beta=0.999
    ```
    where each command's output goes to a folder in `mytmp\sweeplogs` which is a
    hash of the command.

    Running the makefile with
    ```
      make -f mytmp/sweeplogs/Makefile
    ```
    will run any of the commands which have not yet run to completion.
    This means that a sweep that is interrupted can easily be restarted,
    running only the jubs that have not completed.

    If your jobs are small, you can run several in parallel using make's "-j" option:
    ```
      make -f mytmp/sweeplogs/Makefile -j3 # Run at most 3 jobs at a time 
    ```
    """

    def __init__(self, sweepdir: str | Path):
        self.sweepdir = Path(sweepdir)
        print("sweep: Making sweepdir", self.sweepdir)
        self.sweepdir.mkdir(exist_ok=True, parents=True)
        self.makefile_path = Path(sweepdir) / "Makefile"

    def __enter__(self):
        print("sweep: Making makefile", self.makefile_path)
        self.makefile = open(self.makefile_path, "w")
        self.print("all0: all")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.print("all: $(ALL)")
        self.makefile.close()
        print("sweep: Made makefile", self.makefile_path)

    def add(self, cmd, defaults = {}, **kwargs):
        def tomk(*args):
            self.print(*args, sep="")

        cmd += dicttoflags({**defaults, **kwargs})

        hash = hashlib.sha256(cmd.encode()).hexdigest()[:16]
        dir = Path(self.sweepdir) / hash
        if dir.exists():
            print("sweep: Will re-use existing", dir, "for", cmd)
        else:
            print("sweep: Making directory", dir, "for", cmd)
            dir.mkdir(exist_ok=True, parents=True)

        target = dir / "done.txt"
        tmplog = dir / "stdout-running.txt"
        cmdtxt = dir / "cmd.txt"

        # Print command to dir/cmd.txt
        with open(cmdtxt, "w") as f:
            print(cmd, file=f)

        # Create Makefile entry of the form
        #   ALL += hash/done.txt
        #   hash/done.txt:
        #        cat hash/cmd.txt > hash/stdout-running.txt
        #        python train.py ... >> hash/stdout-running.txt
        #        mv hash/stdout-running.txt $@
        #
        tomk("ALL += ", target)
        tomk(f"{target}:")
        tomk(f"\tcat {cmdtxt} > {tmplog}")
        tomk(f"\t{cmd} >> {tmplog}")
        tomk(f"\tmv {tmplog} $@")
        tomk("")



    def print(self, *args, **kwargs):
        print(*args, **kwargs, file=self.makefile)


