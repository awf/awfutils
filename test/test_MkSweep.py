#from pytest import capfd
import re
import os
import shutil
from awfutils import MkSweep


def test_mksweep(capfd):
  shutil.rmtree("tmp/awfutils-sweep-test")
  with MkSweep("tmp/awfutils-sweep-test") as ms:
    for lr in (1e-4, 3e-4, 1e-3):
      for beta in set([0.99, 0.999, 1-lr]): 
        ms.add(f"echo python myrun.py --tvalue=7 --lr={lr} --beta={beta}")
  os.system("make -f tmp/awfutils-sweep-test/Makefile")
  out, err = capfd.readouterr()
  assert re.search(
    "sweep: Making directory tmp/awfutils-sweep-test/.* for echo python myrun.py --tvalue=7 --lr=0.0003 --beta=0.999",
    out 
  )

# shutil.rmtree("tmp/awfutils-sweep-test")
# with MkSweep("tmp/awfutils-sweep-test") as ms:
#   for lr in (1e-4, 3e-4, 1e-3):
#     for beta in [0.99, 0.999, 1-lr]: 
#       ms.add(f"echo python myrun.py --tvalue=7 --lr={lr} --beta={beta}")
# os.system("make -f tmp/awfutils-sweep-test/Makefile")
