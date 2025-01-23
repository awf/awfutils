from awfutils import MkSweep

with MkSweep("tmp/awfutils-sweep-test") as ms:
  for lr in (1e-4, 3e-4, 1e-3):
    for beta in set([0.99, 0.999, 1-lr]): 
      ms.add(f"python myrun.py --tvalue=7 --lr={lr} --beta={beta}")

with MkSweep("tmp/awfutils-sweep-test") as ms:
  for lr in (1e-4, 3e-4, 1e-3):
    for beta in [0.99, 0.999, 1-lr]: 
      ms.add(f"python myrun.py --tvalue=7 --lr={lr} --beta={beta}")
