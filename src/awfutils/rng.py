"""Persistent, hierarchically splittable PyTorch RNGs."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from hashlib import blake2b

import torch
from torch import Tensor


@dataclass(frozen=True)
class SplittableRNG:
    """A random key whose named and sequential children are independent streams."""

    seed: int
    _path: tuple[tuple[str, Hashable], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError(f"seed must be an int, got {type(self.seed).__name__}")

    def __getitem__(self, key: Hashable) -> SplittableRNG:
        return SplittableRNG(self.seed, (*self._path, ("key", key)))

    def split(self, count: int) -> tuple[SplittableRNG, ...]:
        return tuple(
            SplittableRNG(self.seed, (*self._path, ("split", index)))
            for index in range(count)
        )

    def torch(self) -> torch.Generator:
        material = repr((self.seed, self._path)).encode()
        seed = int.from_bytes(
            blake2b(material, digest_size=8, person=b"ft-rng-v1").digest(),
            byteorder="little",
        )
        return torch.Generator(device=torch.get_default_device()).manual_seed(seed)

    def randn(self, *shape: int) -> Tensor:
        return torch.randn(shape, generator=self.torch())

    def uniform(self, *shape: int, abs: float) -> Tensor:
        return torch.empty(shape).uniform_(-abs, abs, generator=self.torch())


def test_splittable_rng_has_stable_disjoint_child_namespaces():
    rng = SplittableRNG(42)

    assert rng["query"] == rng["query"]
    assert rng[("query", 1)] != rng["query"]
    assert rng[0] != rng.split(1)[0]
    assert rng.split(2) == rng.split(3)[:2]
    assert torch.equal(
        rng["query"].randn(8),
        rng["query"].randn(8),
    )
    assert not torch.equal(
        rng[0].uniform(8, abs=1),
        rng.split(1)[0].uniform(8, abs=1),
    )
