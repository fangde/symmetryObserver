import torch
from dataclasses import dataclass
from typing import List, Optional
from diffeo_benchmark.diffeo.triangular import TriangularDiffeoLayer, CompositeDiffeomorphism

@dataclass
class DiffeoConfig:
    dim: int = 8
    num_layers: int = 4
    s_std: float = 0.3
    a_std: float = 0.4
    w_std: float = 0.5
    b_std: float = 0.5
    even_perm_only: bool = False

def sample_permutation(dim: int, even_only: bool = False, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    perm = torch.randperm(dim, generator=generator)
    if even_only:
        # Check parity of permutation
        inv_count = 0
        for i in range(dim):
            for j in range(i + 1, dim):
                if perm[i] > perm[j]:
                    inv_count += 1
        if inv_count % 2 != 0:
            # Swap first two elements to flip parity
            perm[0], perm[1] = perm[1], perm[0].clone()
    return perm

class DiffeoGenerator:
    """
    Generates random CompositeDiffeomorphism instances according to DiffeoConfig.
    """
    def __init__(self, config: DiffeoConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def sample_layer(self) -> TriangularDiffeoLayer:
        dim = self.config.dim
        s = torch.randn(dim, generator=self.rng) * self.config.s_std
        
        a = torch.randn(dim, dim, generator=self.rng) * self.config.a_std
        a = torch.tril(a, diagonal=-1)
        
        w = torch.randn(dim, dim, generator=self.rng) * self.config.w_std
        w = torch.tril(w, diagonal=-1)
        
        b = torch.randn(dim, dim, generator=self.rng) * self.config.b_std
        b = torch.tril(b, diagonal=-1)
        
        perm = sample_permutation(dim, even_only=self.config.even_perm_only, generator=self.rng)
        return TriangularDiffeoLayer(dim, s, a, w, b, perm)

    def sample_diffeomorphism(self) -> CompositeDiffeomorphism:
        layers = [self.sample_layer() for _ in range(self.config.num_layers)]
        return CompositeDiffeomorphism(self.config.dim, layers)
