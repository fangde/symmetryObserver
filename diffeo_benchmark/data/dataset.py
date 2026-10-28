import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List
from diffeo_benchmark.diffeo.generator import DiffeoGenerator, DiffeoConfig
from diffeo_benchmark.diffeo.triangular import CompositeDiffeomorphism

class DiffeoEpisodeDataset(Dataset):
    """
    Dataset that provides episodes of diffeomorphic transformations.
    Each item is an episode:
      - x_ctx: (K, D)
      - y_ctx: (K, D)
      - x_qry: (M, D)
      - y_qry: (M, D)
      - episode_idx: int
    """
    def __init__(
        self,
        num_episodes: int,
        config: DiffeoConfig,
        num_context: int = 32,
        num_query: int = 256,
        noise_std: float = 0.0,
        domain_std: float = 1.0,
        seed: int = 42,
        cache_diffeos: bool = True
    ):
        super().__init__()
        self.num_episodes = num_episodes
        self.config = config
        self.num_context = num_context
        self.num_query = num_query
        self.noise_std = noise_std
        self.domain_std = domain_std
        self.seed = seed
        self.cache_diffeos = cache_diffeos
        
        # Pre-instantiate generator
        self.diffeo_generator = DiffeoGenerator(config, seed=seed)
        
        if cache_diffeos:
            # Deterministically pre-sample all diffeomorphisms for reproducible benchmarks
            self.diffeos: List[CompositeDiffeomorphism] = [
                self.diffeo_generator.sample_diffeomorphism() for _ in range(num_episodes)
            ]
        else:
            self.diffeos = []

    def get_diffeo(self, idx: int) -> CompositeDiffeomorphism:
        if self.cache_diffeos:
            return self.diffeos[idx]
        else:
            gen = DiffeoGenerator(self.config, seed=self.seed + idx * 7919)
            return gen.sample_diffeomorphism()

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, idx: int) -> dict:
        phi = self.get_diffeo(idx)
        
        # Deterministic generation per episode index if needed
        # We can seed with (self.seed + idx) for reproducible context and query points
        rng = torch.Generator()
        rng.manual_seed(self.seed + idx * 104729)
        
        dim = self.config.dim
        # Sample context points
        x_ctx = torch.randn(self.num_context, dim, generator=rng) * self.domain_std
        y_ctx = phi.forward(x_ctx)
        if self.noise_std > 0:
            y_ctx = y_ctx + torch.randn_like(y_ctx, generator=rng) * self.noise_std
            
        # Sample query points
        x_qry = torch.randn(self.num_query, dim, generator=rng) * self.domain_std
        y_qry = phi.forward(x_qry)
        
        return {
            "x_ctx": x_ctx,
            "y_ctx": y_ctx,
            "x_qry": x_qry,
            "y_qry": y_qry,
            "episode_idx": idx
        }

def collate_episodes(batch: List[dict]) -> dict:
    return {
        "x_ctx": torch.stack([b["x_ctx"] for b in batch], dim=0), # (B, K, D)
        "y_ctx": torch.stack([b["y_ctx"] for b in batch], dim=0), # (B, K, D)
        "x_qry": torch.stack([b["x_qry"] for b in batch], dim=0), # (B, M, D)
        "y_qry": torch.stack([b["y_qry"] for b in batch], dim=0), # (B, M, D)
        "episode_idx": torch.tensor([b["episode_idx"] for b in batch], dtype=torch.long)
    }
