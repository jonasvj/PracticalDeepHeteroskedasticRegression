import math
import torch
import torch.nn as nn


class SinusoidalRBFLayer(nn.Module):
    """
    Sinusoidal Radial Basis Function.
    """
    def __init__(self, num_basis: int = 20, cutoff_dist: float = 5.0) -> None:
        """
        Args:
            num_basis: Number of radial basis functions to use.
            cutoff_dist: Euclidean distance threshold for determining whether 
                two nodes (atoms) are neighbours.
        """
        super().__init__()
        self.num_basis = num_basis
        self.cutoff_dist = cutoff_dist     

        self.register_buffer(
            'freqs',
            math.pi * torch.arange(1, self.num_basis + 1) / self.cutoff_dist
        )


    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Computes sinusoidal radial basis functions for a tensor of distances.

        Args:
            distances: torch.Tensor of distances (any size).
        
        Returns:
            A torch.Tensor of radial basis functions with size [*, num_basis]
                where * is the size of the input (the distances).
        """
        distances = distances.unsqueeze(-1)
        return torch.sin(self.freqs * distances) / distances


class CosineCutoff(nn.Module):
    """
    Cosine cutoff function.
    """
    def __init__(self, cutoff_dist: float = 5.0) -> None:
        """
        Args:
            cutoff_dist: Euclidean distance threshold for determining whether 
                two nodes (atoms) are neighbours.
        """
        super().__init__()
        self.cutoff_dist = cutoff_dist


    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Applies cosine cutoff function to input.

        Args:
            distances: torch.Tensor of distances (any size).
        
        Returns:
            torch.Tensor of distances that has been cut with the cosine cutoff
            function.
        """
        return torch.where(
            distances < self.cutoff_dist,
            0.5 * (torch.cos(distances * math.pi / self.cutoff_dist) + 1.),
            0.
        )