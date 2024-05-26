import numpy as np
import torch

from typing import list

class GaussianNoise:

    def __init__(self, sigma: float):
        """Create noise distribution.

        Args:
            sigma (float): Noise scaling.
        """
        self.sigma = sigma

    def sample(self, shape: list[int]) -> torch.FloatTensor:
        """Sample from noise distribution.

        Args:
            shape (list[int]): Shape of samples.

        Returns:
            torch.FloatTensor: Noise samples.
        """
        return self.sigma * torch.randn(*shape)

    def kernel(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Apply kernel from noise distribution.

        Args:
            x (torch.FloatTensor): Input tensor to apply kernel to.

        Returns:
            torch.FloatTensor: Resulting tensor.
        """
        return (
            1
            / (self.sigma * np.sqrt(2 * np.pi))
            * torch.exp(-torch.square(x) / (2 * self.sigma**2))
        )


class UniformNoise:

    def __init__(self, sigma):
        """Create noise distribution.

        Args:
            sigma (float): Noise scaling.
        """
        self.sigma = sigma

    def sample(self, shape):
        """Sample from noise distribution.

        Args:
            shape (list[int]): Shape of samples.

        Returns:
            torch.FloatTensor: Noise samples.
        """
        return self.sigma * (torch.rand(*shape) - 1 / 2)

    def kernel(self, x):
        """Apply kernel from noise distribution.

        Args:
            x (torch.FloatTensor): Input tensor to apply kernel to.

        Returns:
            torch.FloatTensor: Resulting tensor.
        """
        return 1 / (2 * self.sigma) * (torch.abs(x / self.sigma) <= 1)
