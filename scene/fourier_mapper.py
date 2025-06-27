import torch
import numpy as np
from torch import nn

class SimpleFourierMapper(nn.Module):
    """
    Simple Fourier feature mapping for Gaussian embeddings.
    Maps input embeddings to higher dimensional space using sinusoidal functions.
    
    This follows the Random Fourier Features approach where each input dimension
    gets mapped with multiple random frequencies independently.
    """
    def __init__(self, input_dim, num_frequencies=4, scale=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.scale = scale
        
        # Create frequency matrix for each input dimension
        # Shape: (input_dim, num_frequencies)
        # Each column represents frequencies for one input dimension
        frequencies = torch.randn(input_dim, num_frequencies) * scale
        self.register_buffer('frequencies', frequencies)
        
        # Output dimension: input_dim + 2 * input_dim * num_frequencies
        # (original + sin + cos for each input-frequency combination)
        self.output_dim = input_dim + 2 * input_dim * num_frequencies
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (..., input_dim)
        Returns:
            Fourier mapped features of shape (..., output_dim)
        """
        # x shape: (..., input_dim)
        # frequencies shape: (input_dim, num_frequencies)
        
        # Compute x * frequencies for each input dimension
        # x.unsqueeze(-1): (..., input_dim, 1)
        # frequencies: (input_dim, num_frequencies)
        # Result: (..., input_dim, num_frequencies)
        x_expanded = x.unsqueeze(-1)  # (..., input_dim, 1)
        projections = x_expanded * self.frequencies  # (..., input_dim, num_frequencies)
        
        # Apply sin and cos
        sin_features = torch.sin(2 * np.pi * projections)  # (..., input_dim, num_frequencies)
        cos_features = torch.cos(2 * np.pi * projections)  # (..., input_dim, num_frequencies)
        
        # Flatten the last two dimensions
        sin_flat = sin_features.reshape(*x.shape[:-1], -1)  # (..., input_dim * num_frequencies)
        cos_flat = cos_features.reshape(*x.shape[:-1], -1)  # (..., input_dim * num_frequencies)
        
        # Concatenate original features with Fourier features
        fourier_features = torch.cat([x, sin_flat, cos_flat], dim=-1)
        
        return fourier_features

