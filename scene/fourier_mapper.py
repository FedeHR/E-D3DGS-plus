import torch
import torch.nn as nn


class SimpleFourierMapper(nn.Module):
    """
    Simple Fourier Feature Mapping for xyz coordinates.
    
    Enhances Gaussian embeddings with spatial high-frequency features.
    Maps input xyz coordinates to Fourier features that are concatenated
    with learned Gaussian embeddings, preserving the original architecture
    that uses only embeddings (not raw coordinates) as network input.
    
    γ(x) = [a * sin(2π B x), a * cos(2π B x)] / ||a||
    
    where B is a matrix of random frequencies sampled from N(0,1) and a are learnable 
    amplitude coefficients that control the magnitude of different frequencies.
    Based on "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
    """
    
    def __init__(self, input_dim, num_frequencies=4, use_amplitude_coefficients=True):
        """
        Initialize Fourier mapper.
        
        Args:
            input_dim (int): Dimension of input (e.g., 3 for xyz coordinates)
            num_frequencies (int): Number of frequency components
            use_amplitude_coefficients (bool): Whether to use learnable amplitude coefficients
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.use_amplitude_coefficients = use_amplitude_coefficients
        
        # Generate random frequency matrix B (fixed, not learnable)
        # As randnd samples from Gaussian(0,1), it follows the isotropic Gaussian recommended in the paper for no stron priors
        # TODO: could learn a better value for sigma or try the other variants in the paper
        B = torch.randn(num_frequencies, input_dim)
        self.register_buffer('B', B)
        
        # Learnable amplitude coefficients a (optional)
        if use_amplitude_coefficients:
            # Initialize with small positive values to avoid zero gradients
            a = torch.ones(num_frequencies) * 0.1
            self.a = nn.Parameter(a)
        else:
            # Register as buffer (non-learnable) with all ones
            self.register_buffer('a', torch.ones(num_frequencies))
        
        self.output_dim = 2 * num_frequencies
    
    def forward(self, x):
        """
        Apply Fourier mapping to input.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., input_dim)
            
        Returns:
            torch.Tensor: Fourier-mapped features of shape (..., output_dim)
        """
        frequencies = 2. * torch.pi * (x @ self.B.T)
        sin_features = self.a * torch.sin(frequencies)
        cos_features = self.a * torch.cos(frequencies)
        fourier_features = torch.cat([sin_features, cos_features], dim=-1)
        
        # Normalize by the norm of amplitude coefficients (if using them)
        if self.use_amplitude_coefficients:
            a_norm = torch.linalg.norm(self.a)
            fourier_features = fourier_features / a_norm
        
        return fourier_features
