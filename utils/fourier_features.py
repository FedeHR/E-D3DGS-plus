import torch
import numpy as np

class FourierFeatureGenerator:
    """
    Fourier Feature Generator based on:
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
    https://proceedings.neurips.cc/paper/2020/file/55053683268957697aa39fba6f231c68-Paper.pdf
    
    Transforms input coordinates v into Fourier features:
    γ(v) = [cos(2πb₁ᵀv), sin(2πb₁ᵀv), ..., cos(2πbₘᵀv), sin(2πbₘᵀv)]ᵀ
    """
    
    def __init__(self, input_dim=3, mapping_size=128, scale=1.0, device='cuda'):
        """
        Args:
            input_dim (int): Dimension of input coordinates (3 for xyz)
            mapping_size (int): Number of random features to generate
            scale (float): Standard deviation of the Gaussian distribution for sampling b vectors
            device (str): Device to store tensors on
        """
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.scale = scale
        self.device = device
        
        # Generate random frequency vectors b_j ~ N(0, scale²I)
        # Shape: [mapping_size, input_dim]
        self.B = torch.randn(mapping_size, input_dim, device=device) * scale
        
    def encode(self, coords):
        """
        Apply Fourier feature mapping to input coordinates
        
        Args:
            coords (torch.Tensor): Input coordinates of shape [N, input_dim]
            
        Returns:
            torch.Tensor: Fourier features of shape [N, 2 * mapping_size]
        """
        # coords: [N, input_dim], B: [mapping_size, input_dim]
        # proj: [N, mapping_size]
        proj = 2 * np.pi * coords @ self.B.T
        
        # Apply cosine and sine
        cos_features = torch.cos(proj)  # [N, mapping_size]
        sin_features = torch.sin(proj)  # [N, mapping_size]
        
        # Concatenate cos and sin features
        fourier_features = torch.cat([cos_features, sin_features], dim=-1)  # [N, 2 * mapping_size]
        
        return fourier_features
    
    def get_output_dim(self):
        """Returns the output dimension of the Fourier features"""
        return 2 * self.mapping_size

def create_fourier_embeddings(positions, embedding_dim, scale=1.0, device='cuda'):
    """
    Create Fourier feature embeddings from 3D positions
    
    Args:
        positions (torch.Tensor): 3D positions of shape [N, 3]
        embedding_dim (int): Desired embedding dimension
        scale (float): Scale parameter for Fourier features
        device (str): Device to store tensors on
        
    Returns:
        torch.Tensor: Fourier feature embeddings of shape [N, embedding_dim]
    """
    # Calculate mapping size to get desired embedding dimension
    # Since we get 2 * mapping_size features (cos + sin), we need:
    mapping_size = embedding_dim // 2
    
    # Handle odd embedding dimensions by adding one extra feature
    if embedding_dim % 2 != 0:
        mapping_size += 1
    
    # Create Fourier feature generator
    ff_generator = FourierFeatureGenerator(
        input_dim=3,
        mapping_size=mapping_size,
        scale=scale,
        device=device
    )
    
    # Generate Fourier features
    fourier_features = ff_generator.encode(positions)
    
    # Trim to exact embedding dimension if needed
    if fourier_features.shape[1] > embedding_dim:
        fourier_features = fourier_features[:, :embedding_dim]
    
    return fourier_features 