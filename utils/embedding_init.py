import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from utils.fourier_features import create_fourier_embeddings


class EmbeddingInitializer:
    """
    Comprehensive embedding initialization strategies for E-D3DGS
    
    Based on research from:
    - "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains" (Tancik et al., 2020)
    - Standard neural network initialization literature
    """
    
    @staticmethod
    def initialize_gaussian_embeddings(positions, embedding_dim, init_type='random', device='cuda', **kwargs):
        """
        Initialize Gaussian embeddings with various strategies
        
        Args:
            positions (torch.Tensor): 3D positions of shape [N, 3]
            embedding_dim (int): Desired embedding dimension
            init_type (str): Type of initialization
            device (str): Device to store tensors on
            **kwargs: Additional parameters for specific initialization methods
            
        Returns:
            torch.Tensor: Initialized embeddings of shape [N, embedding_dim]
        """
        num_points = positions.shape[0]
        
        if init_type == 'zero':
            return EmbeddingInitializer._zero_init(num_points, embedding_dim, device)
        elif init_type == 'random' or init_type == 'normal':
            return EmbeddingInitializer._normal_init(num_points, embedding_dim, device, **kwargs)
        elif init_type == 'xavier' or init_type == 'xavier_uniform':
            return EmbeddingInitializer._xavier_uniform_init(num_points, embedding_dim, device)
        elif init_type == 'xavier_normal':
            return EmbeddingInitializer._xavier_normal_init(num_points, embedding_dim, device)
        elif init_type == 'kaiming' or init_type == 'he_uniform':
            return EmbeddingInitializer._kaiming_uniform_init(num_points, embedding_dim, device)
        elif init_type == 'kaiming_normal' or init_type == 'he_normal':
            return EmbeddingInitializer._kaiming_normal_init(num_points, embedding_dim, device)
        elif init_type == 'uniform':
            return EmbeddingInitializer._uniform_init(num_points, embedding_dim, device, **kwargs)
        elif init_type == 'fourier' or init_type == 'positional':
            return EmbeddingInitializer._fourier_init(positions, embedding_dim, device, **kwargs)
        elif init_type == 'structured_fourier':
            return EmbeddingInitializer._structured_fourier_init(positions, embedding_dim, device, **kwargs)
        elif init_type == 'learned_fourier':
            return EmbeddingInitializer._learned_fourier_init(positions, embedding_dim, device, **kwargs)
        else:
            raise ValueError(f"Unknown initialization type: {init_type}")
    
    @staticmethod
    def _zero_init(num_points, embedding_dim, device):
        """Zero initialization (baseline)"""
        return torch.zeros((num_points, embedding_dim), dtype=torch.float, device=device)
    
    @staticmethod
    def _normal_init(num_points, embedding_dim, device, std=0.01, mean=0.0):
        """Normal (Gaussian) initialization with configurable mean and std"""
        return torch.normal(mean, std, size=(num_points, embedding_dim), dtype=torch.float, device=device)
    
    @staticmethod
    def _xavier_uniform_init(num_points, embedding_dim, device):
        """Xavier/Glorot uniform initialization"""
        embeddings = torch.empty(num_points, embedding_dim, dtype=torch.float, device=device)
        init.xavier_uniform_(embeddings)
        return embeddings
    
    @staticmethod
    def _xavier_normal_init(num_points, embedding_dim, device):
        """Xavier/Glorot normal initialization"""
        embeddings = torch.empty(num_points, embedding_dim, dtype=torch.float, device=device)
        init.xavier_normal_(embeddings)
        return embeddings
    
    @staticmethod
    def _kaiming_uniform_init(num_points, embedding_dim, device):
        """Kaiming/He uniform initialization"""
        embeddings = torch.empty(num_points, embedding_dim, dtype=torch.float, device=device)
        init.kaiming_uniform_(embeddings, nonlinearity='relu')
        return embeddings
    
    @staticmethod
    def _kaiming_normal_init(num_points, embedding_dim, device):
        """Kaiming/He normal initialization"""
        embeddings = torch.empty(num_points, embedding_dim, dtype=torch.float, device=device)
        init.kaiming_normal_(embeddings, nonlinearity='relu')
        return embeddings
    
    @staticmethod
    def _uniform_init(num_points, embedding_dim, device, low=-0.1, high=0.1):
        """Uniform initialization with configurable range"""
        return torch.uniform(low, high, size=(num_points, embedding_dim), dtype=torch.float, device=device)
    
    @staticmethod
    def _fourier_init(positions, embedding_dim, device, fourier_scale=1.0):
        """
        Fourier Features initialization based on Tancik et al. 2020
        
        This is the key contribution from the paper:
        γ(v) = [cos(2πb₁ᵀv), sin(2πb₁ᵀv), ..., cos(2πbₘᵀv), sin(2πbₘᵀv)]ᵀ
        
        where bⱼ ~ N(0, σ²I) and σ is controlled by fourier_scale
        """
        print(f"Initializing with Fourier Features (scale={fourier_scale})")
        return create_fourier_embeddings(
            positions=positions,
            embedding_dim=embedding_dim,
            scale=fourier_scale,
            device=device
        )
    
    @staticmethod
    def _structured_fourier_init(positions, embedding_dim, device, 
                                fourier_scale=1.0, num_freq_bands=None, 
                                log_sampling=True, max_freq=None):
        """
        Structured Fourier Features similar to NeRF's positional encoding
        
        This uses logarithmically spaced frequencies for more structured encoding:
        L(p) = [sin(2⁰πp), cos(2⁰πp), sin(2¹πp), cos(2¹πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp)]
        """
        if num_freq_bands is None:
            num_freq_bands = embedding_dim // 6  # 6 = 2 * 3 (cos+sin for x,y,z)
        
        if max_freq is None:
            max_freq = num_freq_bands - 1
        
        print(f"Initializing with Structured Fourier Features (scale={fourier_scale}, bands={num_freq_bands})")
        
        # Generate frequency bands
        if log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, num_freq_bands, device=device)
        else:
            freq_bands = torch.linspace(1.0, 2.0**max_freq, num_freq_bands, device=device)
        
        freq_bands = freq_bands * fourier_scale * np.pi
        
        # Apply positional encoding to each coordinate
        embedded_coords = []
        for i in range(3):  # x, y, z coordinates
            coord = positions[:, i:i+1]  # [N, 1]
            coord_embeds = []
            for freq in freq_bands:
                coord_embeds.append(torch.sin(freq * coord))
                coord_embeds.append(torch.cos(freq * coord))
            embedded_coords.extend(coord_embeds)
        
        # Concatenate all embeddings
        full_embedding = torch.cat(embedded_coords, dim=1)  # [N, 6*num_freq_bands]
        
        # Truncate or pad to desired embedding dimension
        if full_embedding.shape[1] > embedding_dim:
            return full_embedding[:, :embedding_dim]
        elif full_embedding.shape[1] < embedding_dim:
            padding = torch.zeros(positions.shape[0], embedding_dim - full_embedding.shape[1], device=device)
            return torch.cat([full_embedding, padding], dim=1)
        else:
            return full_embedding
    
    @staticmethod
    def _learned_fourier_init(positions, embedding_dim, device, fourier_scale=1.0, 
                             num_freq_components=None, learnable_frequencies=False):
        """
        Learnable Fourier Features with optional trainable frequency parameters
        
        This allows for learning optimal frequency distributions during training
        """
        if num_freq_components is None:
            num_freq_components = embedding_dim // 2
        
        print(f"Initializing with Learned Fourier Features (scale={fourier_scale}, components={num_freq_components})")
        
        # Initialize random frequency matrix B ~ N(0, σ²I)
        B = torch.randn(num_freq_components, 3, device=device) * fourier_scale
        
        # Compute projections: 2π * B @ positions.T
        proj = 2 * np.pi * (B @ positions.T)  # [num_freq_components, N]
        proj = proj.T  # [N, num_freq_components]
        
        # Apply trigonometric functions
        cos_features = torch.cos(proj)
        sin_features = torch.sin(proj)
        
        # Concatenate and adjust dimension
        fourier_features = torch.cat([cos_features, sin_features], dim=1)  # [N, 2*num_freq_components]
        
        if fourier_features.shape[1] > embedding_dim:
            fourier_features = fourier_features[:, :embedding_dim]
        elif fourier_features.shape[1] < embedding_dim:
            padding_dim = embedding_dim - fourier_features.shape[1]
            padding = torch.randn(positions.shape[0], padding_dim, device=device) * 0.01
            fourier_features = torch.cat([fourier_features, padding], dim=1)
        
        return fourier_features
    
    @staticmethod
    def initialize_temporal_embeddings(num_frames, embedding_dim, init_type='normal', device='cuda', **kwargs):
        """
        Initialize temporal embeddings with various strategies
        
        Args:
            num_frames (int): Number of temporal frames
            embedding_dim (int): Temporal embedding dimension
            init_type (str): Type of initialization
            device (str): Device to store tensors on
            **kwargs: Additional parameters for specific initialization methods
            
        Returns:
            torch.Tensor: Initialized temporal embeddings of shape [num_frames, embedding_dim]
        """
        if init_type == 'zero':
            return torch.zeros(num_frames, embedding_dim, device=device)
        elif init_type == 'normal' or init_type == 'random':
            std = kwargs.get('std', 0.01 / np.sqrt(embedding_dim))
            return torch.normal(0.0, std, size=(num_frames, embedding_dim), device=device)
        elif init_type == 'xavier_uniform':
            embeddings = torch.empty(num_frames, embedding_dim, device=device)
            init.xavier_uniform_(embeddings)
            return embeddings
        elif init_type == 'xavier_normal':
            embeddings = torch.empty(num_frames, embedding_dim, device=device)
            init.xavier_normal_(embeddings)
            return embeddings
        elif init_type == 'sinusoidal':
            return EmbeddingInitializer._sinusoidal_temporal_init(num_frames, embedding_dim, device, **kwargs)
        else:
            raise ValueError(f"Unknown temporal initialization type: {init_type}")
    
    @staticmethod
    def _sinusoidal_temporal_init(num_frames, embedding_dim, device, base_freq=10000.0):
        """
        Sinusoidal temporal embeddings similar to Transformer positional encodings
        """
        position = torch.arange(num_frames, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float, device=device) * 
                           -(np.log(base_freq) / embedding_dim))
        
        embeddings = torch.zeros(num_frames, embedding_dim, device=device)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)
        
        return embeddings


def get_available_init_methods():
    """Return list of available initialization methods"""
    gaussian_methods = [
        'zero', 'random', 'normal', 'xavier', 'xavier_uniform', 'xavier_normal',
        'kaiming', 'he_uniform', 'kaiming_normal', 'he_normal', 'uniform',
        'fourier', 'positional', 'structured_fourier', 'learned_fourier'
    ]
    
    temporal_methods = [
        'zero', 'normal', 'random', 'xavier_uniform', 'xavier_normal', 'sinusoidal'
    ]
    
    return {
        'gaussian': gaussian_methods,
        'temporal': temporal_methods
    }


def print_initialization_info(init_type, embedding_dim, **kwargs):
    """Print information about the chosen initialization method"""
    print(f"🎯 Embedding Initialization: {init_type}")
    print(f"   Dimension: {embedding_dim}")
    
    if init_type in ['fourier', 'positional']:
        scale = kwargs.get('fourier_scale', 1.0)
        print(f"   Fourier Scale: {scale}")
        print("   Method: Random Fourier Features (Tancik et al., 2020)")
    elif init_type == 'structured_fourier':
        scale = kwargs.get('fourier_scale', 1.0)
        bands = kwargs.get('num_freq_bands', embedding_dim // 6)
        print(f"   Fourier Scale: {scale}, Frequency Bands: {bands}")
        print("   Method: Structured Fourier Features (NeRF-style)")
    elif init_type == 'learned_fourier':
        scale = kwargs.get('fourier_scale', 1.0)
        components = kwargs.get('num_freq_components', embedding_dim // 2)
        print(f"   Fourier Scale: {scale}, Components: {components}")
        print("   Method: Learnable Fourier Features")
    elif 'xavier' in init_type:
        print("   Method: Xavier/Glorot initialization")
    elif 'kaiming' in init_type or 'he_' in init_type:
        print("   Method: Kaiming/He initialization")
    elif init_type == 'normal' or init_type == 'random':
        std = kwargs.get('std', 0.01)
        print(f"   Method: Normal distribution (std={std})")
    elif init_type == 'zero':
        print("   Method: Zero initialization (baseline)") 