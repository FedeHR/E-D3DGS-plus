import torch
import numpy as np
import math


def initialize_gaussian_embeddings(positions, embedding_dim, init_type='zero', 
                                 fourier_scale=1.0, num_freq_bands=None, device='cuda'):
    """
    Comprehensive embedding initialization with scientific methods
    Based on "Fourier Features Let Networks Learn High Frequency Functions" (Tancik et al., 2020)
    """
    num_points = positions.shape[0]
    
    if init_type == 'zero':
        return torch.zeros(num_points, embedding_dim, dtype=torch.float32, device=device)
    
    elif init_type in ['random', 'normal']:
        return torch.randn(num_points, embedding_dim, dtype=torch.float32, device=device) * 0.01
    
    elif init_type == 'uniform':
        return torch.empty(num_points, embedding_dim, dtype=torch.float32, device=device).uniform_(-0.01, 0.01)
    
    elif init_type in ['xavier', 'xavier_normal']:
        # Xavier normal initialization for better gradient flow
        std = math.sqrt(2.0 / (3 + embedding_dim))  # fan_in=3 (xyz), fan_out=embedding_dim
        return torch.randn(num_points, embedding_dim, dtype=torch.float32, device=device) * std
    
    elif init_type == 'xavier_uniform':
        # Xavier uniform initialization
        a = math.sqrt(6.0 / (3 + embedding_dim))
        return torch.empty(num_points, embedding_dim, dtype=torch.float32, device=device).uniform_(-a, a)
    
    elif init_type in ['kaiming', 'kaiming_normal', 'he_normal']:
        # Kaiming/He normal initialization for ReLU networks
        std = math.sqrt(2.0 / 3)  # fan_in=3 (xyz coordinates)
        return torch.randn(num_points, embedding_dim, dtype=torch.float32, device=device) * std
    
    elif init_type == 'he_uniform':
        # He uniform initialization
        bound = math.sqrt(6.0 / 3)  # fan_in=3
        return torch.empty(num_points, embedding_dim, dtype=torch.float32, device=device).uniform_(-bound, bound)
    
    elif init_type in ['fourier', 'positional']:
        # Random Fourier Features as in Tancik et al. 2020
        return _create_random_fourier_features(positions, embedding_dim, fourier_scale, device)
    
    elif init_type == 'structured_fourier':
        # NeRF-style positional encoding with logarithmic frequency spacing
        return _create_positional_encoding(positions, embedding_dim, fourier_scale, num_freq_bands, device)
    
    elif init_type == 'learned_fourier':
        # Learnable Fourier features with adaptive frequencies
        return _create_learned_fourier_features(positions, embedding_dim, fourier_scale, device)
    
    elif init_type == 'positional_encoding':
        # Alias for structured_fourier (commonly used name in literature)
        return _create_positional_encoding(positions, embedding_dim, fourier_scale, num_freq_bands, device)
    
    else:
        print(f"Warning: Unknown embedding_init '{init_type}', using zero initialization")
        return torch.zeros(num_points, embedding_dim, dtype=torch.float32, device=device)


def _create_random_fourier_features(positions, embedding_dim, scale, device):
    """
    Random Fourier Features implementation from Tancik et al. 2020
    γ(v) = [a₁cos(2πb₁ᵀv), a₁sin(2πb₁ᵀv), ..., aₘcos(2πbₘᵀv), aₘsin(2πbₘᵀv)]ᵀ
    """
    # Calculate mapping size to get desired embedding dimension
    mapping_size = embedding_dim // 2
    
    # Generate random frequency matrix B ~ N(0, scale²I)
    B = torch.randn(mapping_size, 3, dtype=torch.float32, device=device) * scale
    
    # Compute projections
    proj = torch.mm(positions, B.T) * (2 * np.pi)  # [N, mapping_size]
    
    # Apply trigonometric functions
    cos_features = torch.cos(proj)
    sin_features = torch.sin(proj)
    
    # Concatenate cos and sin features
    fourier_features = torch.cat([cos_features, sin_features], dim=1)
    
    # Handle dimension mismatch
    if fourier_features.shape[1] > embedding_dim:
        return fourier_features[:, :embedding_dim]
    elif fourier_features.shape[1] < embedding_dim:
        padding_size = embedding_dim - fourier_features.shape[1]
        padding = torch.zeros(positions.shape[0], padding_size, dtype=torch.float32, device=device)
        return torch.cat([fourier_features, padding], dim=1)
    
    return fourier_features


def _create_positional_encoding(positions, embedding_dim, scale, num_freq_bands, device):
    """
    NeRF-style positional encoding with logarithmic frequency spacing
    L(p) = [sin(2⁰πp), cos(2⁰πp), sin(2¹πp), cos(2¹πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp)]
    """
    if num_freq_bands is None:
        # Calculate number of frequency bands based on embedding dimension
        # Each frequency band contributes 2*3=6 features (sin/cos for x,y,z)
        num_freq_bands = embedding_dim // 6
        if num_freq_bands == 0:
            num_freq_bands = 1
    
    features = []
    
    # Add input coordinates themselves (identity mapping)
    features.append(positions)
    
    # Add Fourier features for each frequency band
    for i in range(num_freq_bands):
        freq = (2.0 ** i) * scale
        
        # Apply to all coordinate dimensions
        for coord_dim in range(3):
            coord = positions[:, coord_dim:coord_dim+1]
            features.append(torch.sin(freq * np.pi * coord))
            features.append(torch.cos(freq * np.pi * coord))
    
    # Concatenate all features
    all_features = torch.cat(features, dim=1)
    
    # Adjust to desired embedding dimension
    if all_features.shape[1] > embedding_dim:
        return all_features[:, :embedding_dim]
    elif all_features.shape[1] < embedding_dim:
        padding_size = embedding_dim - all_features.shape[1]
        padding = torch.zeros(positions.shape[0], padding_size, dtype=torch.float32, device=device)
        return torch.cat([all_features, padding], dim=1)
    
    return all_features


def _create_learned_fourier_features(positions, embedding_dim, scale, device):
    """
    Learnable Fourier features with trainable frequency distributions
    Initialize with random frequencies but allow adaptation during training
    """
    mapping_size = embedding_dim // 2
    
    # Initialize with random frequencies (these could be made learnable)
    B = torch.randn(mapping_size, 3, dtype=torch.float32, device=device) * scale
    
    # Add some structure to the initial frequencies
    # Use a mix of low and high frequencies
    low_freq_portion = mapping_size // 3
    med_freq_portion = mapping_size // 3
    
    # Low frequencies
    B[:low_freq_portion] *= 0.5
    # Medium frequencies  
    B[low_freq_portion:low_freq_portion + med_freq_portion] *= 1.0
    # High frequencies
    B[low_freq_portion + med_freq_portion:] *= 2.0
    
    proj = torch.mm(positions, B.T) * (2 * np.pi)
    cos_features = torch.cos(proj)
    sin_features = torch.sin(proj)
    
    fourier_features = torch.cat([cos_features, sin_features], dim=1)
    
    # Handle dimension mismatch
    if fourier_features.shape[1] > embedding_dim:
        return fourier_features[:, :embedding_dim]
    elif fourier_features.shape[1] < embedding_dim:
        padding_size = embedding_dim - fourier_features.shape[1]
        padding = torch.zeros(positions.shape[0], padding_size, dtype=torch.float32, device=device)
        return torch.cat([fourier_features, padding], dim=1)
    
    return fourier_features


def initialize_temporal_embeddings(num_frames, embedding_dim, init_type='normal', device='cuda', scale=1.0):
    """
    Comprehensive temporal embedding initialization
    """
    if init_type == 'zero':
        return torch.zeros(num_frames, embedding_dim, dtype=torch.float32, device=device)
    
    elif init_type in ['normal', 'random']:
        return torch.randn(num_frames, embedding_dim, dtype=torch.float32, device=device) * 0.01
    
    elif init_type == 'uniform':
        return torch.empty(num_frames, embedding_dim, dtype=torch.float32, device=device).uniform_(-0.01, 0.01)
    
    elif init_type in ['xavier', 'xavier_normal']:
        std = math.sqrt(2.0 / (1 + embedding_dim))  # fan_in=1 (time), fan_out=embedding_dim
        return torch.randn(num_frames, embedding_dim, dtype=torch.float32, device=device) * std
    
    elif init_type == 'xavier_uniform':
        a = math.sqrt(6.0 / (1 + embedding_dim))
        return torch.empty(num_frames, embedding_dim, dtype=torch.float32, device=device).uniform_(-a, a)
    
    elif init_type in ['kaiming', 'kaiming_normal', 'he_normal']:
        std = math.sqrt(2.0 / 1)  # fan_in=1 (time dimension)
        return torch.randn(num_frames, embedding_dim, dtype=torch.float32, device=device) * std
    
    elif init_type == 'he_uniform':
        bound = math.sqrt(6.0 / 1)
        return torch.empty(num_frames, embedding_dim, dtype=torch.float32, device=device).uniform_(-bound, bound)
    
    elif init_type == 'sinusoidal':
        # Sinusoidal temporal embeddings for smooth time encoding
        position = torch.arange(num_frames, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32, device=device) *
                           -(math.log(10000.0) / embedding_dim))
        
        embeddings = torch.zeros(num_frames, embedding_dim, dtype=torch.float32, device=device)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        if embedding_dim % 2 == 1:
            embeddings[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            embeddings[:, 1::2] = torch.cos(position * div_term)
        
        return embeddings * scale
    
    else:
        print(f"Warning: Unknown temporal_embedding_init '{init_type}', using normal initialization")
        return torch.randn(num_frames, embedding_dim, dtype=torch.float32, device=device) * 0.01


def calculate_effective_dimension(init_type, embedding_dim, fourier_scale=None, num_freq_bands=None):
    """
    Calculate the effective dimensionality of the embedding for comparison purposes
    This helps understand the relationship between Fourier features and normal embeddings
    """
    if init_type in ['zero', 'normal', 'random', 'uniform', 'xavier', 'xavier_normal', 'xavier_uniform', 
                     'kaiming', 'kaiming_normal', 'he_normal', 'he_uniform']:
        return embedding_dim
    
    elif init_type in ['fourier', 'positional']:
        # Random Fourier features: each mapping contributes 2 dimensions (cos + sin)
        # Total effective dimension is 2 * (embedding_dim // 2) = embedding_dim
        return embedding_dim
    
    elif init_type in ['structured_fourier', 'positional_encoding']:
        # Positional encoding: 3 (identity) + 2*3*num_freq_bands
        if num_freq_bands is None:
            num_freq_bands = embedding_dim // 6
        effective_dim = 3 + 6 * num_freq_bands
        return min(effective_dim, embedding_dim)
    
    elif init_type == 'learned_fourier':
        return embedding_dim
    
    elif init_type == 'sinusoidal':
        return embedding_dim
    
    else:
        return embedding_dim


def print_initialization_info(init_type, embedding_dim, **kwargs):
    """Enhanced initialization info printing with scientific insights"""
    fourier_scale = kwargs.get('fourier_scale', 1.0)
    num_freq_bands = kwargs.get('num_freq_bands', None)
    
    print(f"🎯 Embedding Initialization: {init_type}")
    print(f"   Dimension: {embedding_dim}")
    
    effective_dim = calculate_effective_dimension(init_type, embedding_dim, fourier_scale, num_freq_bands)
    if effective_dim != embedding_dim:
        print(f"   Effective Dimension: {effective_dim}")
    
    if init_type in ['fourier', 'positional']:
        print(f"   Fourier Scale: {fourier_scale} (frequency range: 0-{fourier_scale:.1f})")
        print(f"   Mapping Size: {embedding_dim // 2} frequency vectors")
        
    elif init_type in ['structured_fourier', 'positional_encoding']:
        if num_freq_bands is None:
            num_freq_bands = embedding_dim // 6
        print(f"   Fourier Scale: {fourier_scale} (base frequency)")
        print(f"   Frequency Bands: {num_freq_bands} (powers of 2)")
        print(f"   Frequency Range: [{fourier_scale:.1f}, {fourier_scale * (2**(num_freq_bands-1)):.1f}]")
        
    elif init_type == 'learned_fourier':
        print(f"   Fourier Scale: {fourier_scale} (adaptive)")
        print(f"   Multi-scale initialization: Low/Med/High frequencies")
        
    elif init_type in ['xavier', 'xavier_normal', 'xavier_uniform']:
        fan_in, fan_out = 3, embedding_dim
        print(f"   Xavier initialization: fan_in={fan_in}, fan_out={fan_out}")
        
    elif init_type in ['kaiming', 'kaiming_normal', 'he_normal', 'he_uniform']:
        print(f"   Kaiming/He initialization: fan_in=3 (suitable for ReLU networks)")
        
    elif init_type == 'sinusoidal':
        print(f"   Sinusoidal encoding: smooth temporal transitions")


def get_available_methods():
    """Return all available initialization methods with descriptions"""
    return {
        # Classical methods
        'zero': 'Zero initialization (stable baseline)',
        'normal': 'Standard normal distribution N(0,0.01²)',
        'random': 'Alias for normal',
        'uniform': 'Uniform distribution U(-0.01,0.01)',
        
        # Xavier/Glorot family
        'xavier': 'Xavier normal initialization (better gradient flow)',
        'xavier_normal': 'Xavier normal (same as xavier)',
        'xavier_uniform': 'Xavier uniform initialization',
        
        # Kaiming/He family  
        'kaiming': 'Kaiming normal initialization (for ReLU networks)',
        'kaiming_normal': 'Kaiming normal (same as kaiming)',
        'he_normal': 'He normal (same as kaiming_normal)',
        'he_uniform': 'He uniform initialization',
        
        # Fourier-based methods
        'fourier': 'Random Fourier Features (Tancik et al. 2020)',
        'positional': 'Alias for random Fourier Features',
        'structured_fourier': 'NeRF-style positional encoding',
        'positional_encoding': 'Alias for structured_fourier',
        'learned_fourier': 'Learnable Fourier features with adaptive frequencies',
        
        # Temporal-specific
        'sinusoidal': 'Sinusoidal temporal embeddings (for temporal only)',
    }


def print_dimensionality_comparison(gaussian_dim, fourier_scale, num_freq_bands=None):
    """
    Print a comparison between normal Gaussian embeddings and Fourier features
    to help understand the relationship discussed in the paper
    """
    print("\n📐 Dimensionality Analysis:")
    print(f"   Normal Gaussian Embedding: {gaussian_dim}D")
    
    # Random Fourier Features
    fourier_mappings = gaussian_dim // 2
    print(f"   Random Fourier Features: {fourier_mappings} frequency mappings → {gaussian_dim}D")
    print(f"      Each mapping: 1 frequency vector (3D) → 2 features (cos + sin)")
    
    # Structured Fourier (positional encoding)
    if num_freq_bands is None:
        num_freq_bands = gaussian_dim // 6
        if num_freq_bands == 0:
            num_freq_bands = 1
    
    pos_encoding_dim = 3 + 6 * num_freq_bands  # 3 identity + 6 per freq band
    actual_pos_dim = min(pos_encoding_dim, gaussian_dim)
    
    print(f"   Positional Encoding: {num_freq_bands} freq bands → {actual_pos_dim}D")
    print(f"      Structure: 3 (identity) + {num_freq_bands}×6 (sin/cos for x,y,z)")
    
    print(f"\n🔬 Scientific Insight:")
    print(f"   • Random Fourier with scale {fourier_scale:.1f}: covers frequencies 0-{fourier_scale:.1f}")
    print(f"   • Positional encoding: covers frequencies {fourier_scale:.1f}-{fourier_scale * (2**(num_freq_bands-1)):.1f}")
    print(f"   • Higher scales = better high-frequency details")
    print(f"   • More freq bands = wider frequency coverage") 