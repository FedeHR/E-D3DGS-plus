import torch
import numpy as np
import math


def initialize_gaussian_embeddings(positions, embedding_dim, init_type='zero', device='cuda'):
    """
    Comprehensive embedding initialization with scientific methods
    
    Note: For Fourier features, use SimpleFourierMapper instead - it provides
    superior implementation with learnable amplitude coefficients.
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
        std = math.sqrt(2.0 / (3 + embedding_dim))  # fan_in=3 (xyz), fan_out=embedding_dim (FIXED: Xavier standard uses 2.0)
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
    
    else:
        print(f"Warning: Unknown embedding_init '{init_type}', using zero initialization")
        return torch.zeros(num_points, embedding_dim, dtype=torch.float32, device=device)


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
        std = math.sqrt(2.0 / (1 + embedding_dim))  # fan_in=1 (time), fan_out=embedding_dim (FIXED: Xavier standard uses 2.0)
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


def calculate_effective_dimension(init_type, embedding_dim):
    """
    Calculate the effective dimensionality of the embedding for comparison purposes
    This helps understand the relationship between embeddings and normal embeddings
    """
    if init_type in ['zero', 'normal', 'random', 'uniform', 'xavier', 'xavier_normal', 'xavier_uniform', 
                     'kaiming', 'kaiming_normal', 'he_normal', 'he_uniform', 'sinusoidal']:
        return embedding_dim
    
    else:
        return embedding_dim


def print_initialization_info(init_type, embedding_dim, **kwargs):
    """Enhanced initialization info printing with scientific insights"""
    print(f"🎯 Embedding Initialization: {init_type}")
    print(f"   Dimension: {embedding_dim}")
    
    effective_dim = calculate_effective_dimension(init_type, embedding_dim)
    if effective_dim != embedding_dim:
        print(f"   Effective Dimension: {effective_dim}")
    
    if init_type in ['xavier', 'xavier_normal', 'xavier_uniform']:
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
        
        # Temporal-specific
        'sinusoidal': 'Sinusoidal temporal embeddings (for temporal only)',
    }


def print_dimensionality_comparison(gaussian_dim):
    """
    Print a comparison of embedding dimensions for analysis
    """
    print("\n📐 Dimensionality Analysis:")
    print(f"   Normal Gaussian Embedding: {gaussian_dim}D")
    print(f"   Xavier/Kaiming/He Initialization: {gaussian_dim}D")
    print(f"   Sinusoidal Temporal Embedding: {gaussian_dim}D")
    
    print(f"\n🔬 Scientific Insight:")
    print(f"   • All methods provide {gaussian_dim}D embeddings")
    print(f"   • Xavier/Kaiming provide better gradient flow")
    print(f"   • Sinusoidal is optimized for temporal sequences")
    print(f"   • Use SimpleFourierMapper for Fourier features") 