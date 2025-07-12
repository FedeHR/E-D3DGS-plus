import torch
import numpy as np
import math


def initialize_gaussian_embeddings(positions, embedding_dim, init_type='zero', device='cuda'):
    """
    Comprehensive embedding initialization (identical to version on Niklas branch, but removing the fouier inits)
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
    else:
        print(f"Warning: Unknown embedding_init '{init_type}', using zero initialization")
        return torch.zeros(num_points, embedding_dim, dtype=torch.float32, device=device)


# For now we only work with spatial embeddings
# def initialize_temporal_embeddings(num_frames, embedding_dim, init_type='normal', device='cuda', scale=1.0):
#     """
#     Comprehensive temporal embedding initialization
#     """
#     if init_type == 'zero':
#         return torch.zeros(num_frames, embedding_dim, dtype=torch.float32, device=device)
    
#     elif init_type in ['normal', 'random']:
#         return torch.randn(num_frames, embedding_dim, dtype=torch.float32, device=device) * 0.01
    
#     elif init_type == 'uniform':
#         return torch.empty(num_frames, embedding_dim, dtype=torch.float32, device=device).uniform_(-0.01, 0.01)
    
#     elif init_type in ['xavier', 'xavier_normal']:
#         std = math.sqrt(2.0 / (1 + embedding_dim))  # fan_in=1 (time), fan_out=embedding_dim
#         return torch.randn(num_frames, embedding_dim, dtype=torch.float32, device=device) * std
    
#     elif init_type == 'xavier_uniform':
#         a = math.sqrt(6.0 / (1 + embedding_dim))
#         return torch.empty(num_frames, embedding_dim, dtype=torch.float32, device=device).uniform_(-a, a)
    
#     elif init_type in ['kaiming', 'kaiming_normal', 'he_normal']:
#         std = math.sqrt(2.0 / 1)  # fan_in=1 (time dimension)
#         return torch.randn(num_frames, embedding_dim, dtype=torch.float32, device=device) * std
    
#     elif init_type == 'he_uniform':
#         bound = math.sqrt(6.0 / 1)
#         return torch.empty(num_frames, embedding_dim, dtype=torch.float32, device=device).uniform_(-bound, bound)
    
#     elif init_type == 'sinusoidal':
#         # Sinusoidal temporal embeddings for smooth time encoding
#         position = torch.arange(num_frames, dtype=torch.float32, device=device).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32, device=device) *
#                            -(math.log(10000.0) / embedding_dim))
        
#         embeddings = torch.zeros(num_frames, embedding_dim, dtype=torch.float32, device=device)
#         embeddings[:, 0::2] = torch.sin(position * div_term)
#         if embedding_dim % 2 == 1:
#             embeddings[:, 1::2] = torch.cos(position * div_term[:-1])
#         else:
#             embeddings[:, 1::2] = torch.cos(position * div_term)
        
#         return embeddings * scale
    
#     else:
#         print(f"Warning: Unknown temporal_embedding_init '{init_type}', using normal initialization")
#         return torch.randn(num_frames, embedding_dim, dtype=torch.float32, device=device) * 0.01
