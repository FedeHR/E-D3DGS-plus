_base_ = './default.py'
ModelHiddenParams = dict(
    min_embeddings = 16,
    max_embeddings = 80,
    c2f_temporal_iter = 10000,
    total_num_frames = 164,
    gaussian_embedding_dim = 8,

    # Enable Fourier mapping for Gaussian embeddings
    use_fourier_embedding=True,
    
    # Number of frequency components
    # Higher values give more expressive power but increase computation
    fourier_frequencies=4,
    
    # Scale factor for the random frequencies (default: 1.0)
    # Controls the range of frequencies used in the mapping
    fourier_scale=1.0,
)

OptimizationParams = dict(
    maxtime = 164,
    iterations = 60_000,
    densify_until_iter = 60_000,
    position_lr_max_steps = 60_000,
    deformation_lr_max_steps = 60_000,
)