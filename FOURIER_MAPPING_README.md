# Simple Fourier Mapping for 4D Gaussian Splatting

This implementation adds a simple Fourier feature mapping to the Gaussian embeddings in the 4D Gaussian Splatting model, following the approach from "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains" (Tancik et al., 2020).

## What is Fourier Mapping?

Fourier mapping transforms low-dimensional input features into a higher-dimensional space using sinusoidal functions. This helps neural networks learn high-frequency details more effectively, which can improve training efficiency and final quality.

For each input embedding `x`, the Fourier mapping computes:
```
γ(x) = [x, sin(2π B x), cos(2π B x)]
```
where `B` is a matrix of random frequencies.

## Implementation Details

The `SimpleFourierMapping` class:
- Takes Gaussian embeddings of dimension `D` 
- Maps them to dimension `D + 2 * num_frequencies * D`
- Uses random Gaussian frequencies (not learnable for simplicity)
- Concatenates original features with sin/cos components

## How to Use

### 1. Enable Fourier Mapping

Add these parameters to your configuration:

```python
ModelHiddenParams = dict(
    use_fourier_embedding=True,      # Enable Fourier mapping
    fourier_frequencies=4,           # Number of frequency components
    fourier_scale=1.0,              # Scale for random frequencies
    gaussian_embedding_dim=32,       # Base embedding dimension
)
```

### 2. Command Line Usage

```bash
python train.py -s <scene_path> -m <model_path> \
    --use_fourier_embedding \
    --fourier_frequencies 4 \
    --fourier_scale 1.0
```

### 3. Example Configuration

See `arguments/example_fourier.py` for a complete example configuration.

## Parameters

- **`use_fourier_embedding`** (bool, default: False): Enable/disable Fourier mapping
- **`fourier_frequencies`** (int, default: 4): Number of frequency components
  - Higher values = more expressive power but more computation
  - Typical range: 2-10
- **`fourier_scale`** (float, default: 1.0): Scale factor for random frequencies
  - Controls the frequency range
  - Typical range: 0.1-10.0

## Expected Benefits

1. **Improved Training Efficiency**: Networks can learn high-frequency details faster
2. **Better Detail Capture**: Enhanced ability to represent fine-grained temporal changes
3. **More Stable Training**: Fourier features can provide better gradient flow

## Performance Considerations

- **Memory**: Increases embedding dimension from `D` to `D + 2*freq*D`
- **Computation**: Additional sin/cos operations during forward pass
- **Training Time**: May converge faster despite increased computation per step

## Recommended Settings

For most scenes, start with:
- `fourier_frequencies=4`
- `fourier_scale=1.0`
- Monitor training loss and adjust if needed

For scenes with fine temporal details, try:
- `fourier_frequencies=6-8`
- `fourier_scale=0.5-2.0`

## Integration Points

The Fourier mapping is applied in:
1. `GaussianModel.get_embedding` property
2. Before feeding embeddings to the deformation network
3. Automatically handles dimension updates for the deformation network

## Future Extensions

This simple implementation can be extended with:
- Learnable frequencies
- Different frequency initialization strategies
- Separate Fourier mapping for temporal embeddings
- Adaptive frequency selection during training 