import flax.linen as nn
import jax.numpy as jnp

class PerStepDiscriminator(nn.Module):
    """Per-step discriminator for GAIL-style reward shaping.
    
    Input:  (observation_t, action_t) — observation and action at step t.
    Output: P(expert | observation_t, action_t) ∈ [0, 1]
    
    v4 Changes:
    - Removed LayerNorm (causes issues with Gradient Penalty computation)
    - Using LeakyReLU (standard for GAN/GAIL discriminators)
    - Larger capacity (512, 512, 256)
    - Spectral-norm-friendly architecture (simple Dense + activation)
    """
    hidden_dims: tuple = (512, 512, 256)
    dropout_rate: float = 0.05

    @nn.compact
    def __call__(self, observations, action, deterministic: bool = True):
        # Concatenate observation and action along the last dimension
        x = jnp.concatenate([observations, action], axis=-1)
        
        # Hidden layers: Dense + LeakyReLU + Dropout (no LayerNorm)
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.leaky_relu(x, negative_slope=0.2)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
            
        # Output logits instead of probabilities for numerical stability
        x = nn.Dense(1)(x)
        return x