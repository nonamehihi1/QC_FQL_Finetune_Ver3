import flax.linen as nn
import jax.numpy as jnp

class PerStepDiscriminator(nn.Module):
    """Per-step discriminator for GAIL-style reward shaping.
    
    Input:  (observation_t, action_t) — observation and action at step t.
    Output: P(expert | observation_t, action_t) ∈ [0, 1]
    """
    hidden_dims: tuple = (256, 256)
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, observations, action, deterministic: bool = True):
        # Concatenate observation and action along the last dimension
        x = jnp.concatenate([observations, action], axis=-1)
        
        # Hidden layers
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.swish(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
            
        # Output probability ∈ [0, 1] via Sigmoid
        x = nn.Dense(1)(x)
        return nn.sigmoid(x)