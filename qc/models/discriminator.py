import flax.linen as nn
import jax.numpy as jnp

class SuccessDiscriminator(nn.Module):
    hidden_dims: tuple = (256, 256)
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, observations, actions, deterministic: bool = True):
        # Ghép nối trạng thái và hành động
        x = jnp.concatenate([observations, actions], axis=-1)
        
        # Đi qua các lớp ẩn
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.swish(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
            
        # Đầu ra là 1 node, đi qua sigmoid để ép về xác suất [0, 1]
        x = nn.Dense(1)(x)
        return nn.sigmoid(x)