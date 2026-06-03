import flax.linen as nn
import jax.numpy as jnp

class TrajectoryDiscriminator(nn.Module):
    """Trajectory-level discriminator cho action chunks.
    
    Input:  (obs_0, action_chunk_flat) — obs tại bước đầu + toàn bộ chunk hành động đã flatten
    Output: P(success | obs_0, action_chunk) ∈ [0, 1]
    
    Khác với SuccessDiscriminator cũ (per-step), model này nhìn toàn bộ 
    chunk (s_0, [a_0, ..., a_{H-1}]) để đánh giá trajectory-level.
    """
    hidden_dims: tuple = (256, 256)
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, observations, action_chunk, deterministic: bool = True):
        # Ghép nối trạng thái đầu và toàn bộ action chunk
        x = jnp.concatenate([observations, action_chunk], axis=-1)
        
        # Đi qua các lớp ẩn
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.swish(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
            
        # Đầu ra là 1 node, đi qua sigmoid để ép về xác suất [0, 1]
        x = nn.Dense(1)(x)
        return nn.sigmoid(x)