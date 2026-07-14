import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value

class ACFQLAgent(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent with action chunking and AW-AFC. 
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def mask_actions(self, batch_actions, k, action_dim):
        """Mask actions beyond step k for critic evaluation."""
        horizon = self.config['horizon_length']
        mask = jnp.arange(horizon * action_dim) < (k * action_dim)
        return batch_actions * mask[None, :]

    def critic_loss(self, batch, grad_params, rng):
        """Compute the AW-AFC critic loss for multiple chunk scales."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
            
        action_dim = self.config['action_dim']
        rng, sample_rng = jax.random.split(rng)
        
        # sample_actions returns (actions, optimal_k), we just need actions
        next_actions_full, _ = self.sample_actions(batch['next_observations'][..., -1, :], rng=sample_rng)

        def compute_k_loss(k, target_critic_name, critic_name):
            # Compute k-step return
            r_k = jnp.zeros_like(batch['rewards'][:, 0])
            for i in range(k):
                r_k += (self.config['discount'] ** i) * batch['rewards'][:, i]
            
            next_obs_k = batch['next_observations'][:, k-1, :]
            masked_next_actions = self.mask_actions(next_actions_full, k, action_dim)
            
            next_qs = self.network.select(target_critic_name)(next_obs_k, actions=masked_next_actions)
            next_q = next_qs.min(axis=0) if self.config['q_agg'] == 'min' else next_qs.mean(axis=0)
            
            target_q = r_k + (self.config['discount'] ** k) * batch['masks'][:, k-1] * next_q
            
            masked_actions = self.mask_actions(batch_actions, k, action_dim)
            q = self.network.select(critic_name)(batch['observations'], actions=masked_actions, params=grad_params)
            
            loss = (jnp.square(q - target_q) * batch['valid'][:, k-1]).mean()
            return loss, q

        loss_1, q_1 = compute_k_loss(1, 'target_critic_1', 'critic_1')
        loss_3, q_3 = compute_k_loss(3, 'target_critic_3', 'critic_3')
        loss_5, q_5 = compute_k_loss(5, 'target_critic_5', 'critic_5')
        
        critic_loss = loss_1 + loss_3 + loss_5

        return critic_loss, {
            'critic_loss': critic_loss,
            'loss_1': loss_1,
            'loss_3': loss_3,
            'loss_5': loss_5,
            'q_1_mean': q_1.mean(),
            'q_3_mean': q_3.mean(),
            'q_5_mean': q_5.mean(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss with max advantage weighting."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
        batch_size, full_action_dim = batch_actions.shape
        action_dim = self.config['action_dim']
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, full_action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)

        # Advantage computation
        qs_1 = self.network.select('critic_1')(batch['observations'], actions=self.mask_actions(batch_actions, 1, action_dim))
        qs_3 = self.network.select('critic_3')(batch['observations'], actions=self.mask_actions(batch_actions, 3, action_dim))
        qs_5 = self.network.select('critic_5')(batch['observations'], actions=self.mask_actions(batch_actions, 5, action_dim))
        
        q_1 = jax.lax.stop_gradient(jnp.mean(qs_1, axis=0))
        q_3 = jax.lax.stop_gradient(jnp.mean(qs_3, axis=0))
        q_5 = jax.lax.stop_gradient(jnp.mean(qs_5, axis=0))
        
        gamma = self.config['discount']
        adv_1 = (q_1 - jnp.median(q_1)) / 1.0
        adv_3 = (q_3 - jnp.median(q_3)) / (1.0 + gamma + gamma**2)
        adv_5 = (q_5 - jnp.median(q_5)) / (1.0 + gamma + gamma**2 + gamma**3 + gamma**4)
        
        best_adv = jnp.maximum(adv_1, jnp.maximum(adv_3, adv_5))
        
        if self.config.get("use_q_weighting", True):
            median_adv = jnp.median(best_adv)
            weights = jnp.where(best_adv >= median_adv, 2.0, 0.0)
        else:
            weights = jnp.ones_like(best_adv)
        
        mse_loss = (pred - vel) ** 2
        if self.config["action_chunking"]:
            mse_loss = jnp.reshape(
                mse_loss, 
                (batch_size, self.config["horizon_length"], action_dim) 
            ) * batch["valid"][..., None]
            per_sample_loss = jnp.mean(mse_loss, axis=(1, 2))
        else:
            per_sample_loss = jnp.mean(mse_loss, axis=-1)

        bc_flow_loss = jnp.mean(per_sample_loss * weights)

        actor_loss = bc_flow_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)
        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        
        # update all 3 target critics
        agent.target_update(new_network, 'critic_1')
        agent.target_update(new_network, 'critic_3')
        agent.target_update(new_network, 'critic_5')
        
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)
    
    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)
    
    @jax.jit
    def sample_actions(self, observations, rng=None):
        if self.config["actor_type"] == "best-of-n":
            action_dim = self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1)
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                    self.config["actor_num_samples"], action_dim
                ),
            )
            observations_rep = jnp.repeat(observations[..., None, :], self.config["actor_num_samples"], axis=-2)
            actions = self.compute_flow_actions(observations_rep, noises)
            actions = jnp.clip(actions, -1, 1)
            
            # evaluate critics for AQC
            batch_actions = jnp.reshape(actions, (-1, actions.shape[-1]))
            obs_flat = jnp.reshape(observations_rep, (-1, observations.shape[-1]))
            
            gamma = self.config['discount']
            
            def get_adv(k, critic_name, norm_factor):
                q = self.network.select(critic_name)(obs_flat, actions=self.mask_actions(batch_actions, k, self.config['action_dim']))
                q = q.mean(axis=0) if self.config["q_agg"] == "mean" else q.min(axis=0)
                q = q.reshape(*actions.shape[:-1]) # (batch, num_samples)
                v = jnp.mean(q, axis=-1, keepdims=True)
                return (q - v) / norm_factor
                
            adv_1 = get_adv(1, 'critic_1', 1.0)
            adv_3 = get_adv(3, 'critic_3', 1.0 + gamma + gamma**2)
            adv_5 = get_adv(5, 'critic_5', 1.0 + gamma + gamma**2 + gamma**3 + gamma**4)
            
            advs = jnp.stack([adv_1, adv_3, adv_5], axis=-1) # (batch, num_samples, 3)
            
            # select optimal sample and k
            best_sample_idx = jnp.argmax(jnp.max(advs, axis=-1), axis=-1) # (batch,)
            
            is_unbatched = (len(observations.shape) == len(self.config['ob_dims']))
            
            if is_unbatched:
                best_sample = best_sample_idx
                best_k_idx = jnp.argmax(advs[best_sample])
                optimal_k = jnp.array([1, 3, 5])[best_k_idx]
                optimal_action = actions[best_sample]
            else:
                b_indices = jnp.arange(observations.shape[0])
                best_k_idx = jnp.argmax(advs[b_indices, best_sample_idx], axis=-1)
                optimal_k = jnp.array([1, 3, 5])[best_k_idx]
                optimal_action = actions[b_indices, best_sample_idx]
                
            return optimal_action, optimal_k

        return actions, jnp.array(self.config['horizon_length'])

    @jax.jit
    def compute_flow_actions(self, observations, noises):
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
        )

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
        )

        critic_1_def = critic_def
        critic_3_def = copy.deepcopy(critic_def)
        critic_5_def = copy.deepcopy(critic_def)

        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_times)),
            critic_1=(critic_1_def, (ex_observations, full_actions)),
            target_critic_1=(copy.deepcopy(critic_1_def), (ex_observations, full_actions)),
            critic_3=(critic_3_def, (ex_observations, full_actions)),
            target_critic_3=(copy.deepcopy(critic_3_def), (ex_observations, full_actions)),
            critic_5=(critic_5_def, (ex_observations, full_actions)),
            target_critic_5=(copy.deepcopy(critic_5_def), (ex_observations, full_actions)),
        )
        if encoders.get('actor_bc_flow') is not None:
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
            
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        if config["weight_decay"] > 0.:
            network_tx = optax.adamw(learning_rate=config['lr'], weight_decay=config["weight_decay"])
        else:
            network_tx = optax.adam(learning_rate=config['lr'])
            
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params[f'modules_target_critic_1'] = params[f'modules_critic_1']
        params[f'modules_target_critic_3'] = params[f'modules_critic_3']
        params[f'modules_target_critic_5'] = params[f'modules_critic_5']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='acfql',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            q_agg='mean',
            alpha=100.0,
            num_qs=2,
            flow_steps=10,
            normalize_q_loss=False,
            encoder=ml_collections.config_dict.placeholder(str),
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=True,
            actor_type="best-of-n",
            actor_num_samples=32,
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.,
            use_q_weighting=True,
            alpha_penalty=0.0,
        )
    )
    return config
