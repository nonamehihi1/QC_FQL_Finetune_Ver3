import glob, tqdm, wandb, os, json, random, time, jax
from collections import deque
import jax.numpy as jnp
import numpy as np
import optax

from absl import app, flags
from ml_collections import config_flags

from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger
from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from envs.robomimic_utils import is_robomimic_env

from utils.datasets import Dataset, ReplayBuffer
from evaluation import evaluate
from agents import agents

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'GAIL_Mentor', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task3-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Evaluation interval.')
flags.DEFINE_integer('start_training', 5000, 'when does training start')
flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")
flags.DEFINE_float('discount', 0.99, 'discount factor')
flags.DEFINE_integer('eval_episodes', 10, 'Number of eval episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/acfql.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')
flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")

# === GAIL FLAGS (v3 — improved) ===
flags.DEFINE_bool('use_discriminator', True, "Enable GAIL Discriminator")
flags.DEFINE_float('disc_beta', 0.1, "Weight of discriminator reward")
flags.DEFINE_integer('disc_update_interval', 2000, "Train discriminator every N steps")
flags.DEFINE_integer('disc_gradient_steps', 20, "Number of disc gradient steps per update")
flags.DEFINE_float('disc_lr', 1e-4, "Discriminator learning rate")
flags.DEFINE_float('disc_gp_coeff', 5.0, "Gradient penalty coefficient")
flags.DEFINE_integer('disc_warmup_steps', 100000, "Beta warm-up steps")
flags.DEFINE_integer('disc_buffer_tail', 30, "Number of tail steps from episode to add to success/fail buffer")
flags.DEFINE_integer('disc_min_buffer', 128, "Minimum buffer size before disc training starts")

class LoggingHelper:
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)

def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    run = setup_wandb(project='qc', group=FLAGS.run_group, name=exp_name)
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(get_flag_dict(), f)

    config = FLAGS.agent

    if FLAGS.ogbench_dataset_dir is not None:
        dataset_paths = [file for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz")) if '-val.npz' not in file]
        env, eval_env, train_dataset, _ = make_ogbench_env_and_datasets(FLAGS.env_name, dataset_path=dataset_paths[0], compact_dataset=False)
    else:
        env, eval_env, train_dataset, _ = make_env_and_datasets(FLAGS.env_name)

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)
    log_step = 0
    config["horizon_length"] = FLAGS.horizon_length

    train_dataset = Dataset.create(**train_dataset)
    example_batch = train_dataset.sample(())

    agent = agents[config['agent_name']].create(FLAGS.seed, example_batch['observations'], example_batch['actions'], config)

    # ====================== DISCRIMINATOR SETUP (v3) ======================
    if FLAGS.use_discriminator:
        from models.discriminator import PerStepDiscriminator
        disc_model = PerStepDiscriminator()
        disc_rng = jax.random.PRNGKey(FLAGS.seed + 999)
        
        # Init with per-step input: (obs, action)
        obs_dim = example_batch['observations'].shape[-1]
        act_dim = example_batch['actions'].shape[-1]
        example_obs_init = np.zeros((1, obs_dim))
        example_act_init = np.zeros((1, act_dim))
        disc_params = disc_model.init(disc_rng, example_obs_init, example_act_init)['params']
        disc_tx = optax.adam(learning_rate=FLAGS.disc_lr)
        disc_opt_state = disc_tx.init(disc_params)

        # --- Discriminator update with Gradient Penalty + Label Smoothing ---
        @jax.jit
        def update_discriminator_step(params, opt_state, expert_obs, expert_acts, 
                                       policy_obs, policy_acts, rng_key):
            """Train discriminator with GP + label smoothing.
            
            Args:
                expert_obs, expert_acts: transitions from success buffer (N, dim)
                policy_obs, policy_acts: transitions from fail buffer (N, dim)
            """
            def disc_loss_fn(p):
                rng_dropout, rng_gp = jax.random.split(rng_key)
                
                # Forward pass — expert (success) and policy (fail)
                expert_pred = disc_model.apply(
                    {'params': p}, expert_obs, expert_acts, 
                    deterministic=False, rngs={'dropout': rng_dropout})  # (N, 1)
                policy_pred = disc_model.apply(
                    {'params': p}, policy_obs, policy_acts,
                    deterministic=False, rngs={'dropout': rng_dropout})  # (N, 1)
                
                # BCE with label smoothing (0.9/0.1 instead of 1.0/0.0)
                # Reduces overconfidence when labels are noisy (whole-episode labeling)
                expert_loss = -jnp.mean(
                    0.9 * jnp.log(expert_pred + 1e-8) + 
                    0.1 * jnp.log(1.0 - expert_pred + 1e-8))
                policy_loss = -jnp.mean(
                    0.1 * jnp.log(policy_pred + 1e-8) + 
                    0.9 * jnp.log(1.0 - policy_pred + 1e-8))
                bce_loss = (expert_loss + policy_loss) / 2.0
                
                # Gradient Penalty (WGAN-GP style) on interpolated samples
                n = expert_obs.shape[0]
                alpha = jax.random.uniform(rng_gp, (n, 1))
                interp_obs = alpha * expert_obs + (1.0 - alpha) * policy_obs
                interp_acts = alpha * expert_acts + (1.0 - alpha) * policy_acts
                
                def gp_forward(obs_in, acts_in):
                    return disc_model.apply(
                        {'params': p}, obs_in, acts_in, deterministic=True).sum()
                
                grad_obs, grad_acts = jax.grad(gp_forward, argnums=(0, 1))(interp_obs, interp_acts)
                grad_norm = jnp.sqrt(
                    jnp.sum(grad_obs ** 2, axis=-1) + 
                    jnp.sum(grad_acts ** 2, axis=-1) + 1e-8)
                gp = jnp.mean((grad_norm - 1.0) ** 2)
                
                total_loss = bce_loss + FLAGS.disc_gp_coeff * gp
                
                # Metrics for logging
                expert_acc = jnp.mean(expert_pred > 0.5)
                policy_acc = jnp.mean(policy_pred < 0.5)
                
                return total_loss, {
                    'disc/loss': total_loss,
                    'disc/bce': bce_loss,
                    'disc/gp': gp,
                    'disc/expert_acc': expert_acc,
                    'disc/policy_acc': policy_acc,
                    'disc/accuracy': (expert_acc + policy_acc) / 2.0,
                    'disc/expert_pred_mean': jnp.mean(expert_pred),
                    'disc/policy_pred_mean': jnp.mean(policy_pred),
                }
            
            (loss, metrics), grads = jax.value_and_grad(disc_loss_fn, has_aux=True)(params)
            updates, new_opt_state = disc_tx.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, metrics

        # --- Per-step reward shaping with flatten approach (like Cách 1) ---
        discount_powers = jnp.power(FLAGS.discount, jnp.arange(FLAGS.horizon_length))

        @jax.jit
        def compute_shaped_rewards(params, batch_obs, batch_acts, batch_rewards, beta):
            """Shape cumulative rewards using per-step discriminator (flatten approach).
            
            Uses CENTERED logit reward: log(D/(1-D)) instead of -log(1-D).
            - When D ≈ 0.5 (uncertain) → r_disc ≈ 0 (no bias)
            - When D > 0.5 (expert-like) → r_disc > 0 (encourage)
            - When D < 0.5 (policy-like) → r_disc < 0 (discourage)
            
            This is critical because env reward ∈ [-3, 0]. An always-positive
            disc reward would add positive bias that corrupts Q-values.
            """
            B, H = batch_acts.shape[0], batch_acts.shape[1]
            
            # Flatten (B, H, dim) → (B*H, dim) for per-step evaluation
            flat_obs = batch_obs.reshape(B * H, -1)
            flat_acts = batch_acts.reshape(B * H, -1)
            
            # Per-step discriminator prediction
            d_probs = disc_model.apply(
                {'params': params}, flat_obs, flat_acts, deterministic=True)  # (B*H, 1)
            scores = d_probs.reshape(B, H)  # (B, H)
            
            # Centered logit reward: log(D) - log(1-D) ∈ (-∞, +∞), 0 when D=0.5
            r_disc_raw = jnp.log(scores + 1e-8) - jnp.log(1.0 - scores + 1e-8)
            # Clip to [-3, 3] then normalize to [-1, 1] — matches env reward scale
            r_disc = jnp.clip(r_disc_raw, -3.0, 3.0) / 3.0
            
            # Integrate into cumulative reward structure:
            # batch_rewards[:,i] = Σ_{t=0}^{i} γ^t r_env_t (already cumulative)
            # shaped[:,i] = batch_rewards[:,i] + β * Σ_{t=0}^{i} γ^t r_disc_t
            discounted_r_disc = r_disc * discount_powers[None, :]  # (B, H)
            cum_disc = jnp.cumsum(discounted_r_disc, axis=1)  # (B, H)
            
            shaped_rewards = batch_rewards + beta * cum_disc
            
            return shaped_rewards, jnp.mean(scores), jnp.mean(r_disc), jnp.max(r_disc)

        print("✅ Per-step Discriminator v3 ENABLED — Flatten + GP + Label Smoothing + Adaptive β")

    prefixes = ["eval", "env", "offline_agent", "online_agent"]
    logger = LoggingHelper({prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv")) for prefix in prefixes}, wandb)

    # ====================== OFFLINE RL ======================
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1), desc="Offline"):
        log_step += 1
        batch = train_dataset.sample_sequence(config['batch_size'], sequence_length=FLAGS.horizon_length, discount=FLAGS.discount)
        agent, offline_info = agent.update(batch)
        if i % FLAGS.log_interval == 0: logger.log(offline_info, "offline_agent", step=log_step)
        
        if FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0:
            eval_info, _, _ = evaluate(agent=agent, env=eval_env, action_dim=example_batch["actions"].shape[-1],
                num_eval_episodes=FLAGS.eval_episodes, num_video_episodes=0, video_frame_skip=3)
            logger.log(eval_info, "eval", step=log_step)

    # ====================== KHỞI TẠO 3 BUFFER ======================
    agent_replay_buffer = ReplayBuffer.create_from_initial_dataset(dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1))
    
    dummy_transition = jax.tree_util.tree_map(lambda x: np.zeros_like(x[0]), dict(train_dataset))
    dummy_transition['is_success'] = 0.0
    success_buffer = ReplayBuffer.create(dummy_transition, size=FLAGS.buffer_size)
    fail_buffer = ReplayBuffer.create(dummy_transition, size=FLAGS.buffer_size)

    # ====================== ONLINE RL ======================
    ob, _ = env.reset()
    action_queue = []
    current_episode = []
    disc_metrics = {}  # Store latest disc training metrics
    recent_returns = deque(maxlen=200)  # Track recent episode returns for ranking

    for i in tqdm.tqdm(range(1, FLAGS.online_steps + 1), desc="Online"):
        log_step += 1
        online_rng, key = jax.random.split(online_rng)

        if len(action_queue) == 0:
            action_chunk = np.array(agent.sample_actions(observations=ob, rng=key)).reshape(-1, example_batch["actions"].shape[-1])
            action_queue.extend(action_chunk)
        action = action_queue.pop(0)

        next_ob, int_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        transition = dict(observations=ob, actions=action, rewards=float(int_reward), 
                          terminals=float(done), masks=1.0 - terminated, next_observations=next_ob, is_success=0.0)
        current_episode.append(transition)
        if 'success' in info: logger.log({"success": float(info['success'])}, "env", step=log_step)

        if done:
            is_success_label = 1.0 if float(info.get('success', False)) == 1.0 else 0.0
            episode_return = sum(t['rewards'] for t in current_episode)
            recent_returns.append(episode_return)
            
            # Add ALL transitions to agent replay buffer (as before)
            for t in current_episode:
                t['is_success'] = is_success_label
                agent_replay_buffer.add_transition(t)
            
            # ---- RETURN-RANKED BUFFER MANAGEMENT (v5) ----
            # CRITICAL FIX for cold-start problem:
            # Instead of requiring actual task success (binary), use episode return
            # ranking to fill success/fail buffers. This ensures discriminator
            # ALWAYS has training data, even when no episode has succeeded yet.
            #
            # - Above median return → "better" → success buffer
            # - Below median return → "worse" → fail buffer  
            # - Even with all episodes failing, some are BETTER failures (-1 vs -3)
            tail_steps = min(FLAGS.disc_buffer_tail, len(current_episode))
            tail_transitions = current_episode[-tail_steps:]
            
            if len(recent_returns) >= 20:
                # Enough history: use median return as adaptive threshold
                median_return = float(np.median(list(recent_returns)))
                is_better = (episode_return >= median_return)
            else:
                # Too few episodes: fall back to binary success label
                is_better = (is_success_label == 1.0)
            
            if is_better:
                for t in tail_transitions:
                    success_buffer.add_transition(t)
            else:
                for t in tail_transitions:
                    fail_buffer.add_transition(t)
            
            # Log episode stats
            if i % FLAGS.log_interval == 0:
                logger.log({
                    "disc/episode_return": episode_return,
                    "disc/return_threshold": float(np.median(list(recent_returns))) if len(recent_returns) >= 20 else -999.0,
                    "disc/success_buffer_size": float(success_buffer.size),
                    "disc/fail_buffer_size": float(fail_buffer.size),
                }, "online_agent", step=log_step)
            
            current_episode, action_queue, (ob, _) = [], [], env.reset()
        else: ob = next_ob

        # ====================== DISCRIMINATOR UPDATE (v3) ======================
        disc_ready = (FLAGS.use_discriminator and 
                      i >= FLAGS.start_training and 
                      i % FLAGS.disc_update_interval == 0 and
                      success_buffer.size >= FLAGS.disc_min_buffer and 
                      fail_buffer.size >= FLAGS.disc_min_buffer)
        
        if disc_ready:
            batch_size_half = 128
            for disc_step in range(FLAGS.disc_gradient_steps):
                s_batch = success_buffer.sample(batch_size_half)
                f_batch = fail_buffer.sample(batch_size_half)
                
                online_rng, dropout_key = jax.random.split(online_rng)
                disc_params, disc_opt_state, disc_metrics = update_discriminator_step(
                    disc_params, disc_opt_state,
                    expert_obs=s_batch['observations'],
                    expert_acts=s_batch['actions'],
                    policy_obs=f_batch['observations'],
                    policy_acts=f_batch['actions'],
                    rng_key=dropout_key,
                )
            
            # Log disc training metrics (from last gradient step)
            disc_log = {k: float(v) for k, v in disc_metrics.items()}
            disc_log['disc/success_buffer_size'] = float(success_buffer.size)
            disc_log['disc/fail_buffer_size'] = float(fail_buffer.size)
            logger.log(disc_log, "online_agent", step=log_step)

        # ====================== AGENT UPDATE ======================
        if i >= FLAGS.start_training:
            batch = agent_replay_buffer.sample_sequence(config['batch_size'] * FLAGS.utd_ratio, sequence_length=FLAGS.horizon_length, discount=FLAGS.discount)
            
            # --- PER-STEP ADVERSARIAL REWARD SHAPING (v3) ---
            if FLAGS.use_discriminator:
                can_shape = (success_buffer.size >= FLAGS.disc_min_buffer and 
                             fail_buffer.size >= FLAGS.disc_min_buffer)
                if can_shape:
                    # Adaptive β with warm-up: ramp from 0 → disc_beta over warmup_steps
                    steps_since_start = max(0, i - FLAGS.start_training)
                    warmup_ratio = min(1.0, steps_since_start / max(1, FLAGS.disc_warmup_steps))
                    current_beta = FLAGS.disc_beta * warmup_ratio
                    
                    shaped_rewards, d_prob_mean, r_disc_mean, r_disc_max = compute_shaped_rewards(
                        disc_params, 
                        batch['full_observations'], 
                        batch['actions'], 
                        batch['rewards'],
                        current_beta
                    )
                    batch['rewards'] = np.array(shaped_rewards)
                    
                    if i % FLAGS.log_interval == 0:
                        logger.log({
                            "disc/d_prob_mean": float(d_prob_mean),
                            "disc/r_disc_mean": float(r_disc_mean),
                            "disc/r_disc_max": float(r_disc_max),
                            "disc/beta": float(current_beta),
                            "disc/warmup_ratio": float(warmup_ratio),
                        }, "online_agent", step=log_step)
                else:
                    if i % FLAGS.log_interval == 0:
                        logger.log({
                            "disc/d_prob_mean": 0.5,
                            "disc/r_disc_mean": 0.0,
                            "disc/beta": 0.0,
                        }, "online_agent", step=log_step)
            # ---------------------------------------------

            batch = jax.tree_util.tree_map(lambda x: x.reshape((FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]), batch)
            agent, update_info = agent.batch_update(batch)
            if i % FLAGS.log_interval == 0: logger.log(update_info, "online_agent", step=log_step)

        if FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0:
            eval_info, _, _ = evaluate(agent=agent, env=eval_env, action_dim=example_batch["actions"].shape[-1],
                num_eval_episodes=FLAGS.eval_episodes, num_video_episodes=0, video_frame_skip=3)
            logger.log(eval_info, "eval", step=log_step)

    print("✅ Training completed successfully!")

if __name__ == '__main__': app.run(main)