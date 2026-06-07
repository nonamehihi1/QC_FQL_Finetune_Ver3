import glob, tqdm, wandb, os, json, random, time, jax
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

# === GAIL FLAGS ===
flags.DEFINE_bool('use_discriminator', True, "Enable GAIL Discriminator")
flags.DEFINE_float('disc_beta', 0.05, "Weight of discriminator reward")
flags.DEFINE_integer('disc_update_interval', 5000, "Train discriminator every N steps")

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

    if FLAGS.use_discriminator:
        from models.discriminator import PerStepDiscriminator
        disc_model = PerStepDiscriminator()
        disc_rng = jax.random.PRNGKey(FLAGS.seed + 999)
        # Init với per-step input: (obs, action)
        example_action = np.zeros((1, example_batch['actions'].shape[-1]))
        disc_params = disc_model.init(disc_rng, example_batch['observations'][None], example_action)['params']
        disc_tx = optax.adam(learning_rate=3e-4)
        disc_opt_state = disc_tx.init(disc_params)

        @jax.jit
        def update_discriminator_step(params, opt_state, batch_obs, batch_acts, batch_labels, rng_key):
            """Train discriminator trên per-step transitions: (obs, action)."""
            def disc_loss_fn(p):
                pred = disc_model.apply({'params': p}, batch_obs, batch_acts, deterministic=False, rngs={'dropout': rng_key})
                return -jnp.mean(batch_labels * jnp.log(pred + 1e-8) + (1 - batch_labels) * jnp.log(1 - pred + 1e-8))
            grads = jax.grad(disc_loss_fn)(params)
            updates, new_opt_state = disc_tx.update(grads, opt_state)
            return optax.apply_updates(params, updates), new_opt_state

        # --- PER-STEP ADVERSARIAL REWARD SHAPING ---
        discount_powers = jnp.power(FLAGS.discount, jnp.arange(FLAGS.horizon_length))

        @jax.jit
        def compute_shaped_rewards(params, batch_obs, batch_acts, batch_rewards, beta):
            """Shape sequence cumulative rewards step-by-step using per-step discriminator."""
            # Predict scores for all steps in the sequence
            d_probs = disc_model.apply({'params': params}, batch_obs, batch_acts, deterministic=True)  # (B, H, 1)
            scores = d_probs.squeeze(-1)  # (B, H)
            
            # Compute step-by-step discriminator reward
            r_disc = -jnp.log(1.0 - scores + 1e-8)
            r_disc = jnp.clip(r_disc, 0.0, 5.0) / 5.0  # (B, H)
            
            # Cumulative discounted discriminator reward
            discounted_r_disc = r_disc * discount_powers[None, :]  # (B, H)
            cum_r_disc = jnp.cumsum(discounted_r_disc, axis=1)     # (B, H)
            
            # Shape the environment cumulative rewards
            shaped_rewards = batch_rewards + beta * cum_r_disc
            return shaped_rewards, jnp.mean(scores)

        print("✅ Per-step Discriminator ENABLED — Adversarial Reward Shaping")

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
            
            for t in current_episode:
                t['is_success'] = is_success_label
                agent_replay_buffer.add_transition(t)
                if is_success_label == 1.0: success_buffer.add_transition(t)
                else: fail_buffer.add_transition(t)
            
            current_episode, action_queue, (ob, _) = [], [], env.reset()
        else: ob = next_ob

        # Discriminator Update
        if FLAGS.use_discriminator and i >= FLAGS.start_training and i % FLAGS.disc_update_interval == 0:
            if success_buffer.size >= 128 and fail_buffer.size >= 128:
                for _ in range(10):  # Train discriminator properly with multiple steps
                    s_batch = success_buffer.sample(128)
                    f_batch = fail_buffer.sample(128)
                    
                    batch_obs = np.concatenate([s_batch['observations'], f_batch['observations']], axis=0)  # (256, obs_dim)
                    batch_acts = np.concatenate([s_batch['actions'], f_batch['actions']], axis=0)  # (256, action_dim)
                    labels = np.concatenate([np.ones((128, 1)), np.zeros((128, 1))], axis=0)
                    
                    online_rng, dropout_key = jax.random.split(online_rng)
                    disc_params, disc_opt_state = update_discriminator_step(disc_params, disc_opt_state, batch_obs, batch_acts, labels, dropout_key)

        # Agent Update
        if i >= FLAGS.start_training:
            batch = agent_replay_buffer.sample_sequence(config['batch_size'] * FLAGS.utd_ratio, sequence_length=FLAGS.horizon_length, discount=FLAGS.discount)
            
            # --- PER-STEP ADVERSARIAL REWARD SHAPING ---
            if FLAGS.use_discriminator:
                if success_buffer.size >= 128 and fail_buffer.size >= 128:
                    shaped_rewards, d_prob_mean = compute_shaped_rewards(
                        disc_params, 
                        batch['full_observations'], 
                        batch['actions'], 
                        batch['rewards'],
                        FLAGS.disc_beta
                    )
                    batch['rewards'] = np.array(shaped_rewards)
                    logger.log({"disc/d_prob_mean": float(d_prob_mean)}, "online_agent", step=log_step)
                else:
                    logger.log({"disc/d_prob_mean": 0.5}, "online_agent", step=log_step)
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