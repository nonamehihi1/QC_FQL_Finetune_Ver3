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

# Discriminator flags
flags.DEFINE_boolean('use_discriminator', False, 'Use discriminator reward shaping')
flags.DEFINE_boolean('use_q_weighting', True, 'Use Q-weighting for L_flow (Actor loss)')
flags.DEFINE_float('disc_beta', 0.2, 'Discriminator penalty scale')
flags.DEFINE_integer('disc_update_interval', 1, 'Discriminator update interval')
flags.DEFINE_integer('disc_median_window', 100, 'Window size for running median of episode returns')

config_flags.DEFINE_config_file('agent', 'agents/acfql.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')
flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")

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
    
    # Modify exp_name based on config
    if FLAGS.use_discriminator:
        exp_name = f"Dis_{FLAGS.disc_beta}_{exp_name}"
        
    run = setup_wandb(project='qc', group=FLAGS.run_group, name=exp_name)
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(get_flag_dict(), f, default=str)

    config = FLAGS.agent
    config['use_q_weighting'] = FLAGS.use_q_weighting

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
        disc_params = disc_model.init(disc_rng, example_batch['observations'][None], example_batch['actions'][None])['params']
        disc_tx = optax.adam(learning_rate=3e-4)
        disc_opt_state = disc_tx.init(disc_params)

        @jax.jit
        def update_discriminator_step(params, opt_state, batch_obs, batch_acts, batch_labels, rng_key):
            def disc_loss_fn(p):
                logits = disc_model.apply({'params': p}, batch_obs, batch_acts, deterministic=False, rngs={'dropout': rng_key})
                return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, batch_labels))
            grads = jax.grad(disc_loss_fn)(params)
            updates, new_opt_state = disc_tx.update(grads, opt_state)
            return optax.apply_updates(params, updates), new_opt_state

        # --- HÀM TỐI ƯU SIÊU TỐC CHO REWARD BẰNG JAX ---
        @jax.jit
        def compute_gail_batch(params, batch_full_obs, batch_acts, batch_rewards, discount, beta):
            B, H = batch_acts.shape[:2]
            flat_obs = batch_full_obs.reshape((B * H, -1))
            flat_acts = batch_acts.reshape((B * H, -1))
            
            d_logits_flat = disc_model.apply({'params': params}, flat_obs, flat_acts, deterministic=True)
            d_probs_flat = jax.nn.sigmoid(d_logits_flat)
            
            # --- PENALTY PER-STEP (không cumsum) ---
            r_discs = -(1.0 - d_probs_flat).reshape((B, H))
            
            new_rewards = batch_rewards + beta * r_discs
            return new_rewards, jnp.mean(r_discs)
        # ----------------------------------------------------------------------

        print("✅ GAIL Discriminator ENABLED (Penalty Mode)")

    prefixes = ["eval", "env", "offline_agent", "online_agent", "disc"]
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

    # ====================== KHỞI TẠO BUFFER ======================
    agent_replay_buffer = ReplayBuffer.create_from_initial_dataset(dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1))
    
    if FLAGS.use_discriminator:
        # --- RETURN-BASED SPLIT: Chia offline data theo median return ---
        # 1. Tính return cho từng episode trong offline dataset
        offline_rewards = np.array(train_dataset['rewards']).squeeze()
        offline_terminals = np.array(train_dataset['terminals']).squeeze()
        terminal_locs = np.nonzero(offline_terminals > 0)[0]
        initial_locs = np.concatenate([[0], terminal_locs[:-1] + 1])
        
        episode_returns = []
        episode_ranges = []
        for start, end in zip(initial_locs, terminal_locs):
            ep_return = float(np.sum(offline_rewards[start:end+1]))
            episode_returns.append(ep_return)
            episode_ranges.append((start, end+1))
        
        # 2. Sắp xếp các episode theo return và chia đôi chính xác 50/50
        sorted_ep_idxs = np.argsort(episode_returns)
        mid_point = len(sorted_ep_idxs) // 2
        
        bad_ep_idxs = set(sorted_ep_idxs[:mid_point])
        good_ep_idxs = set(sorted_ep_idxs[mid_point:])
        
        median_return = float(np.median(episode_returns))
        print(f"📊 Offline dataset: {len(episode_returns)} episodes, median return = {median_return:.2f}")
        
        good_idxs = []
        bad_idxs = []
        for i, (start, end) in enumerate(episode_ranges):
            if i in good_ep_idxs:
                good_idxs.extend(range(start, end))
            else:
                bad_idxs.extend(range(start, end))
        
        # 3. Tạo buffer từ dữ liệu đã phân loại
        def create_buffer_from_indices(indices, dataset_dict, buffer_size):
            subset = {k: np.array(v)[indices] for k, v in dataset_dict.items()}
            buf = ReplayBuffer.create_from_initial_dataset(subset, size=buffer_size)
            return buf
        
        buf_size = max(FLAGS.buffer_size, train_dataset.size + 1)
        success_buffer = create_buffer_from_indices(good_idxs, dict(train_dataset), buf_size)
        fail_buffer = create_buffer_from_indices(bad_idxs, dict(train_dataset), buf_size)
        
        print(f"✅ Success buffer: {success_buffer.size} transitions ({len(good_idxs)} from offline)")
        print(f"❌ Fail buffer: {fail_buffer.size} transitions ({len(bad_idxs)} from offline)")
        
        # 4. Running median tracker cho giai đoạn online
        recent_returns = deque(episode_returns, maxlen=FLAGS.disc_median_window)
        running_median = median_return

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
            # --- RETURN-BASED LABELING ---
            episode_return = sum(t['rewards'] for t in current_episode)
            
            for t in current_episode:
                agent_replay_buffer.add_transition(t)
                if FLAGS.use_discriminator:
                    # Chú ý: Dùng dấu '>' thay vì '>=' để nếu agent liên tục thất bại ở mức điểm sàn (ví dụ -3000)
                    # và median cũng tụt xuống -3000, thì các episode -3000 tiếp theo sẽ bị đẩy vào fail_buffer.
                    if episode_return > running_median or (episode_return == running_median and episode_return > -3000.0):
                        success_buffer.add_transition(t)
                    else:
                        fail_buffer.add_transition(t)
            
            # Cập nhật running median
            if FLAGS.use_discriminator:
                recent_returns.append(episode_return)
                running_median = float(np.median(list(recent_returns)))
            
            current_episode, action_queue, (ob, _) = [], [], env.reset()
        else: ob = next_ob

        # Discriminator Update
        if FLAGS.use_discriminator and i >= FLAGS.start_training and i % FLAGS.disc_update_interval == 0:
            if fail_buffer.size > 128:
                s_batch, f_batch = success_buffer.sample(128), fail_buffer.sample(128)
                batch_obs = np.concatenate([s_batch['observations'], f_batch['observations']], axis=0)
                batch_acts = np.concatenate([s_batch['actions'], f_batch['actions']], axis=0)
                labels = np.concatenate([np.ones((128, 1)), np.zeros((128, 1))], axis=0)
                
                online_rng, dropout_key = jax.random.split(online_rng)
                disc_params, disc_opt_state = update_discriminator_step(disc_params, disc_opt_state, batch_obs, batch_acts, labels, dropout_key)

        # Agent Update
        if i >= FLAGS.start_training:
            target_batch_size = config['batch_size'] * FLAGS.utd_ratio
            batch = agent_replay_buffer.sample_sequence(
                target_batch_size, sequence_length=FLAGS.horizon_length, discount=FLAGS.discount)
            
            # --- CHẤM ĐIỂM REWARD (PENALTY) ĐƯỢC TỐI ƯU SIÊU TỐC ---
            if FLAGS.use_discriminator:
                new_rewards, r_disc_mean = compute_gail_batch(
                    disc_params, 
                    batch['full_observations'], 
                    batch['actions'], 
                    batch['rewards'], 
                    FLAGS.discount, 
                    FLAGS.disc_beta
                )
                batch['rewards'] = np.array(new_rewards)
                logger.log({"r_disc_mean": float(r_disc_mean)}, "disc", step=log_step)
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