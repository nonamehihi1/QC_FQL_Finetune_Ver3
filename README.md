# QC_FQL_Finetune_Ver3

Lệnh chạy V5
MUJOCO_GL=egl python main.py \
  --run_group=Task4_GAIL_v5 \
  --env_name=cube-triple-play-singletask-task4-v0 \
  --sparse=False \
  --horizon_length=5 \
  --use_discriminator=True \
  --disc_beta=0.2 \
  --offline_steps=1000000 \
  --online_steps=1000000 \
  --eval_interval=100000 \
  --disc_update_interval=2000 \
  --disc_gradient_steps=20 \
  --disc_warmup_steps=100000 \
  --disc_buffer_tail=30 \
  --disc_gp_coeff=5.0 \
  --disc_lr=1e-4
