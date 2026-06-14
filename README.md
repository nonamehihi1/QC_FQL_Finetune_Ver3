# QC-FQL + GAIL Discriminator — Version History

## Phiên bản hiện tại: v5 (Return-Ranked Buffers)

---

## v5 — Return-Ranked Buffers (2026-06-15)

### Vấn đề v4 giải quyết chưa xong
Cùng code, cùng tham số nhưng **seed khác nhau cho kết quả chênh lệch lớn**: 1 seed lên 0.5, 1 seed dính 0. 
**Nguyên nhân**: Cold-start — discriminator cần `success_buffer >= 128` mới bắt đầu train, nhưng seed xấu không bao giờ thành công → disc không bao giờ được train → vòng lặp chết.

### Fix
Thay binary success/fail bằng **return-ranked**:
- Track 200 episode returns gần nhất
- Tính **median return** làm ngưỡng
- Episode return ≥ median → success buffer ("tốt hơn")
- Episode return < median → fail buffer ("xấu hơn")
- **Disc luôn có data để train**, kể cả khi chưa episode nào succeed

### Files thay đổi
- `qc/main.py` — Thêm `from collections import deque`, đổi logic buffer management

### Chạy thử nghiệm
```bash
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
```

---

## v4 — Centered Logit Reward (2026-06-11)

### Vấn đề v3
Chạy v3 → eval/success = 0 hoàn toàn. 2 nguyên nhân:

1. **LayerNorm trong Discriminator** phá Gradient Penalty — gradients qua LayerNorm trên interpolated samples bị unstable
2. **Disc reward `-log(1-D)` luôn dương** [0,∞) nhưng env reward ∈ [-3, 0]. Khi disc chưa hội tụ (D≈0.5), thêm constant positive bias → Q-values sai

### Fix
1. **Bỏ LayerNorm** — giữ `Dense → LeakyReLU → Dropout` (chuẩn WGAN-GP)
2. **Centered logit reward**: `log(D/(1-D))` thay vì `-log(1-D)`
   - D ≈ 0.5 → reward ≈ 0 (không bias!)
   - D > 0.5 (expert-like) → reward > 0
   - D < 0.5 (policy-like) → reward < 0

### Files thay đổi
- `qc/models/discriminator.py` — Bỏ LayerNorm
- `qc/main.py` — Đổi reward formula

---

## v3 — GAIL Discriminator Overhaul (2026-06-11)

### Thay đổi so với v2 (code gốc)
1. **Discriminator lớn hơn**: (256,256) → (512,512,256) + LeakyReLU
2. **Flatten approach**: (B,H) → (B*H) giống Cách 1 cho per-step evaluation
3. **Gradient Penalty** (λ=5.0) trên interpolated samples
4. **Label Smoothing** (0.9/0.1) giảm overconfidence từ label noise
5. **Adaptive β warm-up**: 0 → disc_beta qua 100k steps
6. **Buffer tail**: Chỉ lấy 30 bước cuối episode
7. **Disc hyperparams**: interval=2000, lr=1e-4, grad_steps=20

### Files thay đổi
- `qc/models/discriminator.py` — Kiến trúc mới
- `qc/main.py` — Toàn bộ GAIL logic

---

## Flags mới (v3+)

| Flag | Default | Mô tả |
|------|---------|-------|
| `--disc_beta` | 0.1 | Trọng số disc reward |
| `--disc_update_interval` | 2000 | Update disc mỗi N steps |
| `--disc_gradient_steps` | 20 | Số gradient steps mỗi lần update disc |
| `--disc_lr` | 1e-4 | Learning rate của disc |
| `--disc_gp_coeff` | 5.0 | Gradient Penalty coefficient |
| `--disc_warmup_steps` | 100000 | β warm-up period |
| `--disc_buffer_tail` | 30 | Số bước cuối episode cho disc buffer |
| `--disc_min_buffer` | 128 | Tối thiểu buffer size trước khi disc train |

---

## Kết quả thực nghiệm (Cube_TripleTask_Task4)

| Phương pháp | Success Rate | Ổn định? | Ghi chú |
|---|---|---|---|
| Base (không Disc) | ~0.6 max | ✅ | Giới hạn exploration |
| Cách 1 (per-action, β=0.1/0.2) | Max 0.7-0.8 | ❌ Dao động | Label noise, Dis overfit |
| Cách 2 (cumsum, code gốc) | 0.1-0.4 | ❌ | Reward scale sai |
| v3 | 0 | ❌ | LayerNorm + positive bias |
| v4 | 0-0.5 (seed-dependent) | ❌ | Cold-start problem |
| v5 | ⏳ Đang chạy | — | Return-ranked fix |