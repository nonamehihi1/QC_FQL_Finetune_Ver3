# QC-FQL + Discriminator — Version History

## Phiên bản hiện tại: v6 (Priority Sampling)

---

## v6 — Priority Sampling (2026-06-15) ⭐ CURRENT

### Thay đổi lớn: Chuyển hoàn toàn từ Cách 1 → Cách 2

**Cách 1 (v3-v5): "Ông gia sư nhiễu sự" — Reward Shaping**
```
Disc cho kẹo (cộng disc reward) → Agent hack kẹo thay vì giải toán
→ Q-values bị "ngộ độc kẹo ảo" → Kết quả dao động, không ổn định
```

**Cách 2 (v6): "Ông gia sư tuyển chọn" — Priority Sampling** ✅
```
1. Agent làm 1024 bài nháp (sample oversized batch)
2. Disc xếp hạng 1024 bài từ cao→thấp (score sequences)
3. Chọn TOP 256 bài sáng sủa nhất (filter top-K)
4. Agent phân tích rút kinh nghiệm 256 bài đó (train on filtered batch)
5. Reward 100% từ environment — KHÔNG bị disc sửa đổi!
```

### So sánh kỹ thuật

| | Cách 1 (v3-v5) | Cách 2 (v6) |
|---|---|---|
| Disc làm gì? | **Cộng/trừ** reward | **Lọc** mẫu tốt nhất |
| Agent thấy reward gì? | env + disc (bị biến đổi) | **env gốc 100%** |
| Q-values? | Bị corrupt | **Sạch** |
| Rủi ro hack? | Agent hack disc reward | **Không** — disc không ảnh hưởng reward |

### Files thay đổi
- `qc/main.py`:
  - Xóa `compute_shaped_rewards()` (reward shaping)
  - Thêm `score_sequences()` (scoring only)
  - Agent update: oversample → filter → train trên reward gốc
  - Xóa flags: `disc_beta`, `disc_warmup_steps`
  - Thêm flag: `disc_oversample_ratio` (default=4, sample 4x lọc top 25%)

### Chạy thử nghiệm
```bash
MUJOCO_GL=egl python main.py \
  --run_group=Task4_GAIL_v6 \
  --env_name=cube-triple-play-singletask-task4-v0 \
  --sparse=False \
  --horizon_length=5 \
  --use_discriminator=True \
  --disc_oversample_ratio=4 \
  --offline_steps=1000000 \
  --online_steps=1000000 \
  --eval_interval=100000 \
  --disc_update_interval=2000 \
  --disc_gradient_steps=20 \
  --disc_gp_coeff=5.0 \
  --disc_lr=1e-4
```

---

## Lịch sử phiên bản cũ

### v5 — Return-Ranked Buffers (2026-06-15)
- Fix cold-start: thay binary success/fail bằng median return ranking
- Vẫn dùng Cách 1 (reward shaping) → vẫn bị vấn đề "kẹo ảo"

### v4 — Centered Logit Reward (2026-06-11)
- Bỏ LayerNorm (phá GP), đổi reward thành centered logit
- Kết quả: 0-0.5 (seed-dependent), 1 seed vẫn dính 0

### v3 — GAIL Discriminator Overhaul (2026-06-11)
- Disc lớn hơn (512,512,256), GP, Label Smoothing, Adaptive β
- Kết quả: success = 0 (LayerNorm + positive bias)

---

## Flags hiện tại (v6)

| Flag | Default | Mô tả |
|------|---------|-------|
| `--use_discriminator` | True | Bật/tắt discriminator |
| `--disc_oversample_ratio` | 4 | Sample Nx, giữ top 1/N |
| `--disc_update_interval` | 2000 | Update disc mỗi N steps |
| `--disc_gradient_steps` | 20 | Gradient steps mỗi lần update |
| `--disc_lr` | 1e-4 | Learning rate disc |
| `--disc_gp_coeff` | 5.0 | Gradient Penalty coefficient |
| `--disc_buffer_tail` | 30 | Bước cuối episode cho disc buffer |
| `--disc_min_buffer` | 128 | Min buffer size trước khi disc train |

---

## Kết quả thực nghiệm (Cube_TripleTask_Task4)

| Phương pháp | Cách | Success Rate | Ổn định? | Ghi chú |
|---|---|---|---|---|
| Base (không Disc) | — | ~0.6 max | ✅ | Giới hạn exploration |
| Cách 1 gốc (β=0.1/0.2) | Reward Shaping | Max 0.7-0.8 | ❌ | Label noise, dao động |
| v3 | Reward Shaping | 0 | ❌ | LayerNorm + positive bias |
| v4 | Reward Shaping | 0-0.5 | ❌ | Cold-start problem |
| v5 | Reward Shaping | — | — | Return-ranked, vẫn reward shaping |
| **v6** | **Priority Sampling** | ⏳ | — | **Không sửa reward, chỉ lọc mẫu** |