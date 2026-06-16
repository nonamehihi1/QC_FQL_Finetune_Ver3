# QC-FQL + AWAC Flow Matching — Version History

## Phiên bản hiện tại: v7 (Advantage-Weighted Flow Matching)

---

## v7 — AWAC Flow Matching (2026-06-16) ⭐ CURRENT

### Thay đổi lớn: Loại bỏ hoàn toàn Discriminator, Áp dụng Trọng số Lợi thế (Advantage-Weighted)

Qua các version từ v3 đến v6, việc cố gắng áp dụng GAIL Discriminator trên một "play-style dataset" (có success rate offline = 0%) đã được chứng minh là không khả thi. Khi Discriminator học từ những thất bại (-1, -2) để đánh giá "đây là thành công", nó tạo ra tín hiệu nhiễu cực lớn và Reward không ổn định (Non-stationary Reward), làm hỏng mạng Critic.
Mặt khác, đối với các bài toán Single-task (không có vector Goal trong Observation), kỹ thuật thay thế Goal (Hindsight Experience Replay - HER) cũng không thể sử dụng được.

**Cách giải quyết (v7): "Thanh lọc bằng Giá trị Cốt lõi (Q-value)"** ✅
```
1. Bỏ hoàn toàn Discriminator, giúp thuật toán chạy nhanh hơn 2 lần và giảm sử dụng RAM/VRAM.
2. Dùng chính mạng Critic để đánh giá từng hành động trong Replay Buffer.
3. Tính Advantage A = Q(buffer) - V(policy_hiện_tại).
4. Áp dụng trọng số W = exp(A / tau) vào hàm Flow Matching Loss.
```
**Kết quả**: 
- Thay vì bắt chước (Clone) mọi thứ trong Replay Buffer một cách mù quáng, mạng Actor giờ đây sẽ **bỏ qua hoàn toàn** những hành động tệ (W ~ 0), và **học cực kỳ mạnh** những hành động có tiềm năng dẫn đến Goal (W cực lớn). 
- Đạt được mục tiêu "Priority Sampling" nhưng bằng một công thức toán học nội tại vững chắc (AWAC) thay vì một Discriminator chắp vá bên ngoài.

### Files thay đổi
- `qc/main.py`:
  - **XÓA SẠCH** mọi cờ (flags), khởi tạo, logic cập nhật và tính điểm của Discriminator.
  - Xóa `success_buffer`, `fail_buffer`. Lấy mẫu thẳng từ `agent_replay_buffer` theo chuẩn gốc.
- `qc/agents/acfql.py`:
  - Trong hàm `actor_loss`, thêm bước tính toán Advantage `A = Q - V`.
  - Tính hàm trọng số Exponential `weights = jnp.exp(A / 3.0)` có chuẩn hóa.
  - Thay thế hàm Behavior Cloning Loss thông thường bằng **Advantage-Weighted Flow Matching Loss**.

### Chạy thử nghiệm
```bash
MUJOCO_GL=egl python main.py \
  --run_group=Task4_AWAC_v7 \
  --env_name=cube-triple-play-singletask-task4-v0 \
  --sparse=False \
  --horizon_length=10 \
  --offline_steps=1000000 \
  --online_steps=1000000 \
  --eval_interval=100000
```
*(Lưu ý: Tăng `--horizon_length=10` như bài báo gốc khuyên dùng cho `cube-triple` sẽ giúp hiệu quả tốt hơn).*

---

## Lịch sử phiên bản cũ (v3 - v6 GAIL)

### v6 — Priority Sampling (2026-06-15)
- Dùng Discriminator để lọc top-25% mẫu từ Oversized Batch.
- Thất bại vì Discriminator bị nhiễu do online buffer thay đổi liên tục.

### v5 — Return-Ranked Buffers (2026-06-15)
- Thử sửa lỗi cold-start bằng xếp hạng median-return. Vẫn bị "kẹo ảo" từ reward shaping.

### v4 — Centered Logit Reward (2026-06-11)
- Bỏ LayerNorm, dùng reward `log(D/(1-D))`.

### v3 — GAIL Discriminator Overhaul (2026-06-11)
- Reward Shaping thuần túy nhưng sụp đổ do LayerNorm phá hỏng hàm GP.