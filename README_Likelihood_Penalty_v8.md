# Khắc phục lỗi ở Task 4 bằng Likelihood Penalty (Lấy cảm hứng từ DRIFT)

## Vấn đề ở các phiên bản trước (v4 - v7)
Trong các phiên bản trước, thuật toán sử dụng **GAIL Discriminator** để phân loại quỹ đạo thành công (`success`) và thất bại (`fail`), từ đó sinh ra hàm phạt (penalty) bổ sung vào hàm phần thưởng gốc của môi trường. Cách này chạy khá ổn ở Task 3.

Tuy nhiên, khi áp dụng cho **Task 4 (quỹ đạo dài hơn, phần thưởng thưa thớt hơn rất nhiều)**, Discriminator gặp phải hai vấn đề lớn:
1. **Sự cố cạn kiệt (Starvation) của Success Buffer:** Ở những bước đầu, agent gần như không đạt được "success" thực sự. Phân loại theo "median return" có thể chỉ là so sánh giữa các lần thất bại, khiến Discriminator bị hội tụ sai (mode collapse).
2. **Bi quan thái quá (Over-pessimism):** Quỹ đạo càng dài, khả năng agent chệch khỏi dữ liệu offline càng cao (OOD). Discriminator phạt điểm quá lớn ở phần cuối quỹ đạo, khiến RL agent không dám khám phá tiếp và bị mắc kẹt.
3. Mặc dù là một ý tưởng hay, nhưng Discriminator chưa tác động thay đổi bản chất hàm Loss của RL về mặt toán học, dẫn tới sự khó thuyết phục khi đánh giá học thuật.

## Giải Pháp: Likelihood Penalty (DRIFT)
Để thay thế Discriminator, phiên bản v8 này sử dụng trực tiếp **Flow Matching Model (Offline)** để phạt các hành động đi quá xa khỏi dữ liệu offline (OOD).

Cụ thể, hàm Loss của Critic (Q-learning) được thêm một hằng số phạt (penalty), biến đổi phương trình Bellman thành:
$$Target\_Q = R + \gamma \cdot \left[ Q(s', a') - \alpha \cdot NLL(a'|s') \right]$$

Vì Flow Matching Model rất khó tính Log-Probability chính xác (cần tính tích phân Jacobian), phương pháp này sử dụng **MSE (Mean Squared Error) của luồng dự đoán vận tốc** làm proxy đại diện cho giá trị NLL (Negative Log-Likelihood).

### Chi tiết thay đổi Code:
- **`qc/agents/acfql.py`**:
  - Dùng `actor_bc_flow` (Offline Flow Model - không có gradient) để đoán lại vận tốc cho các hành động `next_actions`.
  - Tính toán `nll_surrogate = MSE(pred_vel, vel)`.
  - Cập nhật `target_q` bằng cách trừ đi `alpha_penalty * nll_surrogate`.
- **`qc/main.py`**:
  - Tắt cờ `use_discriminator` mặc định.
  - Thêm cờ `--alpha_penalty` (mặc định = 1.0).
  - Tên experiment (để log lên Wandb) giờ đây sẽ tự động thêm chữ `LikelihoodPenalty` nếu giá trị `alpha_penalty > 0`.

## Cách khởi chạy huấn luyện Task 4
Sử dụng cờ `--alpha_penalty` để điều chỉnh độ lớn của hàm phạt. Mức `1.0` là điểm xuất phát tốt.
```bash
python qc/main.py --env_name=cube-quadruple-task4-v0 --alpha_penalty=1.0 --use_q_weighting=True
```
