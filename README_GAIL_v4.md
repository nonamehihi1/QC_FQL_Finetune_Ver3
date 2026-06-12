# Giải thích phiên bản GAIL v4 (Cập nhật mới nhất)

Phiên bản v4 này là một sự thay đổi lớn về cách tiếp cận, sử dụng các kỹ thuật chuẩn từ GAN (Generative Adversarial Networks) để giải quyết triệt để vấn đề dao động (variance) và bùng nổ điểm số (reward explosion) của các phiên bản trước.

## 1. Các thay đổi cốt lõi trong Code

### A. Centered Logit Reward (Quan trọng nhất)
- **Vấn đề cũ:** Reward cũ (dùng `-log(1-D)`) luôn là số dương. Khi cộng vào môi trường (môi trường triple-cube có reward là `[-3, 0]`), nó làm lệch hẳn thang đo Q-value, khiến agent bị bias hướng lên.
- **Cách v4:** Sử dụng **Centered Logit Reward**: `r_disc = log(D) - log(1 - D)`.
  - Nếu `D = 0.5` (Không phân biệt được Expert/Policy) -> `r_disc = 0`.
  - Nếu `D > 0.5` (Giống Expert) -> `r_disc > 0` (Thưởng).
  - Nếu `D < 0.5` (Giống Policy thất bại) -> `r_disc < 0` (Phạt).
- **Kết quả:** Reward sau đó được ép vào khoảng `[-1, 1]`. Việc có cả thưởng và phạt đối xứng giúp Q-value không bị bùng nổ và phù hợp hoàn hảo với reward âm của môi trường.

### B. WGAN-GP (Gradient Penalty) và Mạng lưới mới
- **Mạng mới:** Bỏ `LayerNorm` (vì nó làm hỏng việc tính gradient độc lập của GP), đổi sang `LeakyReLU` và tăng kích thước mạng lên `(512, 512, 256)`.
- **WGAN-GP:** Thêm một loss phạt độ lớn của gradient. Điều này ép Discriminator thay đổi điểm số một cách mượt mà (Lipschitz continuous), ngăn chặn điểm số bị nhảy vọt quá gắt.

### C. Label Smoothing
- Thay vì ép mạng dự đoán tuyệt đối `1.0` (Thành công) và `0.0` (Thất bại), mạng giờ học mục tiêu `0.9` và `0.1`.
- Việc này giúp model bớt "tự tin thái quá" (overconfident), đặc biệt hữu ích khi nhãn (label) của chúng ta gán cho cả một episode có thể chứa một vài step không thực sự hoàn hảo.

---

## 2. Lệnh chạy chuẩn cho v4

Sử dụng lệnh dưới đây để chạy phiên bản này. Chú ý các tham số mới đã được tối ưu:

```bash
MUJOCO_GL=egl python main.py \
  --run_group=Task4_GAIL_v4_Centered \
  --env_name=cube-triple-play-singletask-task4-v0 \
  --sparse=False \
  --horizon_length=5 \
  --use_discriminator=True \
  --disc_beta=0.5 \
  --disc_update_interval=2000 \
  --disc_gradient_steps=20 \
  --disc_gp_coeff=5.0 \
  --offline_steps=1000000 \
  --online_steps=1000000 \
  --eval_interval=100000
```

### Giải thích tham số mới:
- `--disc_beta=0.5`: Trọng số của Discriminator reward. Do reward giờ nằm trong khoảng `[-1, 1]` (có bù trừ âm dương), ta có thể để beta lớn hơn (ví dụ 0.5 hoặc 1.0) để agent học rõ hơn từ Discriminator mà không sợ hỏng Q-value.
- `--disc_update_interval=2000` & `--disc_gradient_steps=20`: Tối ưu lại cách train. Gom 2000 steps mới train Discriminator một lần với 20 vòng lặp, thay vì update vụn vặt mỗi step.
- `--disc_gp_coeff=5.0`: Hệ số phạt Gradient Penalty chuẩn.
