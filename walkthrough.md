# Hướng dẫn chạy các kịch bản thực nghiệm (Ablation Study)

Toàn bộ code khôi phục mạng **Discriminator** (chuyển sang dạng Penalty) và cơ chế **Top-50% Q-Weighting** đã được đẩy lên Github thành công!

> [!NOTE] 
> Nhờ việc thiết kế hệ thống sử dụng các cờ (flags) linh hoạt, bạn không cần phải sửa code mỗi khi đổi phương pháp. Code sẽ tự động gán tên thư mục (exp_name) trên Wandb tương ứng với từng Config để bạn dễ dàng theo dõi.

Dưới đây là các câu lệnh (commands) bạn cần copy-paste để chạy trên server.

## 1. Config A: Good-bad only (Chỉ sử dụng Discriminator Penalty)
Cấu hình này tắt `use_q_weighting`, tức là mạng Flow sẽ học toàn bộ (giống Base), nhưng thuật toán có sử dụng Discriminator để phạt các hành động xấu. 

```bash
# Trên Task 3
python qc/main.py --env_name cube-triple-play-singletask-task3-v0 --use_discriminator=True --use_q_weighting=False --disc_beta=0.2 --seed=1

# Trên Task 4 
python qc/main.py --env_name cube-triple-play-singletask-task4-v0 --use_discriminator=True --use_q_weighting=False --disc_beta=0.2 --seed=1
```

## 2. Config B: Good-bad + Actor Loss (Sử dụng CẢ HAI)
Cấu hình này bật cả 2 tính năng: Vừa phạt hành động xấu bằng Discriminator, vừa ưu tiên học hành động tốt bằng Top-50% Q-Filter trong L_flow. Đây là cấu hình kỳ vọng cho ra kết quả mạnh nhất.

```bash
# Trên Task 3
python qc/main.py --env_name cube-triple-play-singletask-task3-v0 --use_discriminator=True --use_q_weighting=True --disc_beta=0.2 --seed=1

# Trên Task 4
python qc/main.py --env_name cube-triple-play-singletask-task4-v0 --use_discriminator=True --use_q_weighting=True --disc_beta=0.2 --seed=1
```

## 3. Base (Nếu cần so sánh thêm)
Cấu hình gốc của bài báo (tắt toàn bộ).
```bash
python qc/main.py --env cube-triple-play-singletask-task4-v0 --use_discriminator=False --use_q_weighting=False --seed=1
```

> [!TIP]
> Bạn nên cắm chạy Config A và Config B mỗi cái 1 seed (seed = 1) trên màn hình server trước để xem Loss có ổn định và Success Rate có nhích lên không. Kết quả trên Wandb sẽ tự động hiển thị tiền tố `ConfigA_...` hoặc `ConfigB_...` rất dễ nhìn.
