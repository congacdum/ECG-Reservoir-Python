# Dataset provenance investigation

Phạm vi: chỉ train và validation; test split không được đọc.

## Quan sát xác minh được

- File gốc chứa value_sequence, finalLabel, original_index, original_len, generated_window_id và new_len.
- value_sequence là chuỗi 20 giá trị được dùng làm input.
- finalLabel đã có sẵn trong dataset và được copy vào split artifacts.
- Tên file và metadata có mô tả single_peak_hardneg, nhưng repository không cung cấp đầy đủ source provenance bên ngoài hoặc định nghĩa lâm sàng của nhãn.

## Kết luận

The exact semantic/generation provenance of finalLabel could not be established from the repository. Kết quả baseline rất mạnh của tree/HistGradientBoosting là evidence cần điều tra thêm artifact/morphology; chưa đủ bằng chứng để gọi đây là data leakage.
