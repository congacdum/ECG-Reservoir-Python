# Model v2 recommendation

Phạm vi của báo cáo này chỉ là evidence analysis trên train/validation. Không có model v2 nào được triển khai và final test không được đọc.

## 1. Raw signal so với reservoir

- Linear RAW20 baselines đạt validation Balanced Accuracy khoảng 0.8282–0.8371.
- Linear RAW20_DIFF19 baselines đạt khoảng 0.8283–0.8388.
- Locked fixed-point + INT8 reservoir đạt 0.961410.
- Vì vậy reservoir có lợi thế rõ ràng so với các baseline linear hiện tại.

## 2. Nonlinear baselines

- Decision Tree depth 5 đạt 0.994166 trên RAW20.
- HistGradientBoosting đạt 0.997289 trên RAW20 và 0.998915 trên RAW20_DIFF19.
- Đây là evidence rằng dataset có morphology/artifact phân tách mạnh bằng mô hình tabular đơn giản. Không gọi dataset là leaked vì chưa có bằng chứng leakage.

## 3. Recurrence

Khi retrain no-bias Ridge readout trên cùng fixed-point feature procedure:

- Wres ON: validation BA 0.958323.
- Wres = 0: validation BA 0.674066.
- Delta ON so với OFF: +0.284257.

Recurrence có đóng góp đo được trong ablation này.

## 4. Rank và reservoir size

- 90% variance: 11 components.
- 95% variance: 24 components.
- 99% variance: 50 components.

Prefix subset exploratory của locked graph cho validation BA:

- N=8: 0.921186.
- N=16: 0.909524.
- N=32: 0.911030.
- N=64: 0.958323.

Đây không phải sweep của các graph được regenerate độc lập, nên chưa đủ để thay đổi N=64.

## 5. Seed stability

10 deterministic seeds:

- Mean validation BA: 0.962125.
- Standard deviation: 0.007437.
- Min/max: 0.953108 / 0.973564.

## 6. Robustness

Locked validation model:

- DC offset ±0.01 đã tạo thay đổi đáng kể tùy chiều.
- Gain 0.9 và 1.1 làm BA giảm mạnh.
- Gaussian noise sigma 0.005 chỉ giảm nhẹ; sigma 0.02 giảm validation BA xuống khoảng 0.924.
- Đây là sensitivity evidence, không phải lý do để patch model ngay.

## 7. Recommendation

Giữ locked model và RTL baseline hiện tại cho verification. Candidate model-v2 nên được đánh giá riêng trên train/validation:

- DC blocker hoặc mean removal đơn giản.
- Gain normalization.
- First-difference input.

Không chọn candidate, N, seed hoặc preprocessing bằng final test. Không triển khai model v2 trước khi có experiment report riêng.

