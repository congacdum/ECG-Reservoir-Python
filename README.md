# ECG Reservoir Computing → FPGA

Thiết kế Reservoir Computing / Liquid State Machine cho phân loại nhị phân các cửa sổ ECG dài 20 mẫu, theo hướng fixed-point và có thể ánh xạ sang RTL. Đây là dự án kỹ thuật/nghiên cứu mô phỏng; ý nghĩa y khoa của nhãn 0/1 chưa được xác nhận từ metadata dataset.

~~~text
20 giá trị ECG → 64 neuron sparse LIF reservoir
              → 64 spike-count features
              → INT8 no-bias readout
              → signed score → class 0/1
~~~

Python fixed-point model là hardware golden reference. Brian2 chỉ là exploratory reference. Reservoir, LIF và readout đều không có bias/intercept.

## Kết quả chính hiện tại

Metric reference của locked hardware-oriented model trên validation là Balanced Accuracy = 0.9614.

| Hạng mục | Kết quả đã xác nhận |
|---|---:|
| Locked fixed-point + INT8 reservoir | BA 0.9614 |
| Linear baselines trên RAW20 | BA 0.8282–0.8371 |
| Decision Tree depth 5 trên RAW20 | BA 0.9942 |
| HistGradientBoosting trên RAW20 | BA 0.9973 |
| HistGradientBoosting trên RAW20 + DIFF19 | BA 0.9989 |
| Recurrence Wres ON / OFF | BA 0.9583 / 0.6741 |
| 10-seed reservoir | mean 0.9621, std 0.0074 |
| Expanded RTL E2E | 64 validation samples, 0 score/class mismatch |
| RTL latency | 1386 cycles/sample |
| FPGA synthesis | Chưa có report |

### Kết luận hiện tại

Reservoir vượt các baseline tuyến tính và recurrence có đóng góp rõ ràng. Tuy nhiên, Decision Tree và HistGradientBoosting đạt BA cao hơn trên dataset hiện tại; không thể kết luận reservoir là mô hình tốt nhất.

Functional RTL đã đạt bit-exact equivalence với hardware golden model. Bước còn thiếu lớn nhất là synthesis trên FPGA mục tiêu để đo LUT, FF, BRAM, DSP, timing, Fmax và power thực tế.

## Trạng thái dự án

Đã hoàn thành:

- Group-wise dataset split, NumPy reference và fixed-point FPGA golden model.
- No-bias Ridge readout và INT8 quantization.
- LIF PE, sparse recurrent engine, 20-timestep controller, readout MAC và full classifier RTL.
- Icarus Verilog bit-exact regression.
- Train/validation-only baseline, ablation, seed-stability và robustness analysis.

Chưa hoàn thành:

- Vivado synthesis/implementation.
- Đo LUT, FF, BRAM, DSP, WNS, Fmax, power.
- Target FPGA/board, clock constraint, physical I/O và board deployment.

Repository không claim FPGA optimized, low-resource, real-time hoặc clinical classifier khi chưa có bằng chứng tương ứng.

## Kiến trúc và cấu hình locked

~~~text
20 input samples
      ↓
Q10 fixed-point conversion
      ↓
20 timestep sparse recurrent LIF updates
      ↓
64 spike-count features
      ↓
signed INT8 no-bias MAC
      ↓
20-bit score; score >= 0 → class 1
~~~

| Thành phần | Giá trị |
|---|---|
| Logical neurons / sequence length | 64 / 20 |
| Recurrent graph | 403 ternary edges |
| Input weights | {0.5, 1, 2} |
| Leak / input gain / recurrent gain | 7/8 / 3/2 / 1/16 |
| Threshold / reset | 2.5 / 0 |
| Input / membrane | signed 12-bit Q10 / signed 16-bit Q10 |
| Spike count | unsigned 5-bit, 0…20 |
| Readout / score | signed INT8 / signed 20-bit |
| Bias/intercept | Không dùng |

Các phép toán shift/add, leak 7/8 và recurrent weights ternary được chọn để thuận tiện cho datapath số nguyên. Đây là mapping logic, không phải resource claim.

## Dataset và split discipline

Dataset: data/raw/training_data_20_points_with_single_peak_hardneg.csv

| Thuộc tính | Giá trị |
|---|---:|
| Số cửa sổ / original_index duy nhất | 17,214 / 9,094 |
| Sequence length | 20 |
| Train | 12,059 cửa sổ / 6,365 groups |
| Validation | 2,594 cửa sổ / 1,364 groups |
| Test | 2,561 cửa sổ / 1,365 groups |

Split theo original_index; đã kiểm tra group overlap và exact waveform overlap. Baseline, ablation, seed-stability và robustness mới chỉ dùng train/validation. Final test không được dùng để chọn model, seed, preprocessing hoặc kiến trúc.

## Kết quả model

### Locked validation result

| Metric | Giá trị |
|---|---:|
| Accuracy | 0.9653 |
| Balanced Accuracy | 0.9614 |
| Precision / Sensitivity / Specificity | 0.9541 / 0.9479 / 0.9749 |
| F1 / MCC | 0.9510 / 0.9242 |
| ROC-AUC / PR-AUC | 0.9905 / 0.9630 |

Các metric có context khác:

| Context | Validation BA |
|---|---:|
| Locked fixed-point + INT8 model | 0.9614 |
| Development/model-health run | 0.9602 |
| Ablation, Wres ON, retrained readout | 0.9583 |
| Brian2 exploratory reference | 0.9530 |

Historical sealed-test result của locked architecture là BA 0.9680 trên 2,561 mẫu; đây không phải kết quả của các variant analysis mới.

## Baseline comparison

| Model | Features | Validation BA |
|---|---|---:|
| LogisticRegression | RAW20 | 0.8371 |
| LinearSVC | RAW20 | 0.8295 |
| RidgeClassifier | RAW20 | 0.8282 |
| Decision Tree depth 5 | RAW20 | 0.9942 |
| HistGradientBoosting | RAW20 | 0.9973 |
| HistGradientBoosting | RAW20_DIFF19 | 0.9989 |
| Locked reservoir + INT8 readout | Reservoir spike-count | 0.9614 |

Tree và HistGradientBoosting cao hơn reservoir trên validation. Đây là evidence về cấu trúc morphology/artifact của dataset, chưa đủ để kết luận leakage.

Chi tiết: [baseline_metrics.csv](outputs/analysis/baseline_metrics.csv) và [baseline_summary.md](outputs/analysis/baseline_summary.md).

## Reservoir ablation và stability

Readout được train lại cho từng cấu hình:

| Cấu hình | Validation BA |
|---|---:|
| Wres ON | 0.9583 |
| Wres = 0 | 0.6741 |

Delta ON so với OFF là +0.2843, cho thấy recurrent graph đóng góp đáng kể vào feature representation.

SVD trên spike-count features: 90% = 11, 95% = 24, 99% = 50 components. Prefix subset exploratory của locked graph đạt N=8: 0.9212, N=16: 0.9095, N=32: 0.9110, N=64: 0.9583. Đây không phải graph sweep regenerate độc lập, nên chưa đủ để kết luận N=64 tối ưu hoặc bắt buộc.

10 deterministic seeds: mean 0.9621, standard deviation 0.0074, range 0.9531–0.9736. Có variation giữa các graph ngẫu nhiên.

Chi tiết: [model_v2_recommendation.md](outputs/analysis/model_v2_recommendation.md).

## Robustness

Robustness được đo trên validation bằng locked model:

- Có sensitivity với additive DC offset.
- Gain scaling 0.9 và 1.1 làm BA giảm mạnh.
- Gaussian noise nhỏ ảnh hưởng nhẹ hơn.
- Sigma = 0.02 đưa fixed-point BA xuống khoảng 0.9240.

Đây là sensitivity evidence trên perturbation synthetic, không phải claim về ECG thực. Candidate model-v2 như DC blocker, mean removal, gain normalization hoặc first-difference chưa được chọn.

Chi tiết: [robustness_metrics.csv](outputs/analysis/robustness_metrics.csv).

## Fixed-point, RTL và verification

RTL gồm LIF PE, sparse recurrent engine, reservoir controller, readout MAC và full classifier trong rtl/src/. Python golden model nằm tại src/hardware/hardware_model.py.

Icarus Verilog 13.0 regression:

| Phạm vi | Kết quả |
|---|---|
| Phase 1 LIF: 10,246 vectors / 40,984 comparisons | Zero mismatch |
| Phase 2 recurrent và reservoir-step | Zero mismatch |
| Phase 3: 14 segments / 280 checkpoints | Zero mismatch |
| Phase 4: 14 readout vectors / 896 MAC comparisons | Zero mismatch |
| Original golden bundle | 8 train/validation samples, exact score/class |
| Expanded validation regression | 64 validation samples, 0 score mismatch, 0 class mismatch |

8 mẫu là golden-vector bundle ban đầu; 64 mẫu là expanded validation regression sau đó. Latency đã xác minh là 1320 reservoir + 2 bridge + 64 readout = 1386 cycles/sample.
### Tương thích simulator trong CI

CI production dùng Icarus Verilog 13.0 từ tag v13_0, khóa tại commit 30a7d1a11b7586aa0fc868e509f04f514effc0ad. Cùng source và cùng smoke workload đã được kiểm tra A/B: Icarus 12.0 (Ubuntu apt) biên dịch được nhưng tb_reservoir_step.vvp bị kẹt ở golden simulation; Icarus 13.0 hoàn tất recurrent smoke với zero mismatch. Đây là khác biệt runtime của simulator, không phải thay đổi RTL, model, graph, fixed-point hay golden data. Workflow matrix tạm thời nằm tại .github/workflows/rtl-simulator-compat.yml.

Chi tiết: [BUILD_AUDIT.md](outputs/fpga/reports/BUILD_AUDIT.md).

## FPGA implementation status

Functional RTL đã được verify bit-exact. Physical implementation chưa có:

- synthesis/place-and-route;
- LUT, FF, BRAM, DSP;
- WNS, timing, Fmax;
- power/energy;
- target board và XDC/clock constraints.

RTL hiện mới được xác nhận về functional equivalence, chưa được chứng minh về tài nguyên hoặc timing trên FPGA thực.

Chi tiết: [baseline_synthesis_summary.md](outputs/synthesis/baseline_synthesis_summary.md).

## Cài đặt và cách chạy

Broad development dependencies:

~~~bash
python -m pip install -r requirements.txt
~~~

Environment reproducible đã verify cho CI và verification:

~~~bash
python -m pip install -r requirements-lock.txt
~~~

Python checks:

~~~bash
python -m pytest -q
python -m compileall .
~~~

RTL regression:

~~~bash
python rtl/scripts/run_lif_tb.py
python rtl/scripts/run_recurrent_tb.py
python rtl/scripts/run_controller_tb.py
python rtl/scripts/run_readout_tb.py
python rtl/scripts/run_classifier_tb.py
python rtl/scripts/run_validation_classifier_tb.py
~~~

Analysis artifacts: [outputs/analysis](outputs/analysis/). CI: [.github/workflows/ci.yml](.github/workflows/ci.yml).

## Known limitations

- Ý nghĩa y khoa của nhãn chưa được xác nhận.
- Không có patient_id/record_id để xác minh patient-independent split.
- Chưa có target FPGA, clock contract hoặc physical streaming interface.
- Chưa có synthesis, timing, resource và power measurements.
- Robustness hiện chỉ là synthetic perturbation analysis.
- Model-v2 chưa được chọn.
- Cycle count không phải timing hoặc power result.

## Next steps

1. Chọn FPGA reference target và clock/reset contract.
2. Chạy synthesis baseline hiện tại.
3. Đo LUT, FF, BRAM, DSP, timing và Fmax.
4. Xác định bottleneck bằng report thực tế.
5. Chỉ tối ưu RTL khi có evidence từ synthesis.
6. Re-run bit-exact regression sau mỗi optimization.
7. Đánh giá candidate preprocessing/model-v2 riêng trên train/validation.

## Cấu trúc và tài liệu kỹ thuật

~~~text
ecg_reservoir_fpga/
├── README.md
├── PROJECT_KNOWLEDGE.md
├── CHANGE_LOG.md
├── configs/  data/  src/  scripts/
├── rtl/  tests/  notebooks/
└── outputs/analysis/  outputs/fpga/
~~~

- [PROJECT_KNOWLEDGE.md](PROJECT_KNOWLEDGE.md): locked facts, reproducibility contract và model hashes.
- [CHANGE_LOG.md](CHANGE_LOG.md): lịch sử thay đổi và xác minh.
- [BUILD_AUDIT.md](outputs/fpga/reports/BUILD_AUDIT.md): build/verification audit.
- [Model-v2 recommendation](outputs/analysis/model_v2_recommendation.md).
- [Synthesis status](outputs/synthesis/baseline_synthesis_summary.md).
- [Environment lock](outputs/analysis/requirements-lock.txt).

## License

Repository hiện chưa có tệp LICENSE. Chủ sở hữu cần chọn và thêm giấy phép phù hợp trước khi phân phối công khai.
