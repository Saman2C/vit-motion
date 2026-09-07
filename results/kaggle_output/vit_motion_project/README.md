# ViT motion training v0.2.1 — clean 9-column RPM source / m/s model input

本版以乾淨版 `samples.csv` 為唯一資料介面。每個 CSV 必須恰好提供下列模型資料欄位（可有額外欄位，但程式不依賴它們）：

```text
rgb_path
current_body_dx
current_body_dy
current_delta_yaw
left_rpm
right_rpm
next_body_dx
next_body_dy
next_delta_yaw
```

`samples.csv` 仍保留原始 `left_rpm/right_rpm`，不需要改檔。但 inspector 在建立
manifest 時會先用 `m/s = RPM / rpm_to_mps_gain` 轉成 `left_mps/right_mps`。
預設 `rpm_to_mps_gain=4524.0`，可在 config 校正。

因此模型實際的 5 個 numeric inputs 為
`[current_body_dx, current_body_dy, current_delta_yaw, left_mps, right_mps]`，
3 個 targets 為 `[next_body_dx, next_body_dy, next_delta_yaw]`。
Normalization 也只針對這組 model-facing 5+3 欄計算，確保訓練、evaluation 與部署尺度一致。

若 `rgb_path` 的檔名本身是數字 nanosecond timestamp（目前資料即為此格式），
manifest 會額外保留 `rgb_timestamp_ns` 作同步/除錯 metadata。它不送入模型，也不取代
10 Hz 控制時間；因此不需要為了 v0.2.1 重新修改乾淨的 9 欄 `samples.csv`。

## Temporal / RGB 設計

模型保留「1 個快取影像 token + 影像年齡 + K 個連續數值 token」。
預設 `K=6`；`M=image_update_interval` 可獨立調整。RGB 每 M 個控制週期更新一次，
兩次更新之間重用同一張影像；`image_age_steps` 從 0 到 M-1。

資料檢查仍逐張使用 Pillow 驗證 RGB 完整性。乾淨 CSV 沒有舊的 `sample_index`
或 timing metadata，因此 inspector 會在 manifest 內自行產生 `source_row_index`。
若某列因壞圖或非有限數值被排除，K-step window 不會跨過該缺口。

## 設定

```yaml
data:
  manifest_dir: artifacts/manifest
  rgb_column: rgb_path
  rpm_to_mps_gain: 4524.0
  sample_period_sec: 0.1
  image_size: [224, 224]

model:
  sequence_length: 6
  image_update_interval: 4
```

`sample_period_sec` 只用於沒有時間欄位的 evaluation/validation video 時間軸；
目前 10 Hz 資料預設為 0.1 s。

## 1. 建立新的 manifest / normalization

v0.2.1 不可沿用 v0.2.0/v1.3 的舊 manifest，請重新執行：

```bat
python inspect_dataset.py ^
  --config config_CR.yaml ^
  --data-root "\\192.168.111.168\APL_Share\002超級大空間000\YiLin\ViT_data_tracked Vehicle\20260805\processed"
```

成功後會建立 `artifacts\manifest\manifest.csv`、`normalization.json`、
`splits.json` 與 quality report。

## 2. Smoke test

```bat
python smoke_test.py ^
  --manifest-dir artifacts\manifest ^
  --sequence-length 6 ^
  --image-update-interval 4
```

## 3. 訓練

```bat
python train.py --config config_CR.yaml
```

正式訓練的 K/M 以 config 為準。由於 numeric 欄位與 normalization schema 已改成 m/s，
舊版 checkpoint 不應拿來 resume v0.2.1 訓練。

## 4. 單趟 evaluation

```bat
python evaluate_experiment.py ^
  --config config_CR.yaml ^
  --checkpoint artifacts\runs\vit_motion_temporal_cr_v0_2_1\best.pt ^
  --experiment 20260804_morning_dry_001
```

## 5. 驗證影片

```bat
python make_validation_video.py ^
  --config config_CR.yaml ^
  --predictions-csv artifacts\evaluation\20260804_morning_dry_001\predictions.csv
```

影片右側顯示最新 numeric token 的 5 個實際輸入值（履帶速度為 m/s）、3 個 prediction/ground truth，
以及 cached RGB age；左側顯示模型實際使用的快取 RGB。

## 部署介面

`encode_image(image)` 只需在 M-step 更新時呼叫；其他控制週期重用 visual feature。
`predict_from_feature(feature, numeric_sequence, image_age)` 接受 `[B,K,5]` 與
正規化 age。部署端 age 使用 `age_steps / max(M-1, 1)`，須與訓練一致。
