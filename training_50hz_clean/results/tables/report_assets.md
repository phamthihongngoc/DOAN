# Tai san bao cao 50Hz

## Dataset su dung

- Manifest 50Hz: `G:/DOAN2/data_collection/data/processed_50hz/manifest_50hz.csv`
- Moi trial dai 4 giay, 201 diem thoi gian.
- Chia tap theo subject: train S01-S11, val S12-S13, test S14-S15.

## Bang nen dua vao bao cao

- Bang so sanh model theo Params | GFLOPs | Inference | Accuracy | F1: `training_50hz_clean/results/tables/model_comparison_report_vi.csv`
- Bang tong hop model goc: `training_50hz_clean/results/tables/model_metrics_best_vi.csv`
- Bang so luong mau train/val/test: `training_50hz_clean/results/tables/dataset_split_vi.csv`
- Bang so luong mau va diem IMU train/val/test: `training_50hz_clean/results/tables/dataset_split_samples_vi.csv`
- Bang phan bo mau theo cu chi: `training_50hz_clean/results/tables/gesture_counts_vi.csv`
- Bang Precision/Recall/F1 theo tung hoat dong: `training_50hz_clean/results/tables/per_activity_f1_score_vi.csv`
- Bang Accuracy nhan dang theo tung cu chi: `training_50hz_clean/results/tables/per_activity_accuracy_vi.csv`
- Bang history huan luyen CNN theo epoch: `training_50hz_clean/results/tables/cnn_training_history_curve.csv`
- Bang so mau IMU theo nguoi thuc hien: `training_50hz_clean/results/tables/subject_imu_samples_vi.csv`

## Hinh nen dua vao bao cao

- Bieu do Accuracy/F1 theo model: `training_50hz_clean/results/plots/model_accuracy_f1_comparison.png`
- Bieu do Accuracy theo model: `training_50hz_clean/results/plots/accuracy_by_model.png`
- Bieu do Macro F1 theo model: `training_50hz_clean/results/plots/f1_by_model.png`
- Bieu do so luong mau theo cu chi: `training_50hz_clean/results/plots/gesture_counts.png`
- Bieu do so luong mau theo tap train/val/test: `training_50hz_clean/results/plots/split_counts.png`
- Bieu do so sanh F1-score theo tung hoat dong: `training_50hz_clean/results/plots/per_activity_f1_score.png`
- Bieu do so sanh Accuracy nhan dang theo tung cu chi: `training_50hz_clean/results/plots/per_activity_accuracy.png`
- Bieu do ham mat mat va do chinh xac CNN theo epoch: `training_50hz_clean/results/plots/cnn_training_loss_accuracy_curve.png`
- Bieu do so mau IMU theo nguoi thuc hien: `training_50hz_clean/results/plots/subject_imu_samples.png`
- Bieu do so trial theo nguoi thuc hien: `training_50hz_clean/results/plots/subject_trial_counts.png`

## Confusion matrix cho model chinh

- CNN: `training_50hz_clean/results/confusion/confusion_cnn.png`
- LSTM: `training_50hz_clean/results/confusion/confusion_lstm.png`
- Transformer: `training_50hz_clean/results/confusion/confusion_transformer.png`

## t-SNE cho model hoc sau

- CNN: `training_50hz_clean/results/tsne/tsne_cnn.png`
- LSTM: `training_50hz_clean/results/tsne/tsne_lstm.png`
- Transformer: `training_50hz_clean/results/tsne/tsne_transformer.png`

## Goi y su dung trong bao cao

- Dung bang so sanh model de trinh bay do chinh xac, F1, do tre suy dien va do phuc tap.
- Dung confusion matrix de phan tich cac lop/cu chi de nham lan.
- Dung t-SNE de minh hoa kha nang tach lop cua dac trung hoc duoc.
- Dung cac bieu do phan bo du lieu de chung minh tap du lieu can bang va cach chia train/val/test.

