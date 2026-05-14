# CNN Analysis Report

## Experiment Summary

- Total eksperimen: 16
- Model terbaik berdasarkan validation macro F1: `cnn_2conv_f32-64_k5-5_maxpool`
- Validation macro F1 terbaik: `0.663827`
- Plot loss tersimpan: `16` file di `artifacts/plots/cnn/history`

## Hyperparameter Findings

- Jumlah layer conv terbaik secara rata-rata: `2` layer, mean macro F1 `0.605287`.
- Kombinasi filter terbaik secara rata-rata: `32-64`, mean macro F1 `0.621458`.
- Kombinasi kernel terbaik secara rata-rata: `5-5`, mean macro F1 `0.614431`.
- Pooling terbaik secara rata-rata: `max`, mean macro F1 `0.562706`.

## Evaluation Findings

- Model evaluasi: `cnn_2conv_f32-64_k5-5_maxpool`
- Keras test macro F1: `0.632313` pada `3000` sampel.
- Scratch shared macro F1: `0.096296` pada `64` sampel.
- Scratch non-shared macro F1: `0.096296` pada `64` sampel.
- Agreement scratch shared vs Keras pada prefix yang sama: `1.000000`.
- Agreement scratch non-shared vs Keras pada prefix yang sama: `1.000000`.
- Parameter scratch shared: `62790`.
- Parameter scratch non-shared: `140534662`.
- Rasio parameter non-shared/shared: `2238.17`.

Scratch non-shared di evaluasi ini mengganti Conv2D dengan LocallyConnected2D memakai bobot Conv2D yang direplikasi per posisi. Karena bobotnya direplikasi, prediksi bisa sangat dekat dengan shared, tetapi jumlah parameter jauh lebih besar. Ini menunjukkan parameter sharing lebih efisien untuk pola visual yang berulang di berbagai lokasi gambar.

## Caveats

- Artifact training yang ada memakai `epochs=5` dan `early_stopping_patience=1`, sedangkan `configs/cnn/shared_conv.yaml` saat ini menulis `epochs=20`. Samakan atau jelaskan di laporan.
- Macro F1 test scratch bisa mahal karena forward propagation NumPy menjalankan sliding window eksplisit.
