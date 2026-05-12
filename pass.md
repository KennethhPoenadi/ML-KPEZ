# Checklist Kesesuaian Tugas Tubes 2

Audit ini dibuat dari kondisi repo lokal saat ini. Status `sudah` berarti implementasi/artifact-nya terlihat ada di repo; status `belum` berarti belum ada artifact hasil jalan/evaluasi, belum lengkap untuk laporan, atau masih berupa kode yang belum dieksekusi penuh.

## Ringkasan Cepat

- [x] Implementasi utama CNN sudah ada.
- [x] Eksperimen CNN 16 arsitektur sudah punya metadata, history, model, dan weights.
- [x] Evaluasi CNN terbaik, plot loss, dan analisis CNN sudah tersimpan sebagai artifact.
- [ ] Evaluasi scratch CNN full test belum dijalankan; artifact saat ini memakai `scratch_max_batches=1` untuk smoke run.
- [x] Implementasi kode RNN/LSTM captioning sudah ada.
- [x] Dataset Flickr8k lokal sudah ada dan valid: 8.091 image, 40.455 caption record, 5 caption per image.
- [x] Preprocessing caption sudah dijalankan dan artifact split/vocabulary sudah ada.
- [x] Feature extraction Flickr8k sudah selesai dengan InceptionV3, 8.091 feature vector shape 2.048.
- [ ] Training 12 variasi RNN/LSTM, evaluasi BLEU-4/METEOR, prediksi, dan analisis captioning belum selesai.
- [x] File laporan/ringkasan CNN sudah ada di `artifacts/reports`.

## CNN

### Bagian 1: Utility Functions

- [x] Image loader menggunakan PIL, resize, konversi NumPy, normalisasi `[0, 1]`.
  - File: `src/tubes2_ml/cnn/data.py`
  - Fungsi: `load_image`
- [x] Batch loader menghasilkan array `(N, H, W, C)`.
  - File: `src/tubes2_ml/cnn/data.py`
  - Fungsi: `load_image_batch`
- [x] Feature extractor menerima list path, memakai encoder Keras frozen, dan menyimpan `.npy`.
  - File: `src/tubes2_ml/cnn/feature_extraction.py`
  - Fungsi: `extract_features_to_npy`
- [ ] Artifact feature extraction CNN belum terlihat di `data/features/cnn`.

### Bagian 2: Forward Propagation From Scratch

- [x] `Conv2D` shared parameter dari scratch.
  - File: `src/tubes2_ml/scratch/layers/conv.py`
- [x] `LocallyConnected2D` non-shared parameter dari scratch.
  - File: `src/tubes2_ml/scratch/layers/locally_connected.py`
- [x] `MaxPooling2D` dan `AveragePooling2D`.
  - File: `src/tubes2_ml/scratch/layers/pooling.py`
- [x] `GlobalAveragePooling2D` dan `GlobalMaxPooling2D`.
  - File: `src/tubes2_ml/scratch/layers/pooling.py`
- [x] `Flatten` row-major / C order.
  - File: `src/tubes2_ml/scratch/layers/flatten.py`
- [x] Aktivasi `relu`, `softmax`, `sigmoid`, `tanh`, `linear`.
  - File: `src/tubes2_ml/scratch/layers/activations.py`
- [x] `Dense` dari scratch dan loader bobot Keras.
  - File: `src/tubes2_ml/scratch/layers/dense.py`
- [x] Builder scratch CNN dari model Keras.
  - File: `src/tubes2_ml/scratch/models/cnn_classifier.py`
  - Mendukung mode shared dan penggantian Conv2D menjadi LocallyConnected2D.

### Bagian 3: Pelatihan Model

- [x] Model Keras CNN Conv2D shared parameter tersedia.
  - File: `src/tubes2_ml/cnn/models.py`
  - Loss: `SparseCategoricalCrossentropy`
  - Optimizer: `Adam`
- [x] Grid hyperparameter 16 eksperimen tersedia.
  - File: `configs/cnn/hparam_grid.yaml`
  - Variasi: 2 jumlah layer, 2 kombinasi filter, 2 ukuran kernel, 2 pooling.
- [x] Script training grid tersedia.
  - File: `scripts/run_cnn_experiments.py`
- [x] 16 artifact eksperimen CNN sudah ada.
  - Folder: `artifacts/experiments/cnn`
- [x] Bobot dan full model 16 CNN sudah tersimpan.
  - Folder: `models/keras/cnn`
- [x] Macro F1 validation sudah tersimpan di metadata tiap eksperimen.
  - Best validation macro F1 saat audit: `cnn_2conv_f32-64_k5-5_maxpool` = `0.663827`.
- [ ] Config aktif `configs/cnn/shared_conv.yaml` menulis `epochs: 20`, tetapi artifact yang ada dilatih dengan `epochs: 5` dan `early_stopping_patience: 1`. Ini perlu dijelaskan di laporan atau disamakan.

### Bagian 4: Eksperimen dan Evaluasi

- [x] Fungsi evaluasi Keras dan scratch pada test split tersedia.
  - File: `src/tubes2_ml/cnn/evaluate.py`
- [x] Notebook CNN punya cell untuk memilih model terbaik dan menjalankan Keras vs scratch.
  - File: `notebooks/cnn/CNN.ipynb`
- [ ] Notebook CNN belum dieksekusi; semua code cell `execution_count = null`.
- [x] Script evaluasi model terbaik tersedia.
  - File: `scripts/evaluate_cnn_best.py`
- [x] Hasil test macro F1 Keras dan smoke metric scratch shared tersimpan.
  - File: `artifacts/experiments/cnn/best_model_evaluation.json`
  - Catatan: Keras full test, scratch shared memakai `scratch_max_batches=1`.
- [x] Hasil smoke scratch non-shared `LocallyConnected2D` pada arsitektur terbaik tersimpan.
  - File: `artifacts/experiments/cnn/best_model_evaluation.json`
- [x] Jumlah parameter shared vs non-shared tersimpan.
  - File: `artifacts/experiments/cnn/best_model_evaluation.json`
- [x] Prediksi evaluasi CNN tersimpan.
  - File: `artifacts/predictions/cnn/best_model_predictions.npz`
- [x] Grafik training/validation loss untuk semua 16 eksperimen sudah dibuat.
  - Folder: `artifacts/plots/cnn/history`
- [x] Analisis pengaruh hyperparameter sudah ditulis sebagai laporan/ringkasan.
  - File: `artifacts/reports/cnn_analysis.md`
  - File: `artifacts/reports/cnn_hparam_summary.csv`
  - File: `artifacts/reports/cnn_hparam_summary.json`
- [x] Analisis shared vs non-shared sudah ditulis sebagai laporan/ringkasan.
  - File: `artifacts/reports/cnn_analysis.md`
- [ ] Evaluasi scratch shared dan non-shared full test masih perlu dijalankan ulang tanpa `--scratch-max-batches`.

## RNN/LSTM Image Captioning

### Bagian 0: Forward Propagation From Scratch

- [x] Embedding layer dari scratch.
  - File: `src/tubes2_ml/scratch/layers/embedding.py`
- [x] SimpleRNN cell dari scratch.
  - File: `src/tubes2_ml/scratch/layers/recurrent.py`
- [x] LSTM cell dari scratch dengan format bobot Keras `kernel`, `recurrent_kernel`, `bias`.
  - File: `src/tubes2_ml/scratch/layers/recurrent.py`
- [x] Dense projection dan Dense output dari scratch.
  - File: `src/tubes2_ml/scratch/layers/dense.py`
- [x] Scratch RNN captioner dari bobot Keras.
  - File: `src/tubes2_ml/scratch/models/rnn_captioner.py`
- [x] Scratch LSTM captioner dari bobot Keras.
  - File: `src/tubes2_ml/scratch/models/lstm_captioner.py`
- [x] Notebook pernah melakukan parity smoke test kecil RNN/LSTM dengan selisih sekitar `1e-8`.
  - File: `notebooks/captioning/RNN.ipynb`

### Bagian 1: Feature Extraction CNN Encoder Frozen

- [x] Kode feature extraction Flickr8k memakai InceptionV3 atau VGG16 pretrained ImageNet, `include_top=False`, `pooling="avg"`, dan frozen.
  - File: `src/tubes2_ml/captioning/feature_extraction.py`
- [x] Script extraction tersedia.
  - File: `scripts/extract_caption_features.py`
- [x] Data Flickr8k lokal sudah ada di path yang benar.
  - `data/raw/flickr8k/images`: 8.091 image.
  - `data/raw/flickr8k/captions/captions.txt`: 40.455 caption record.
  - Validasi: semua caption image id punya file gambar, semua gambar punya caption, min/max caption per image = 5/5.
- [x] Artifact feature extraction lokal sudah ada.
  - `data/features/captioning/features.npy`
  - `data/features/captioning/image_ids.json`
  - `data/features/captioning/metadata.json`
  - Hasil: 8.091 feature vector, shape `(8091, 2048)`, encoder `inception_v3`.
- [x] Feature extraction captioning sudah reuse utility image CNN.
  - File: `src/tubes2_ml/captioning/feature_extraction.py`

### Bagian 2: Preprocessing Caption

- [x] Cleaning lowercase dan hapus punctuation menggunakan `str.lower()` dan `re.sub()`.
  - File: `src/tubes2_ml/captioning/preprocessing.py`
- [x] Tokenisasi menggunakan `str.split()`.
  - File: `src/tubes2_ml/captioning/preprocessing.py`
- [x] Vocabulary dari caption training dengan special tokens `<pad>`, `<start>`, `<end>`, `<unk>`.
  - File: `src/tubes2_ml/captioning/preprocessing.py`
- [x] Padding sequence menggunakan NumPy.
  - File: `src/tubes2_ml/captioning/preprocessing.py`
- [x] Split default `6000 / 1000 / 1000` tersedia.
  - File: `src/tubes2_ml/captioning/preprocessing.py`
- [x] Script preprocessing tersedia.
  - File: `scripts/preprocess_captions.py`
- [x] Artifact preprocessing lokal sudah ada.
  - `data/processed/captioning/vocabulary.json`
  - `data/processed/captioning/metadata.json`
  - `data/processed/captioning/train.npz`
  - `data/processed/captioning/validation.npz`
  - `data/processed/captioning/test.npz`
  - Hasil: vocab size 7.464, max caption length 38, split 30.000 train / 5.000 validation / 5.000 test captions.

### Bagian 3: Pelatihan Decoder Keras

- [x] Arsitektur decoder pre-inject sudah diimplementasikan.
  - File: `src/tubes2_ml/captioning/models.py`
  - CNN feature diproyeksikan Dense ke `embed_dim`, reshape sebagai timestep awal, lalu concat dengan embedding token.
- [x] Mendukung `SimpleRNN` dan `LSTM`.
  - File: `src/tubes2_ml/captioning/models.py`
- [x] Loss `SparseCategoricalCrossentropy` dan optimizer `Adam`.
  - File: `src/tubes2_ml/captioning/models.py`
- [x] Teacher forcing dengan input sequence dan target sequence bergeser.
  - File: `src/tubes2_ml/captioning/preprocessing.py`
- [x] Grid 12 eksperimen tersedia: RNN/LSTM x 3 jumlah layer x 2 hidden size.
  - File: `configs/captioning/hparam_grid.yaml`
  - File: `scripts/run_captioning_experiments.py`
- [ ] Training 12 variasi belum selesai. Notebook menunjukkan training sempat jalan lalu `KeyboardInterrupt`.
  - File: `notebooks/captioning/RNN.ipynb`
- [ ] Model RNN/LSTM belum tersimpan.
  - `models/keras/captioning` kosong selain `.gitkeep`
  - `models/keras/rnn` kosong selain `.gitkeep`
  - `models/keras/lstm` kosong selain `.gitkeep`
- [ ] History dan metadata captioning belum tersimpan.
  - `artifacts/experiments/captioning` kosong selain `.gitkeep`

### Bagian 4: Implementasi Arsitektur RNN dan LSTM

- [x] Loader/predictor Keras dan scratch tersedia untuk inference dari feature vector.
  - File: `src/tubes2_ml/captioning/inference.py`
  - File: `src/tubes2_ml/captioning/decoding.py`
- [x] Greedy decode dan beam search tersedia.
  - File: `src/tubes2_ml/captioning/decoding.py`
- [x] Pipeline raw image ke caption sudah tersedia lewat frozen CNN encoder + decoder.
  - File: `src/tubes2_ml/captioning/inference.py`
  - Gunakan `--image-path` untuk raw image atau `--image-id` untuk cached feature.
- [ ] Belum bisa dijalankan end-to-end lokal karena feature extraction dan model captioning belum ada.

### Bagian 5: Eksperimen

- [x] Script evaluasi captioning tersedia.
  - File: `scripts/evaluate_captioning_experiments.py`
  - Menghitung BLEU-4, METEOR, waktu eksekusi, dan menyimpan prediksi.
- [x] Variasi panjang maksimum caption didukung lewat argumen `--max-caption-lengths`.
  - File: `scripts/evaluate_captioning_experiments.py`
- [ ] Pipeline semua variasi Bagian 3 belum dijalankan.
- [ ] BLEU-4 dan waktu eksekusi untuk 12 variasi belum ada.
- [ ] METEOR untuk 12 variasi belum ada.
- [ ] Perbandingan Keras vs scratch belum ada.
- [ ] Pemilihan model terbaik RNN dan LSTM belum ada.
- [ ] Variasi maksimum panjang caption minimal 3 variasi belum ada hasilnya.
- [ ] Prediksi captioning belum ada.
  - `artifacts/predictions/captioning` kosong selain `.gitkeep`

### Evaluasi dan Analisis Captioning

- [ ] Analisis pengaruh jumlah layer recurrent belum ada.
- [ ] Analisis pengaruh hidden state belum ada.
- [ ] Grafik training/validation loss RNN/LSTM belum ada.
- [ ] Perbandingan BLEU-4 dan METEOR RNN vs LSTM belum ada.
- [ ] Perbandingan Keras vs scratch score dan waktu eksekusi belum ada.
- [ ] Analisis utama RNN vs LSTM belum ada.
- [ ] Qualitative analysis minimal 10 gambar belum ada.
- [ ] Analisis vanishing gradient dan memori jangka panjang belum ditulis.
- [ ] Analisis pengaruh panjang maksimum caption belum ada.

## Verifikasi Teknis

- [x] `python3 -m compileall -q src scripts` berhasil.
- [x] Import layer/model scratch berhasil.
- [x] Generator grid CNN menghasilkan 16 konfigurasi.
- [x] Generator grid captioning menghasilkan 12 konfigurasi.
- [x] Validasi dataset Flickr8k berhasil: 8.091 image id, 40.455 caption, tidak ada mismatch image-caption.
- [x] `scripts/preprocess_captions.py` berhasil membuat vocabulary dan split captioning.
- [x] `scripts/extract_caption_features.py` berhasil membuat feature InceptionV3 untuk 8.091 gambar.
- [x] `scripts/generate_cnn_report.py` berhasil membuat 16 plot loss dan report CNN.
- [x] `scripts/evaluate_cnn_best.py --scratch-max-batches 1` berhasil membuat artifact evaluasi CNN.

## Yang Paling Mendesak Dikerjakan

1. Kalau butuh angka scratch CNN full test, jalankan `PYTHONPATH=src python3 scripts/evaluate_cnn_best.py` tanpa `--scratch-max-batches`, lalu rerun `PYTHONPATH=src python3 scripts/generate_cnn_report.py`.
2. Train 12 decoder RNN/LSTM sampai selesai.
3. Jalankan evaluasi captioning untuk Keras dan scratch, termasuk BLEU-4, METEOR, waktu eksekusi, dan variasi max caption length.
4. Buat qualitative analysis 10 gambar dan kesimpulan RNN vs LSTM.
