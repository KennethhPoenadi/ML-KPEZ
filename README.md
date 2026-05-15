# Kelompok - KPEZ

Repository ini berisi implementasi tugas besar Machine Learning untuk dua task utama:

1. **Image Classification** menggunakan Convolutional Neural Network pada dataset Intel Image Classification.
2. **Image Captioning** menggunakan SimpleRNN dan LSTM pada dataset Flickr8k.

Project mencakup training model Keras, implementasi forward propagation from scratch menggunakan NumPy, evaluasi metrik, visualisasi hasil eksperimen, serta analisis perbandingan model.

## Struktur Singkat Repository

| Path | Deskripsi |
|---|---|
| `src/tubes2_ml/cnn/` | Pipeline model, training, dan evaluasi CNN Keras |
| `src/tubes2_ml/captioning/` | Pipeline feature extraction, preprocessing, training, decoding, dan inference image captioning |
| `src/tubes2_ml/scratch/` | Implementasi layer dan model from scratch menggunakan NumPy |
| `scripts/` | Script untuk menjalankan eksperimen, evaluasi, preprocessing, dan report |
| `configs/` | Konfigurasi eksperimen CNN dan captioning |
| `notebooks/` | Notebook CNN dan RNN/LSTM |
| `artifacts/` | Hasil eksperimen, plot, prediksi, dan report |
| `models/` | Bobot/model Keras hasil training |
| `data/` | Dataset raw, processed, dan feature hasil ekstraksi |

## Setup

Clone repository dan masuk ke folder project:

```bash
git clone https://github.com/KennethhPoenadi/ML-KPEZ.git
cd ML-KPEZ
```

Buat virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Untuk Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependency:

```bash
pip install -r requirements.txt
```

Untuk Mac Apple Silicon, dependency GPU dapat mengikuti file:

```bash
pip install -r requirements-macos-gpu.txt
```

Jika `tensorflow-macos` tidak tersedia pada versi Python yang digunakan, gunakan TensorFlow versi reguler:

```bash
pip install tensorflow
```

## Persiapan Dataset

Dataset Intel Image Classification diletakkan pada struktur data CNN yang digunakan oleh konfigurasi project.

Dataset Flickr8k diletakkan pada:

```text
data/raw/flickr8k/images/
data/raw/flickr8k/captions/captions.txt
```

Feature hasil ekstraksi captioning disimpan pada:

```text
data/features/captioning/
```

Data caption hasil preprocessing disimpan pada:

```text
data/processed/captioning/
```

## Cara Menjalankan Program

### CNN

Menjalankan training 16 variasi CNN:

```bash
PYTHONPATH=src python3 scripts/run_cnn_experiments.py --config configs/cnn/hparam_grid.yaml
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
py scripts/run_cnn_experiments.py --config configs/cnn/hparam_grid.yaml
```

Generate report, plot history, dan evaluasi model CNN:

```bash
PYTHONPATH=src python3 scripts/generate_cnn_report.py
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
py scripts/generate_cnn_report.py
```

Notebook CNN juga dapat dijalankan dari:

```text
notebooks/cnn/CNN.ipynb
```

### Captioning RNN/LSTM

Preprocessing caption Flickr8k:

```bash
PYTHONPATH=src python3 scripts/preprocess_captions.py
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
py scripts/preprocess_captions.py
```

Feature extraction CNN encoder untuk Flickr8k:

```bash
PYTHONPATH=src python3 scripts/extract_captioning_features.py
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
py scripts/extract_captioning_features.py
```

Training 12 variasi RNN/LSTM:

```bash
PYTHONPATH=src python3 scripts/run_captioning_experiments.py --config configs/captioning/hparam_grid.yaml
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
py scripts/run_captioning_experiments.py --config configs/captioning/hparam_grid.yaml
```

Evaluasi BLEU-4, METEOR, Keras vs Scratch, dan variasi panjang caption:

```bash
PYTHONPATH=src python3 scripts/evaluate_captioning_experiments.py \
  --backends keras,scratch \
  --searches greedy \
  --max-caption-lengths 10,20,38 \
  --batch-size 1024
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
py scripts/evaluate_captioning_experiments.py `
  --backends keras,scratch `
  --searches greedy `
  --max-caption-lengths 10,20,38 `
  --batch-size 1024
```

Evaluasi beam search:

```bash
PYTHONPATH=src python3 scripts/evaluate_captioning_experiments.py \
  --backends keras \
  --searches greedy,beam \
  --beam-widths 3 \
  --max-caption-lengths 20 \
  --limit-images 50 \
  --batch-size 1024
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
py scripts/evaluate_captioning_experiments.py `
  --backends keras `
  --searches greedy,beam `
  --beam-widths 3 `
  --max-caption-lengths 20 `
  --limit-images 50 `
  --batch-size 1024
```

Notebook RNN/LSTM juga dapat dijalankan dari:

```text
notebooks/captioning/RNN.ipynb
```

### Plot dan Ringkasan Evaluasi

Mengumpulkan hasil evaluasi dan membuat plot training history:

```bash
PYTHONPATH=src python3 scripts/evaluate_all.py --format both
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
py scripts/evaluate_all.py --format both
```

Hasil utama tersimpan pada:

```text
artifacts/experiments/
artifacts/plots/
artifacts/predictions/
artifacts/reports/
```

## Pembagian Tugas

| Implementasi | Penanggung Jawab |
|---|---:|
| Integrasi pipeline training, inference, dan evaluasi dari bagian CNN serta RNN/LSTM; implementasi helper umum seperti load/save weights, metrik, waktu eksekusi, dan plotting history training; pelaksanaan dan pengumpulan hasil eksperimen utama seperti evaluasi macro F1-score CNN, BLEU-4, METEOR, waktu eksekusi Keras vs Scratch, serta grafik loss; pelaksanaan dan perekapan eksperimen bonus seperti feature maps, Grad-CAM, pre-inject vs init-inject, beam search, batch inference, dan backward propagation. | 13523029 |
| Feature extraction CNN encoder untuk Flickr8k; preprocessing caption; decoder Keras untuk SimpleRNN dan LSTM dengan arsitektur pre-inject; variasi eksperimen captioning; forward propagation from scratch; inference caption generation untuk RNN dan LSTM; implementasi bonus RNN/LSTM; serta evaluasi BLEU-4 dan METEOR untuk semua variasi RNN/LSTM. | 13523033 |
| Implementasi utility image loading untuk dataset Intel Image Classification; pipeline training CNN Keras dengan Conv2D shared parameter; variasi eksperimen CNN; forward propagation CNN from scratch; implementasi bonus CNN seperti visualisasi intermediate feature maps, Grad-CAM, batch inference, dan backward propagation; evaluasi CNN menggunakan macro F1-score; serta analisis perbandingan Keras vs Scratch, shared vs non-shared parameter, dan pengaruh hyperparameter CNN. | 13523040 |

## Link Repository

https://github.com/KennethhPoenadi/ML-KPEZ
