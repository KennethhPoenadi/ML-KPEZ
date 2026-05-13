# Penjelasan Kelas dan Atribut

Dokumen ini menjelaskan kelas, atribut, dan method utama yang digunakan pada implementasi CNN, RNN, dan LSTM. Format dibuat ringkas mengikuti gaya tabel laporan: nama fungsi, tipe, parameter, dan deskripsi. Catatan: modul aktivasi pada kode ini diimplementasikan sebagai fungsi, bukan class.

## 1. Activation Functions

Lokasi file: `src/tubes2_ml/scratch/layers/activations.py`

### a. Function `linear`

| Nama Fungsi | Tipe | Parameter |
|---|---|---|
| `linear` | Function | `x: np.ndarray` |

**Deskripsi:**  
Menerapkan fungsi aktivasi linear `f(x) = x`. Fungsi ini mengembalikan input tanpa perubahan.

```python
def linear(x: np.ndarray) -> np.ndarray:
    return x
```

### b. Function `relu`

| Nama Fungsi | Tipe | Parameter |
|---|---|---|
| `relu` | Function | `x: np.ndarray` |

**Deskripsi:**  
Menerapkan fungsi aktivasi ReLU `f(x) = max(0, x)`. Nilai negatif menjadi 0, sedangkan nilai positif dipertahankan.

```python
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)
```

### c. Function `sigmoid`

| Nama Fungsi | Tipe | Parameter |
|---|---|---|
| `sigmoid` | Function | `x: np.ndarray` |

**Deskripsi:**  
Menerapkan fungsi sigmoid `f(x) = 1 / (1 + exp(-x))`. Fungsi ini banyak digunakan pada gate LSTM.

```python
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
```

### d. Function `tanh`

| Nama Fungsi | Tipe | Parameter |
|---|---|---|
| `tanh` | Function | `x: np.ndarray` |

**Deskripsi:**  
Menerapkan fungsi aktivasi hyperbolic tangent. Output berada pada rentang `[-1, 1]`.

```python
def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)
```

### e. Function `softmax`

| Nama Fungsi | Tipe | Parameter |
|---|---|---|
| `softmax` | Function | `x: np.ndarray`, `axis: int = -1` |

**Deskripsi:**  
Mengubah logits menjadi distribusi probabilitas. Implementasi menggunakan shifting dengan `max(x)` agar lebih stabil secara numerik.

```python
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)
```

### f. Function `get_activation`

| Nama Fungsi | Tipe | Parameter |
|---|---|---|
| `get_activation` | Function | `name: str \| None` |

**Deskripsi:**  
Mengambil fungsi aktivasi berdasarkan nama. Mendukung `linear`, `relu`, `sigmoid`, `tanh`, dan `softmax`.

---

## 2. CNN Scratch Layers

### a. Class `Conv2D`

Lokasi file: `src/tubes2_ml/scratch/layers/conv.py`

**Atribut**

| Nama Atribut | Tipe | Deskripsi |
|---|---|---|
| `kernel` | `np.ndarray \| None` | Bobot filter konvolusi dengan shape `[kH, kW, C_in, C_out]`. |
| `bias` | `np.ndarray \| None` | Bias untuk setiap filter output. |
| `strides` | `tuple[int, int]` | Langkah perpindahan filter pada arah tinggi dan lebar. |
| `padding` | `str` | Jenis padding, yaitu `valid` atau `same`. |
| `activation_name` | `str \| None` | Nama fungsi aktivasi yang digunakan. |
| `activation` | `Callable` | Fungsi aktivasi hasil dari `get_activation`. |

**Method**

| Nama Fungsi | Tipe | Parameter |
|---|---|---|
| `__init__` | Method | `kernel`, `bias`, `strides`, `padding`, `activation` |
| `load_keras_weights` | Method | `weights: list[np.ndarray] \| tuple[np.ndarray, ...]` |
| `forward` | Method | `x: np.ndarray` |

**Deskripsi:**  
`Conv2D` mengimplementasikan konvolusi 2D dengan parameter sharing. Input harus berbentuk `(N, H, W, C)`. Pada setiap posisi spasial, layer mengambil patch input, menghitung dot product dengan kernel, menambahkan bias, lalu menerapkan aktivasi.

```python
patch = padded[:, row_start:row_end, col_start:col_end, :]
output[:, row, col, :] = np.tensordot(
    patch,
    self.kernel,
    axes=((1, 2, 3), (0, 1, 2)),
)
```

### b. Class `LocallyConnected2D`

Lokasi file: `src/tubes2_ml/scratch/layers/locally_connected.py`

**Atribut**

| Nama Atribut | Tipe | Deskripsi |
|---|---|---|
| `kernel` | `np.ndarray \| None` | Bobot non-shared dengan shape `(out_positions, patch_size, filters)`. |
| `bias` | `np.ndarray \| None` | Bias per posisi output atau per filter. |
| `kernel_size` | `tuple[int, int] \| None` | Ukuran kernel lokal. |
| `filters` | `int \| None` | Banyak output channel. |
| `strides` | `tuple[int, int]` | Langkah sliding window. |
| `padding` | `str` | Jenis padding. |
| `activation_name` | `str \| None` | Nama aktivasi. |
| `activation` | `Callable` | Fungsi aktivasi. |

**Method**

| Nama Fungsi | Tipe | Parameter |
|---|---|---|
| `__init__` | Method | `kernel`, `bias`, `kernel_size`, `filters`, `strides`, `padding`, `activation` |
| `load_keras_weights` | Method | `weights: list[np.ndarray] \| tuple[np.ndarray, ...]` |
| `forward` | Method | `x: np.ndarray` |
| `_reshape_bias` | Static Method | `bias`, `output_height`, `output_width`, `output_channels` |

**Deskripsi:**  
`LocallyConnected2D` mengimplementasikan konvolusi tanpa parameter sharing. Setiap posisi output memiliki bobot sendiri, sehingga jumlah parameter jauh lebih besar dibanding `Conv2D`.

```python
patch = padded[:, row_start:row_end, col_start:col_end, :].reshape(batch_size, -1)
output[:, row, col, :] = patch @ self.kernel[position]
```

### c. Class `Pooling2D`

Lokasi file: `src/tubes2_ml/scratch/layers/pooling.py`

**Atribut**

| Nama Atribut | Tipe | Deskripsi |
|---|---|---|
| `pool_size` | `tuple[int, int]` | Ukuran window pooling. |
| `strides` | `tuple[int, int]` | Langkah perpindahan window pooling. |
| `padding` | `str` | Jenis padding. |
| `mode` | `str` | Jenis pooling, yaitu `max` atau `average`. |

**Method**

| Nama Fungsi | Tipe | Parameter |
|---|---|---|
| `__init__` | Method | `pool_size`, `strides`, `padding`, `mode` |
| `forward` | Method | `x: np.ndarray` |

**Deskripsi:**  
`Pooling2D` melakukan reduksi spasial pada tiap channel. Jika `mode="max"`, output berisi nilai maksimum pada window. Jika `mode="average"`, output berisi rata-rata window.

### d. Class `MaxPooling2D`

Lokasi file: `src/tubes2_ml/scratch/layers/pooling.py`

| Nama Fungsi | Tipe | Parameter |
|---|---|---|
| `__init__` | Method | `pool_size`, `strides`, `padding` |

**Deskripsi:**  
Turunan dari `Pooling2D` dengan `mode="max"`. Digunakan untuk mengambil aktivasi terbesar pada tiap window.

### e. Class `AveragePooling2D`

Lokasi file: `src/tubes2_ml/scratch/layers/pooling.py`

| Nama Fungsi | Tipe | Parameter |
|---|---|---|
| `__init__` | Method | `pool_size`, `strides`, `padding` |

**Deskripsi:**  
Turunan dari `Pooling2D` dengan `mode="average"`. Digunakan untuk mengambil nilai rata-rata pada tiap window.

### f. Class `GlobalAveragePooling2D`

Lokasi file: `src/tubes2_ml/scratch/layers/pooling.py`

| Nama Fungsi | Tipe | Parameter |
|---|---|---|
| `forward` | Method | `x: np.ndarray` |

**Deskripsi:**  
Mereduksi seluruh dimensi spasial `(H, W)` menjadi satu nilai rata-rata per channel. Output berbentuk `(N, C)`.

### g. Class `GlobalMaxPooling2D`

Lokasi file: `src/tubes2_ml/scratch/layers/pooling.py`

| Nama Fungsi | Tipe | Parameter |
|---|---|---|
| `forward` | Method | `x: np.ndarray` |

**Deskripsi:**  
Mereduksi seluruh dimensi spasial `(H, W)` menjadi nilai maksimum per channel. Output berbentuk `(N, C)`.

### h. Class `Flatten`

Lokasi file: `src/tubes2_ml/scratch/layers/flatten.py`

| Nama Fungsi | Tipe | Parameter |
|---|---|---|
| `forward` | Method | `x: np.ndarray` |

**Deskripsi:**  
Mengubah tensor dengan batch dimension menjadi matriks 2D `(N, -1)` menggunakan urutan row-major / C order, konsisten dengan Keras.

---

## 3. Dense dan Embedding Scratch Layers

### a. Class `Dense`

Lokasi file: `src/tubes2_ml/scratch/layers/dense.py`

**Atribut**

| Nama Atribut | Tipe | Deskripsi |
|---|---|---|
| `weights` | `np.ndarray \| None` | Matriks bobot Dense. |
| `bias` | `np.ndarray \| None` | Bias Dense. |
| `activation_name` | `str \| None` | Nama aktivasi. |
| `activation` | `Callable` | Fungsi aktivasi. |

**Method**

| Nama Fungsi | Tipe | Parameter |
|---|---|---|
| `__init__` | Method | `weights`, `bias`, `activation` |
| `load_keras_weights` | Method | `weights: list[np.ndarray] \| tuple[np.ndarray, ...]` |
| `forward` | Method | `x: np.ndarray` |

**Deskripsi:**  
`Dense` menghitung operasi linear `output = x @ W + b`, lalu menerapkan aktivasi. Layer ini digunakan pada CNN classifier, feature projection captioning, dan output vocabulary.

```python
output = x @ self.weights
if self.bias is not None:
    output += self.bias
return self.activation(output)
```

### b. Class `Embedding`

Lokasi file: `src/tubes2_ml/scratch/layers/embedding.py`

**Atribut**

| Nama Atribut | Tipe | Deskripsi |
|---|---|---|
| `weights` | `np.ndarray \| None` | Matriks embedding dengan shape `(vocab_size, embed_dim)`. |

**Method**

| Nama Fungsi | Tipe | Parameter |
|---|---|---|
| `__init__` | Method | `weights` |
| `load_keras_weights` | Method | `weights: list[np.ndarray] \| tuple[np.ndarray, ...]` |
| `forward` | Method | `token_ids: np.ndarray` |

**Deskripsi:**  
`Embedding` mengubah token id menjadi vektor embedding. Input adalah token integer, output adalah tensor embedding sesuai shape input token ditambah dimensi embedding.

---

## 4. RNN dan LSTM Scratch Layers

### a. Class `SimpleRNN`

Lokasi file: `src/tubes2_ml/scratch/layers/recurrent.py`

**Atribut**

| Nama Atribut | Tipe | Deskripsi |
|---|---|---|
| `kernel` | `np.ndarray \| None` | Bobot input-to-hidden dari Keras. |
| `recurrent_kernel` | `np.ndarray \| None` | Bobot hidden-to-hidden dari Keras. |
| `bias` | `np.ndarray \| None` | Bias SimpleRNN. |
| `units` | `int` | Property yang menunjukkan jumlah hidden unit. |

**Method**

| Nama Fungsi | Tipe | Parameter |
|---|---|---|
| `__init__` | Method | `kernel`, `recurrent_kernel`, `bias` |
| `load_keras_weights` | Method | `weights: list[np.ndarray] \| tuple[np.ndarray, ...]` |
| `units` | Property | - |
| `forward` | Method | `x`, `initial_state`, `return_state` |

**Deskripsi:**  
`SimpleRNN` melakukan forward propagation recurrent dengan rumus:

```python
h_t = tanh(x_t @ kernel + h_{t-1} @ recurrent_kernel + bias)
```

Input berbentuk `(batch, timesteps, features)`. Jika `initial_state` tidak diberikan, hidden state awal diisi nol.

### b. Class `LSTM`

Lokasi file: `src/tubes2_ml/scratch/layers/recurrent.py`

**Atribut**

| Nama Atribut | Tipe | Deskripsi |
|---|---|---|
| `kernel` | `np.ndarray \| None` | Bobot input untuk semua gate LSTM. |
| `recurrent_kernel` | `np.ndarray \| None` | Bobot recurrent untuk semua gate LSTM. |
| `bias` | `np.ndarray \| None` | Bias LSTM. |
| `units` | `int` | Property jumlah hidden unit. |

**Method**

| Nama Fungsi | Tipe | Parameter |
|---|---|---|
| `__init__` | Method | `kernel`, `recurrent_kernel`, `bias` |
| `load_keras_weights` | Method | `weights: list[np.ndarray] \| tuple[np.ndarray, ...]` |
| `units` | Property | - |
| `forward` | Method | `x`, `initial_state`, `return_state` |

**Deskripsi:**  
`LSTM` melakukan forward propagation dengan empat gate sesuai format bobot Keras: input gate, forget gate, candidate cell, dan output gate.

```python
z = x_t @ kernel + h_t @ recurrent_kernel + bias
z_i, z_f, z_c, z_o = np.split(z, 4, axis=-1)
i_t = sigmoid(z_i)
f_t = sigmoid(z_f)
c_hat_t = tanh(z_c)
o_t = sigmoid(z_o)
c_t = f_t * c_t + i_t * c_hat_t
h_t = o_t * tanh(c_t)
```

Jika `initial_state` tidak diberikan, `h0` dan `c0` diisi nol.

---

## 5. Scratch Model Wrappers

### a. Class `ScratchCNNClassifier`

Lokasi file: `src/tubes2_ml/scratch/models/cnn_classifier.py`

**Atribut**

| Nama Atribut | Tipe | Deskripsi |
|---|---|---|
| `layers` | `list` | Daftar layer scratch yang dijalankan berurutan. |

**Method**

| Nama Fungsi | Tipe | Parameter |
|---|---|---|
| `__init__` | Method | `layers: list` |
| `forward` | Method | `x: np.ndarray` |
| `predict` | Method | `x: np.ndarray` |
| `count_parameters` | Method | - |

**Deskripsi:**  
Wrapper untuk menjalankan model CNN dari scratch. Method `forward` melewatkan input ke semua layer, `predict` mengambil `argmax` probabilitas kelas, dan `count_parameters` menghitung total parameter.

### b. Class `ScratchRNNCaptioner`

Lokasi file: `src/tubes2_ml/scratch/models/rnn_captioner.py`

**Atribut**

| Nama Atribut | Tipe | Deskripsi |
|---|---|---|
| `feature_projection` | `Dense` | Dense layer untuk memproyeksikan CNN feature ke embedding dimension. |
| `embedding` | `Embedding` | Token embedding layer. |
| `recurrent_layers` | `list[SimpleRNN]` | Daftar layer SimpleRNN scratch. |
| `output_dense` | `Dense` | Dense output menuju vocab logits. |

**Method**

| Nama Fungsi | Tipe | Parameter |
|---|---|---|
| `__init__` | Method | `feature_projection`, `embedding`, `recurrent_layers`, `output_dense` |
| `from_keras_model` | Class Method | `keras_model` |
| `forward` | Method | `image_features`, `caption_input_ids` |

**Deskripsi:**  
`ScratchRNNCaptioner` membangun decoder captioning RNN dari bobot model Keras. Feature CNN diproyeksikan sebagai timestep awal, lalu digabung dengan embedding token. Output berupa probabilitas vocabulary.

### c. Class `ScratchLSTMCaptioner`

Lokasi file: `src/tubes2_ml/scratch/models/lstm_captioner.py`

**Atribut**

| Nama Atribut | Tipe | Deskripsi |
|---|---|---|
| `feature_projection` | `Dense` | Dense layer untuk memproyeksikan CNN feature ke embedding dimension. |
| `embedding` | `Embedding` | Token embedding layer. |
| `recurrent_layers` | `list[LSTM]` | Daftar layer LSTM scratch. |
| `output_dense` | `Dense` | Dense output menuju vocab logits. |

**Method**

| Nama Fungsi | Tipe | Parameter |
|---|---|---|
| `__init__` | Method | `feature_projection`, `embedding`, `recurrent_layers`, `output_dense` |
| `from_keras_model` | Class Method | `keras_model` |
| `forward` | Method | `image_features`, `caption_input_ids` |

**Deskripsi:**  
`ScratchLSTMCaptioner` membangun decoder captioning LSTM dari bobot model Keras. Alurnya sama dengan `ScratchRNNCaptioner`, tetapi recurrent cell yang digunakan adalah LSTM.

---

## 6. CNN Configuration dan Training

### a. Class `SharedConvCNNConfig`

Lokasi file: `src/tubes2_ml/cnn/models.py`

**Atribut**

| Nama Atribut | Tipe | Deskripsi |
|---|---|---|
| `input_shape` | `tuple[int, int, int]` | Shape input gambar, default `(96, 96, 3)`. |
| `num_classes` | `int` | Jumlah kelas Intel Image Classification, default 6. |
| `conv_filters` | `tuple[int, ...]` | Banyak filter tiap layer Conv2D. |
| `kernel_sizes` | `tuple[int, ...]` | Ukuran kernel tiap Conv2D. |
| `pooling_type` | `str` | Jenis pooling, `max` atau `average`. |
| `dense_units` | `tuple[int, ...]` | Unit pada Dense hidden head. |
| `dropout_rate` | `float` | Dropout setelah Dense hidden. |
| `learning_rate` | `float` | Learning rate Adam. |
| `activation` | `str` | Aktivasi Conv2D dan Dense hidden. |
| `name` | `str` | Nama model/eksperimen. |
| `compile_model` | `bool` | Menentukan apakah model langsung di-compile. |
| `metrics` | `tuple[str, ...]` | Metrik Keras saat training. |

**Deskripsi:**  
Dataclass konfigurasi untuk membangun CNN shared-parameter menggunakan Keras.

### b. Class `CNNTrainingConfig`

Lokasi file: `src/tubes2_ml/cnn/train.py`

**Atribut**

| Nama Atribut | Tipe | Deskripsi |
|---|---|---|
| `train_dir` | `str \| Path` | Folder data training Intel. |
| `validation_dir` | `str \| Path \| None` | Folder validation eksplisit; jika `None`, memakai validation split. |
| `output_dir` | `str \| Path` | Folder output model dan bobot. |
| `history_dir` | `str \| Path` | Folder output history dan metadata. |
| `image_size` | `tuple[int, int]` | Ukuran resize gambar. |
| `batch_size` | `int` | Ukuran batch training. |
| `epochs` | `int` | Jumlah epoch. |
| `validation_split` | `float` | Rasio validation dari train directory. |
| `seed` | `int` | Seed random. |
| `save_format` | `str` | Format penyimpanan model. |
| `early_stopping_patience` | `int \| None` | Patience early stopping. |

**Deskripsi:**  
Dataclass konfigurasi training CNN Keras.

---

## 7. Captioning Configuration, Data, dan Training

### a. Class `CaptionDecoderConfig`

Lokasi file: `src/tubes2_ml/captioning/models.py`

**Atribut**

| Nama Atribut | Tipe | Deskripsi |
|---|---|---|
| `vocab_size` | `int` | Jumlah token vocabulary. |
| `feature_dim` | `int` | Dimensi feature vector CNN encoder. |
| `max_caption_length` | `int` | Panjang maksimum sequence caption. |
| `embed_dim` | `int` | Dimensi embedding token dan proyeksi image feature. |
| `hidden_units` | `int` | Ukuran hidden state RNN/LSTM. |
| `num_recurrent_layers` | `int` | Jumlah layer recurrent. |
| `dropout_rate` | `float` | Dropout recurrent layer. |
| `learning_rate` | `float` | Learning rate Adam. |
| `decoder_type` | `Literal["rnn", "lstm"]` | Jenis decoder. |
| `injection_mode` | `Literal["pre", "init"]` | Mode injeksi image feature. Spek utama memakai `pre`. |
| `name` | `str` | Nama model. |

**Deskripsi:**  
Dataclass konfigurasi decoder captioning Keras untuk RNN/LSTM.

### b. Class `CaptionTrainingConfig`

Lokasi file: `src/tubes2_ml/captioning/train.py`

**Atribut**

| Nama Atribut | Tipe | Deskripsi |
|---|---|---|
| `processed_dir` | `Path` | Folder hasil preprocessing caption. |
| `features_dir` | `Path` | Folder feature vector CNN encoder. |
| `output_dir` | `Path` | Folder output model Keras. |
| `history_dir` | `Path` | Folder output history dan metadata. |
| `batch_size` | `int` | Ukuran batch training. |
| `epochs` | `int` | Jumlah epoch. |
| `seed` | `int` | Seed random. |

**Deskripsi:**  
Konfigurasi proses training decoder captioning.

### c. Class `CaptionTrainingResult`

Lokasi file: `src/tubes2_ml/captioning/train.py`

**Atribut**

| Nama Atribut | Tipe | Deskripsi |
|---|---|---|
| `model_path` | `Path` | Path model `.keras` hasil training. |
| `history_path` | `Path` | Path file history JSON. |
| `metadata_path` | `Path` | Path file metadata JSON. |
| `decoder_type` | `str` | Jenis decoder, `rnn` atau `lstm`. |
| `injection_mode` | `str` | Mode injeksi feature. |
| `num_recurrent_layers` | `int` | Jumlah recurrent layer. |
| `hidden_units` | `int` | Ukuran hidden state. |
| `best_validation_loss` | `float` | Validation loss terbaik. |

**Deskripsi:**  
Dataclass hasil training satu variasi decoder captioning.

### d. Class `CaptionRecord`

Lokasi file: `src/tubes2_ml/captioning/preprocessing.py`

**Atribut**

| Nama Atribut | Tipe | Deskripsi |
|---|---|---|
| `image_id` | `str` | ID gambar tanpa ekstensi file. |
| `caption` | `str` | Caption mentah dari dataset. |

**Deskripsi:**  
Representasi satu baris data caption.

### e. Class `CaptionPreprocessingConfig`

Lokasi file: `src/tubes2_ml/captioning/preprocessing.py`

**Atribut**

| Nama Atribut | Tipe | Deskripsi |
|---|---|---|
| `captions_path` | `Path` | Path file `captions.txt`. |
| `output_dir` | `Path` | Folder output preprocessing. |
| `min_freq` | `int` | Frekuensi minimum token untuk masuk vocabulary. |
| `max_vocab_size` | `int \| None` | Batas maksimum ukuran vocabulary. |
| `max_caption_length` | `int \| None` | Panjang caption maksimum; jika `None`, diinfer dari data. |
| `train_size` | `int` | Jumlah image train. |
| `validation_size` | `int` | Jumlah image validation. |
| `test_size` | `int` | Jumlah image test. |
| `limit_images` | `int \| None` | Limit jumlah image untuk debugging. |

**Deskripsi:**  
Konfigurasi preprocessing caption Flickr8k.

### f. Class `CaptionPreprocessingResult`

Lokasi file: `src/tubes2_ml/captioning/preprocessing.py`

**Atribut**

| Nama Atribut | Tipe | Deskripsi |
|---|---|---|
| `output_dir` | `Path` | Folder output preprocessing. |
| `vocabulary_path` | `Path` | Path vocabulary JSON. |
| `metadata_path` | `Path` | Path metadata JSON. |
| `train_path` | `Path` | Path split train `.npz`. |
| `validation_path` | `Path` | Path split validation `.npz`. |
| `test_path` | `Path` | Path split test `.npz`. |
| `vocab_size` | `int` | Ukuran vocabulary. |
| `max_caption_length` | `int` | Panjang caption maksimum. |
| `num_train_images` | `int` | Jumlah image train. |
| `num_validation_images` | `int` | Jumlah image validation. |
| `num_test_images` | `int` | Jumlah image test. |
| `num_train_captions` | `int` | Jumlah caption train. |
| `num_validation_captions` | `int` | Jumlah caption validation. |
| `num_test_captions` | `int` | Jumlah caption test. |

**Deskripsi:**  
Dataclass ringkasan hasil preprocessing caption.

### g. Class `FeatureExtractionConfig`

Lokasi file: `src/tubes2_ml/captioning/feature_extraction.py`

**Atribut**

| Nama Atribut | Tipe | Deskripsi |
|---|---|---|
| `images_dir` | `Path` | Folder gambar Flickr8k. |
| `output_dir` | `Path` | Folder output feature `.npy`. |
| `encoder_name` | `Literal["inception_v3", "vgg16"]` | Encoder pretrained yang digunakan. |
| `batch_size` | `int` | Batch size saat ekstraksi feature. |
| `limit` | `int \| None` | Limit image untuk debugging. |
| `overwrite` | `bool` | Menentukan apakah feature lama ditimpa. |

**Deskripsi:**  
Konfigurasi ekstraksi feature gambar menggunakan CNN encoder frozen.

### h. Class `FeatureExtractionResult`

Lokasi file: `src/tubes2_ml/captioning/feature_extraction.py`

**Atribut**

| Nama Atribut | Tipe | Deskripsi |
|---|---|---|
| `features_path` | `Path` | Path file `features.npy`. |
| `image_ids_path` | `Path` | Path file `image_ids.json`. |
| `metadata_path` | `Path` | Path metadata ekstraksi. |
| `num_images` | `int` | Jumlah gambar yang diekstraksi. |
| `feature_shape` | `tuple[int, ...]` | Shape feature per gambar. |
| `encoder_name` | `str` | Nama encoder yang digunakan. |

**Deskripsi:**  
Dataclass ringkasan hasil ekstraksi feature.

### i. Class `CaptionVocabulary`

Lokasi file: `src/tubes2_ml/captioning/decoding.py`

**Atribut**

| Nama Atribut | Tipe | Deskripsi |
|---|---|---|
| `word_to_id` | `dict[str, int]` | Mapping token ke id. |
| `id_to_word` | `dict[str, str]` | Mapping id ke token. |
| `pad_token` | `str` | Token padding. |
| `start_token` | `str` | Token awal caption. |
| `end_token` | `str` | Token akhir caption. |
| `unk_token` | `str` | Token unknown. |
| `pad_id` | `int` | Property id token padding. |
| `start_id` | `int` | Property id token start. |
| `end_id` | `int` | Property id token end. |

**Method**

| Nama Fungsi | Tipe | Parameter |
|---|---|---|
| `pad_id` | Property | - |
| `start_id` | Property | - |
| `end_id` | Property | - |
| `ids_to_words` | Method | `token_ids`, `skip_special` |
| `ids_to_caption` | Method | `token_ids` |

**Deskripsi:**  
Menyimpan vocabulary captioning dan menyediakan method untuk mengubah token id hasil decoder menjadi kata atau caption string.

---

## 8. Utility Classes

### a. Class `Timer`

Lokasi file: `src/tubes2_ml/evaluation/timing.py`

**Atribut**

| Nama Atribut | Tipe | Deskripsi |
|---|---|---|
| `durations` | `dict[str, float]` | Menyimpan durasi eksekusi untuk nama blok tertentu. |

**Method**

| Nama Fungsi | Tipe | Parameter |
|---|---|---|
| `track` | Context Manager Method | `name: str` |

**Deskripsi:**  
Mencatat durasi eksekusi blok kode menggunakan context manager.

### b. Class `MetricsPayload`

Lokasi file: `src/tubes2_ml/experiments/metrics.py`

**Atribut**

| Nama Atribut | Tipe | Deskripsi |
|---|---|---|
| `experiment` | `str` | Nama eksperimen. |
| `metrics` | `dict[str, float]` | Nilai metrik eksperimen. |
| `metadata` | `dict[str, Any] \| None` | Metadata tambahan. |

**Deskripsi:**  
Dataclass untuk menyimpan payload metrik eksperimen ke JSON.
