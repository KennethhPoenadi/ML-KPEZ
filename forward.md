# Penjelasan Implementasi Forward Propagation pada Kode

Forward propagation adalah proses menghitung output model dari input dengan melewatkan data secara berurutan melalui layer-layer jaringan. Pada repository ini, forward propagation diimplementasikan untuk:

1. CNN untuk klasifikasi gambar.
2. RNN untuk image captioning.
3. LSTM untuk image captioning.

Penjelasan di bawah menghubungkan konsep matematis, arti simbol, dan kode implementasi yang digunakan.

---

## 1. Konsep Dasar Forward Pass

Forward pass menghitung aktivasi layer demi layer:

```text
input -> layer 1 -> layer 2 -> ... -> output
```

Untuk layer umum, notasinya:

```math
z^{(l)} = f^{(l)}(a^{(l-1)}, W^{(l)}, b^{(l)})
```

```math
a^{(l)} = g(z^{(l)})
```

Keterangan:

| Simbol | Arti |
|---|---|
| `a^(l-1)` | Aktivasi/output dari layer sebelumnya. |
| `W^(l)` | Bobot layer ke-`l`. |
| `b^(l)` | Bias layer ke-`l`. |
| `z^(l)` | Pre-activation, hasil operasi linear/konvolusi sebelum aktivasi. |
| `g` | Fungsi aktivasi seperti ReLU, tanh, sigmoid, atau softmax. |
| `a^(l)` | Output layer ke-`l`. |

Pada kode, pola ini terlihat pada layer `Dense`:

```python
output = x @ self.weights
if self.bias is not None:
    output += self.bias

return self.activation(output)
```

Kode tersebut sesuai dengan rumus:

```math
z = xW + b
```

```math
a = g(z)
```

---

## 2. Forward Propagation CNN

CNN digunakan untuk task image classification pada dataset Intel Image Classification. Forward CNN terdiri dari operasi konvolusi, aktivasi, pooling, global pooling/flatten, dense, dan softmax.

### 2.1 Alur Forward CNN

Pada model Keras CNN, alurnya adalah:

```text
Input image
-> Conv2D
-> ReLU
-> Max/Average Pooling
-> Conv2D berikutnya, jika ada
-> GlobalAveragePooling2D
-> Dense
-> Dropout
-> Dense output
-> Softmax
-> Probabilitas kelas
```

Implementasi model Keras:

```python
def _add_conv_block(x, filters: int, kernel_size: int, pooling_type: str, activation: str, index: int):
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(kernel_size, kernel_size),
        padding="same",
        activation=activation,
        name=f"conv_{index}",
    )(x)

    pooling_layer = (
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name=f"pool_{index}")
        if pooling_type == "max"
        else tf.keras.layers.AveragePooling2D(pool_size=(2, 2), name=f"pool_{index}")
    )

    return pooling_layer(x)
```

Head klasifikasi:

```python
def _add_dense_head(x, config: SharedConvCNNConfig):
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)

    for dense_index, units in enumerate(config.dense_units, start=1):
        x = tf.keras.layers.Dense(units, activation=config.activation, name=f"dense_{dense_index}")(x)
        if config.dropout_rate > 0:
            x = tf.keras.layers.Dropout(config.dropout_rate, name=f"dropout_{dense_index}")(x)

    return tf.keras.layers.Dense(
        config.num_classes,
        activation="softmax",
        name="class_probabilities",
    )(x)
```

### 2.2 Forward Conv2D Shared Parameter

Conv2D menghitung dot product antara patch input dan kernel. Karena memakai shared parameter, kernel yang sama digunakan pada semua posisi gambar.

Rumus untuk satu output pixel:

```math
Y[n, i, j, k] =
\sum_{u=0}^{K_h-1}
\sum_{v=0}^{K_w-1}
\sum_{c=0}^{C_{in}-1}
X[n, i+u, j+v, c] \cdot W[u, v, c, k] + b[k]
```

uKeterangan:

| Simbol | Arti |
|---|---|
| `X` | Input image/batch feature map. |
| `Y` | Output feature map. |
| `n` | Index batch. |
| `i, j` | Posisi spasial output. |
| `u, v` | Posisi di dalam kernel. |
| `c` | Channel input. |
| `k` | Channel/filter output. |
| `W` | Kernel Conv2D dengan shape `(kH, kW, C_in, C_out)`. |
| `b` | Bias per filter output. |

Kode yang sesuai:

```python
for row in range(output_height):
    row_start = row * stride_h
    row_end = row_start + kernel_height
    for col in range(output_width):
        col_start = col * stride_w
        col_end = col_start + kernel_width
        patch = padded[:, row_start:row_end, col_start:col_end, :]
        output[:, row, col, :] = np.tensordot(
            patch,
            self.kernel,
            axes=((1, 2, 3), (0, 1, 2)),
        )

if self.bias is not None:
    output += self.bias.reshape(1, 1, 1, output_channels)

return self.activation(output)
```

jHubungan kode dengan rumus:

| Rumus | Kode |
|---|---|
| `X[n, i+u, j+v, c]` | `patch` |
| `W[u, v, c, k]` | `self.kernel` |
| `sum(...)` | `np.tensordot(...)` |
| `+ b[k]` | `output += self.bias.reshape(...)` |
| `g(Y)` | `return self.activation(output)` |

Contoh shape:

```text
Input X       : (64, 96, 96, 3)
Kernel W      : (3, 3, 3, 32)
Bias b        : (32,)
Output Y same : (64, 96, 96, 32)
```

Jika memakai pooling 2x2 setelah Conv2D:

```text
Output pooling: (64, 48, 48, 32)
```

### 2.3 Forward LocallyConnected2D Non-Shared Parameter

LocallyConnected2D mirip Conv2D, tetapi tidak memakai parameter sharing. Setiap posisi output punya kernel sendiri.

Rumus:

```math
Y[n, p, k] =
\sum_{q=0}^{K_hK_wC_{in}-1}
P[n, p, q] \cdot W[p, q, k] + b[p, k]
```

Keterangan:

| Simbol | Arti |
|---|---|
| `p` | Index posisi output, hasil flatten dari `(i, j)`. |
| `q` | Index elemen patch yang sudah diratakan. |
| `P` | Patch input yang sudah diratakan. |
| `W[p]` | Kernel khusus untuk posisi output ke-`p`. |
| `b[p]` | Bias untuk posisi output ke-`p`. |

Kode yang sesuai:

```python
position = 0
for row in range(output_height):
    row_start = row * stride_h
    row_end = row_start + kernel_height
    for col in range(output_width):
        col_start = col * stride_w
        col_end = col_start + kernel_width
        patch = padded[:, row_start:row_end, col_start:col_end, :].reshape(batch_size, -1)
        output[:, row, col, :] = patch @ self.kernel[position]
        position += 1

if self.bias is not None:
    output += self._reshape_bias(self.bias, output_height, output_width, output_channels)

return self.activation(output)
```

Hubungan kode dengan rumus:

| Rumus | Kode |
|---|---|
| `P[n, p, q]` | `patch` |
| `W[p, q, k]` | `self.kernel[position]` |
| `sum_q P * W` | `patch @ self.kernel[position]` |
| `b[p, k]` | `self._reshape_bias(...)` |

Perbandingan:

| Layer | Parameter |
|---|---|
| Conv2D | Satu kernel dipakai ulang di seluruh posisi. |
| LocallyConnected2D | Tiap posisi output punya kernel sendiri. |

Karena itu LocallyConnected2D jauh lebih mahal secara parameter.

### 2.4 Forward Pooling

Pooling mereduksi ukuran spasial feature map. Ada dua mode:

Max pooling:

```math
Y[n, i, j, c] = \max_{(u,v) \in R} X[n, i+u, j+v, c]
```

Average pooling:

```math
Y[n, i, j, c] = \frac{1}{|R|}
\sum_{(u,v) \in R} X[n, i+u, j+v, c]
```

Keterangan:

| Simbol | Arti |
|---|---|
| `R` | Area/window pooling. |
| `c` | Channel feature map. |
| `|R|` | Jumlah elemen dalam window pooling. |

Kode:

```python
patch = padded[:, row_start:row_end, col_start:col_end, :]
if self.mode == "max":
    output[:, row, col, :] = np.max(patch, axis=(1, 2))
else:
    output[:, row, col, :] = np.mean(patch, axis=(1, 2))
```

Pooling dilakukan independen untuk setiap channel.

### 2.5 Forward Global Average Pooling

GlobalAveragePooling2D mengubah feature map `(N, H, W, C)` menjadi `(N, C)`.

Rumus:

```math
Y[n, c] =
\frac{1}{H \cdot W}
\sum_{i=0}^{H-1}
\sum_{j=0}^{W-1}
X[n, i, j, c]
```

Kode:

```python
return np.mean(x, axis=(1, 2))
```
d
Hubungan:

| Rumus | Kode |
|---|---|
| `sum over H and W` | `axis=(1, 2)` |
| dibagi `H * W` | `np.mean` |

### 2.6 Forward Dense dan Softmax pada CNN

Dense layer:

```math
z = xW + b
```

Aktivasi:

```math
a = g(z)
```

Kode:

```python
output = x @ self.weights
if self.bias is not None:
    output += self.bias

return self.activation(output)
```

Untuk output klasifikasi, aktivasi softmax:

```math
\hat{y}_k =
\frac{e^{z_k}}{\sum_{r=1}^{C} e^{z_r}}
```

Kode softmax:

```python
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)
```

Output akhir CNN:

```text
probabilities.shape = (batch_size, 6)
```

Setiap nilai menunjukkan probabilitas kelas `buildings`, `forest`, `glacier`, `mountain`, `sea`, atau `street`.

---

## 3. Forward Propagation RNN

RNN digunakan sebagai decoder image captioning. Arsitektur utama memakai metode pre-inject, yaitu feature CNN dimasukkan sebagai timestep pertama sebelum token `<start>`.

### 3.1 Konsep Forward RNN Captioning

Alur:

```text
image
-> CNN encoder frozen
-> feature vector
-> Dense projection
-> x_-1
-> Embedding caption input
-> SimpleRNN
-> Dense output
-> Softmax vocabulary
```

Input training:

```text
[CNN_feature, <start>, S0, S1, ..., S_{N-1}]
```

Target:

```text
[S0, S1, ..., S_N]
```

### 3.2 Dense Projection untuk Feature CNN

Feature CNN dari InceptionV3 memiliki shape:

```text
image_feature.shape = (batch, 2048)
```

Feature ini diproyeksikan ke `embed_dim`:

```math
x_{-1} = W_f v + b_f
```

Keterangan:

| Simbol | Arti |
|---|---|
| `v` | Feature vector CNN. |
| `W_f` | Bobot Dense projection. |
| `b_f` | Bias Dense projection. |
| `x_-1` | Feature timestep sebelum token `<start>`. |

Kode Keras:

```python
feature_embedding = tf.keras.layers.Dense(config.embed_dim, name="feature_projection")(feature_input)
feature_embedding = tf.keras.layers.Reshape((1, config.embed_dim), name="feature_timestep")(feature_embedding)
```

Kode scratch:

```python
feature_timestep = self.feature_projection.forward(features)[:, np.newaxis, :]
```

Jika:

```text
features.shape = (batch, 2048)
embed_dim = 256
```

maka:

```text
feature_timestep.shape = (batch, 1, 256)
```

### 3.3 Embedding Token Caption

Token id caption diubah menjadi vektor embedding.

Rumus:

```math
x_t = E[w_t]
```

Keterangan:

| Simbol | Arti |
|---|---|
| `w_t` | Token id pada timestep `t`. |
| `E` | Matriks embedding dengan shape `(vocab_size, embed_dim)`. |
| `x_t` | Vektor embedding token. |

Kode:

```python
ids = np.asarray(token_ids, dtype=np.int64)
return self.weights[ids]
```

Jika:

```text
token_ids.shape = (batch, max_caption_length)
embedding.shape = (vocab_size, 256)
```

maka:

```text
token_embeddings.shape = (batch, max_caption_length, 256)
```

### 3.4 Penggabungan Feature Timestep dan Token Embedding

Pada pre-inject, feature timestep ditempel di depan sequence token:

```math
X = [x_{-1}, x_0, x_1, ..., x_T]
```

Kode Keras:

```python
sequence = tf.keras.layers.Concatenate(axis=1, name="preinject_sequence")(
    [feature_embedding, token_embedding]
)
```

Kode scratch:

```python
sequence = np.concatenate([feature_timestep, token_embeddings], axis=1)
```

Contoh shape:

```text
feature_timestep.shape = (batch, 1, 256)
token_embeddings.shape = (batch, 38, 256)
sequence.shape = (batch, 39, 256)
```

### 3.5 Forward SimpleRNN Cell

SimpleRNN menyimpan satu hidden state `h_t`. Rumus forward:

```math
h_t = \tanh(x_t W_x + h_{t-1} W_h + b)
```

Keterangan:

| Simbol | Arti |
|---|---|
| `x_t` | Input pada timestep ke-`t`. |
| `W_x` | Bobot input-to-hidden (`kernel`). |
| `h_{t-1}` | Hidden state timestep sebelumnya. |
| `W_h` | Bobot hidden-to-hidden (`recurrent_kernel`). |
| `b` | Bias. |
| `h_t` | Hidden state baru. |

Kode:

```python
for timestep in range(timesteps):
    h_t = tanh(inputs[:, timestep, :] @ self.kernel + h_t @ self.recurrent_kernel + bias)
    outputs.append(h_t)
```

Hubungan rumus dan kode:

| Rumus | Kode |
|---|---|
| `x_t` | `inputs[:, timestep, :]` |
| `W_x` | `self.kernel` |
| `h_{t-1}` | `h_t` sebelum update |
| `W_h` | `self.recurrent_kernel` |
| `b` | `bias` |
| `tanh(...)` | `tanh(...)` |

Initial state:

```python
if initial_state is None:
    h_t = np.zeros((batch_size, self.units), dtype=np.float32)
```

Artinya `h0` bernilai nol.

### 3.6 Output RNN ke Vocabulary

Setelah sequence diproses RNN, output timestep feature dibuang:

```python
token_outputs = x[:, 1:, :]
```

Alasannya, timestep pertama adalah image feature `x_-1`, sedangkan target caption dimulai dari kata pertama.

Logits vocabulary:

```math
z_t = h_t W_o + b_o
```

Probabilitas vocabulary:

```math
P(w_t = k) =
\frac{e^{z_{t,k}}}{\sum_{r=1}^{V} e^{z_{t,r}}}
```

Kode:

```python
logits = self.output_dense.forward(token_outputs)
return softmax(logits, axis=-1)
```

Jika:

```text
token_outputs.shape = (batch, 38, hidden_units)
vocab_size = 7464
```

maka:

```text
probabilities.shape = (batch, 38, 7464)
```

---

## 4. Forward Propagation LSTM

LSTM digunakan sebagai decoder captioning seperti RNN, tetapi memiliki hidden state `h_t` dan cell state `c_t`.

### 4.1 Konsep Dasar LSTM

LSTM dirancang untuk mengatasi masalah vanishing gradient pada RNN biasa. Cell state `c_t` berperan sebagai jalur memori yang dapat mempertahankan informasi lebih lama.

Forward LSTM memiliki empat komponen utama:

1. Input gate.
2. Forget gate.
3. Candidate cell.
4. Output gate.

### 4.2 Format Bobot LSTM Keras

Keras menyimpan bobot LSTM dalam tiga array:

```text
kernel
recurrent_kernel
bias
```

Shape:

```text
kernel.shape = (input_dim, 4 * units)
recurrent_kernel.shape = (units, 4 * units)
bias.shape = (4 * units,)
```

Kode load bobot:

```python
def load_keras_weights(self, weights: list[np.ndarray] | tuple[np.ndarray, ...]) -> None:
    if len(weights) != 3:
        raise ValueError("LSTM weights must be [kernel, recurrent_kernel, bias]")
    self.kernel = np.asarray(weights[0], dtype=np.float32)
    self.recurrent_kernel = np.asarray(weights[1], dtype=np.float32)
    self.bias = np.asarray(weights[2], dtype=np.float32)
```

Dimensi `4 * units` kemudian dipisah menjadi:

```text
input gate, forget gate, candidate cell, output gate
```

### 4.3 Rumus Forward LSTM

Pertama, hitung gabungan pre-activation:

```math
z_t = x_t W_x + h_{t-1} W_h + b
```

Kemudian split:

```math
z_t = [z_i, z_f, z_c, z_o]
```

Gate:

```math
i_t = \sigma(z_i)
```

```math
f_t = \sigma(z_f)
```

```math
\tilde{c}_t = \tanh(z_c)
```

```math
o_t = \sigma(z_o)
```

Update cell state:

```math
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
```

Update hidden state:

```math
h_t = o_t \odot \tanh(c_t)
```

Keterangan:

| Simbol | Arti |
|---|---|
| `i_t` | Input gate, mengatur informasi baru yang masuk. |
| `f_t` | Forget gate, mengatur informasi lama yang dipertahankan. |
| `c~_t` | Candidate cell, kandidat memori baru. |
| `o_t` | Output gate, mengatur informasi yang keluar ke hidden state. |
| `c_t` | Cell state/memori internal. |
| `h_t` | Hidden state/output timestep. |
| `⊙` | Perkalian elemen-wise. |

### 4.4 Kode Forward LSTM

Kode implementasi:

```python
for timestep in range(timesteps):
    z = inputs[:, timestep, :] @ self.kernel + h_t @ self.recurrent_kernel + bias
    z_i, z_f, z_c, z_o = np.split(z, 4, axis=-1)
    i_t = sigmoid(z_i)
    f_t = sigmoid(z_f)
    c_hat_t = tanh(z_c)
    o_t = sigmoid(z_o)
    c_t = f_t * c_t + i_t * c_hat_t
    h_t = o_t * tanh(c_t)
    outputs.append(h_t)
```

Hubungan rumus dan kode:

| Rumus | Kode |
|---|---|
| `z_t = x_t W_x + h_{t-1} W_h + b` | `z = inputs[:, timestep, :] @ self.kernel + h_t @ self.recurrent_kernel + bias` |
| `[z_i, z_f, z_c, z_o]` | `np.split(z, 4, axis=-1)` |
| `i_t = sigmoid(z_i)` | `i_t = sigmoid(z_i)` |
| `f_t = sigmoid(z_f)` | `f_t = sigmoid(z_f)` |
| `c~_t = tanh(z_c)` | `c_hat_t = tanh(z_c)` |
| `o_t = sigmoid(z_o)` | `o_t = sigmoid(z_o)` |
| `c_t = f_t * c_{t-1} + i_t * c~_t` | `c_t = f_t * c_t + i_t * c_hat_t` |
| `h_t = o_t * tanh(c_t)` | `h_t = o_t * tanh(c_t)` |

Initial state:

```python
if initial_state is None:
    h_t = np.zeros((batch_size, self.units), dtype=np.float32)
    c_t = np.zeros((batch_size, self.units), dtype=np.float32)
```

Artinya `h0` dan `c0` bernilai nol.

### 4.5 Forward LSTM Captioner

Alur LSTM captioner sama seperti RNN captioner:

```text
image_features
-> Dense projection
-> feature timestep
caption_input_ids
-> Embedding
-> concatenate
-> LSTM layer(s)
-> remove feature timestep
-> Dense output
-> softmax
```

Kode:

```python
feature_timestep = self.feature_projection.forward(features)[:, np.newaxis, :]
token_embeddings = self.embedding.forward(token_ids)
sequence = np.concatenate([feature_timestep, token_embeddings], axis=1)

x = sequence
for layer in self.recurrent_layers:
    x = layer.forward(x)

token_outputs = x[:, 1:, :]
logits = self.output_dense.forward(token_outputs)
return softmax(logits, axis=-1)
```

Jika menggunakan LSTM, `self.recurrent_layers` berisi objek `LSTM`. Jika menggunakan RNN, `self.recurrent_layers` berisi objek `SimpleRNN`.

---

## 5. Forward Inference Caption Generation

Setelah model menghasilkan probabilitas vocabulary, caption dibuat dengan decoding.

### 5.1 Greedy Decoding

Greedy decoding memilih token dengan probabilitas terbesar pada setiap timestep.

Rumus:

```math
w_t = \arg\max_k P(w_t = k \mid w_{<t}, image)
```

Kode:

```python
for position in range(max_caption_length):
    model_input = make_padded_input(prefix, max_caption_length, vocabulary.pad_id)
    probabilities = predict_probs(np.asarray(image_feature, dtype=np.float32)[np.newaxis, :], model_input)
    next_id = int(np.argmax(probabilities[0, position]))
    if next_id == vocabulary.end_id:
        break
    generated.append(next_id)
    prefix.append(next_id)
```

Penjelasan:

1. Prefix dimulai dengan token `<start>`.
2. Model memprediksi probabilitas vocabulary.
3. Token dengan probabilitas terbesar dipilih.
4. Token tersebut ditambahkan ke prefix.
5. Proses berhenti jika token `<end>` muncul.

### 5.2 Beam Search

Beam search menyimpan beberapa kandidat caption terbaik, bukan hanya satu kandidat seperti greedy decoding.

Skor beam:

```math
score = \sum_t \log P(w_t)
```

Kode:

```python
top_ids = np.argsort(probabilities)[-beam_width:][::-1]
for token_id in top_ids:
    token_id = int(token_id)
    probability = max(float(probabilities[token_id]), 1e-12)
    next_prefix = prefix + [token_id]
    candidates.append((next_prefix, score + math.log(probability), token_id == vocabulary.end_id))
```

Kandidat terbaik dipilih dengan normalisasi panjang:

```python
beams = sorted(
    candidates,
    key=lambda item: item[1] / max(1, len(item[0]) - 1),
    reverse=True,
)[:beam_width]
```

---

## 6. Ringkasan Perbandingan Forward CNN, RNN, dan LSTM

| Model | Input | Rumus Utama | Output |
|---|---|---|---|
| CNN Conv2D | Gambar `(N, H, W, C)` | `Y = X * W + b` | Probabilitas kelas |
| LocallyConnected2D | Gambar `(N, H, W, C)` | `Y[p] = P[p]W[p] + b[p]` | Probabilitas kelas |
| SimpleRNN | Sequence embedding | `h_t = tanh(x_tW_x + h_{t-1}W_h + b)` | Probabilitas token |
| LSTM | Sequence embedding | `c_t = f_t c_{t-1} + i_t c~_t`, `h_t = o_t tanh(c_t)` | Probabilitas token |

Kesimpulan:

- CNN fokus mengekstraksi fitur spasial dari gambar.
- Conv2D memakai parameter sharing sehingga efisien.
- LocallyConnected2D tidak memakai sharing sehingga parameter jauh lebih banyak.
- RNN memproses caption sebagai sequence dengan satu hidden state.
- LSTM menambahkan cell state dan gate sehingga lebih kuat untuk dependensi jangka panjang.
- Pada image captioning, output akhir RNN/LSTM adalah distribusi probabilitas atas vocabulary pada setiap timestep.
