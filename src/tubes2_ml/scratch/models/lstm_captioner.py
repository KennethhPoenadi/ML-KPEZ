from __future__ import annotations

import numpy as np

from tubes2_ml.scratch.layers.activations import softmax
from tubes2_ml.scratch.layers.dense import Dense
from tubes2_ml.scratch.layers.embedding import Embedding
from tubes2_ml.scratch.layers.recurrent import LSTM


class ScratchLSTMCaptioner:
    def __init__(
        self,
        feature_projection: Dense,
        embedding: Embedding,
        recurrent_layers: list[LSTM],
        output_dense: Dense,
    ):
        if not recurrent_layers:
            raise ValueError("At least one LSTM layer is required")
        self.feature_projection = feature_projection
        self.embedding = embedding
        self.recurrent_layers = recurrent_layers
        self.output_dense = output_dense

    @classmethod
    def from_keras_model(cls, keras_model) -> "ScratchLSTMCaptioner":
        feature_projection = Dense()
        feature_projection.load_keras_weights(keras_model.get_layer("feature_projection").get_weights())

        embedding = Embedding()
        embedding.load_keras_weights(keras_model.get_layer("token_embedding").get_weights())

        recurrent_layers: list[LSTM] = []
        index = 1
        while True:
            try:
                keras_layer = keras_model.get_layer(f"lstm_{index}")
            except ValueError:
                break
            layer = LSTM()
            layer.load_keras_weights(keras_layer.get_weights())
            recurrent_layers.append(layer)
            index += 1

        output_dense = Dense()
        output_dense.load_keras_weights(keras_model.get_layer("word_logits").get_weights())
        return cls(feature_projection, embedding, recurrent_layers, output_dense)

    def forward(self, image_features: np.ndarray, caption_input_ids: np.ndarray) -> np.ndarray:
        features = np.asarray(image_features, dtype=np.float32)
        if features.ndim == 1:
            features = features[np.newaxis, :]

        token_ids = np.asarray(caption_input_ids, dtype=np.int32)
        if token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, :]

        feature_timestep = self.feature_projection.forward(features)[:, np.newaxis, :]
        token_embeddings = self.embedding.forward(token_ids)
        sequence = np.concatenate([feature_timestep, token_embeddings], axis=1)

        x = sequence
        for layer in self.recurrent_layers:
            x = layer.forward(x)

        token_outputs = x[:, 1:, :]
        logits = self.output_dense.forward(token_outputs)
        return softmax(logits, axis=-1)


def build_scratch_lstm_captioner_from_keras(keras_model) -> ScratchLSTMCaptioner:
    return ScratchLSTMCaptioner.from_keras_model(keras_model)
