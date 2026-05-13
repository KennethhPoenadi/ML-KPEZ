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
        injection_mode: str = "pre",
        context_fusion: Dense | None = None,
    ):
        if not recurrent_layers:
            raise ValueError("At least one LSTM layer is required")
        if injection_mode not in {"pre", "init"}:
            raise ValueError("injection_mode must be either 'pre' or 'init'")
        if injection_mode == "init" and context_fusion is None:
            raise ValueError("context_fusion is required for init-inject models")
        self.feature_projection = feature_projection
        self.embedding = embedding
        self.recurrent_layers = recurrent_layers
        self.output_dense = output_dense
        self.injection_mode = injection_mode
        self.context_fusion = context_fusion

    @classmethod
    def from_keras_model(cls, keras_model) -> "ScratchLSTMCaptioner":
        layer_names = {layer.name for layer in keras_model.layers}
        injection_mode = "init" if "init_feature_projection" in layer_names else "pre"

        feature_layer_name = "init_feature_projection" if injection_mode == "init" else "feature_projection"
        feature_projection = Dense(activation="tanh" if injection_mode == "init" else None)
        feature_projection.load_keras_weights(keras_model.get_layer(feature_layer_name).get_weights())

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

        context_fusion = None
        if injection_mode == "init":
            context_fusion = Dense(activation="tanh")
            context_fusion.load_keras_weights(keras_model.get_layer("init_context_fusion").get_weights())

        return cls(
            feature_projection,
            embedding,
            recurrent_layers,
            output_dense,
            injection_mode=injection_mode,
            context_fusion=context_fusion,
        )

    def forward(self, image_features: np.ndarray, caption_input_ids: np.ndarray) -> np.ndarray:
        features = np.asarray(image_features, dtype=np.float32)
        if features.ndim == 1:
            features = features[np.newaxis, :]

        token_ids = np.asarray(caption_input_ids, dtype=np.int32)
        if token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, :]

        token_embeddings = self.embedding.forward(token_ids)

        if self.injection_mode == "pre":
            feature_timestep = self.feature_projection.forward(features)[:, np.newaxis, :]
            x = np.concatenate([feature_timestep, token_embeddings], axis=1)
        else:
            x = token_embeddings

        for layer in self.recurrent_layers:
            x = layer.forward(x)

        if self.injection_mode == "pre":
            token_outputs = x[:, 1:, :]
        else:
            feature_context = self.feature_projection.forward(features)[:, np.newaxis, :]
            feature_context = np.repeat(feature_context, token_ids.shape[1], axis=1)
            token_outputs = self.context_fusion.forward(np.concatenate([x, feature_context], axis=-1))

        logits = self.output_dense.forward(token_outputs)
        return softmax(logits, axis=-1)


def build_scratch_lstm_captioner_from_keras(keras_model) -> ScratchLSTMCaptioner:
    return ScratchLSTMCaptioner.from_keras_model(keras_model)
