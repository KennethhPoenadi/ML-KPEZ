from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DecoderType = Literal["rnn", "lstm"]
InjectionMode = Literal["pre", "init"]


@dataclass(frozen=True)
class CaptionDecoderConfig:
    vocab_size: int
    feature_dim: int
    max_caption_length: int
    embed_dim: int = 256
    hidden_units: int = 256
    num_recurrent_layers: int = 1
    dropout_rate: float = 0.0
    learning_rate: float = 1e-3
    decoder_type: DecoderType = "lstm"
    injection_mode: InjectionMode = "pre"
    name: str = "caption_decoder"


def build_preinject_decoder(config: CaptionDecoderConfig):
    import tensorflow as tf

    _validate_decoder_config(config)

    feature_input = tf.keras.Input(shape=(config.feature_dim,), name="image_feature")
    caption_input = tf.keras.Input(shape=(config.max_caption_length,), dtype="int32", name="caption_input")

    feature_embedding = tf.keras.layers.Dense(config.embed_dim, name="feature_projection")(feature_input)
    feature_embedding = tf.keras.layers.Reshape((1, config.embed_dim), name="feature_timestep")(feature_embedding)

    token_embedding = tf.keras.layers.Embedding(
        input_dim=config.vocab_size,
        output_dim=config.embed_dim,
        mask_zero=False,
        name="token_embedding",
    )(caption_input)
    sequence = tf.keras.layers.Concatenate(axis=1, name="preinject_sequence")(
        [feature_embedding, token_embedding]
    )

    recurrent_cls = {
        "rnn": tf.keras.layers.SimpleRNN,
        "lstm": tf.keras.layers.LSTM,
    }.get(config.decoder_type)
    if recurrent_cls is None:
        raise ValueError("decoder_type must be either 'rnn' or 'lstm'")

    x = sequence
    for layer_index in range(config.num_recurrent_layers):
        x = recurrent_cls(
            config.hidden_units,
            return_sequences=True,
            dropout=config.dropout_rate,
            name=f"{config.decoder_type}_{layer_index + 1}",
        )(x)

    token_outputs = tf.keras.layers.Lambda(lambda tensor: tensor[:, 1:, :], name="drop_feature_timestep")(x)
    logits = tf.keras.layers.Dense(config.vocab_size, name="word_logits")(token_outputs)
    outputs = tf.keras.layers.Activation("softmax", name="word_softmax")(logits)

    model = tf.keras.Model(inputs=[feature_input, caption_input], outputs=outputs, name=config.name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def build_initinject_decoder(config: CaptionDecoderConfig):
    import tensorflow as tf

    _validate_decoder_config(config)

    feature_input = tf.keras.Input(shape=(config.feature_dim,), name="image_feature")
    caption_input = tf.keras.Input(shape=(config.max_caption_length,), dtype="int32", name="caption_input")

    x = tf.keras.layers.Embedding(
        input_dim=config.vocab_size,
        output_dim=config.embed_dim,
        mask_zero=False,
        name="token_embedding",
    )(caption_input)

    recurrent_cls = {
        "rnn": tf.keras.layers.SimpleRNN,
        "lstm": tf.keras.layers.LSTM,
    }.get(config.decoder_type)
    if recurrent_cls is None:
        raise ValueError("decoder_type must be either 'rnn' or 'lstm'")

    for layer_index in range(config.num_recurrent_layers):
        h0 = tf.keras.layers.Dense(config.hidden_units, activation="tanh", name=f"init_h_{layer_index + 1}")(
            feature_input
        )
        if config.decoder_type == "lstm":
            c0 = tf.keras.layers.Dense(config.hidden_units, activation="tanh", name=f"init_c_{layer_index + 1}")(
                feature_input
            )
            initial_state = [h0, c0]
        else:
            initial_state = [h0]

        x = recurrent_cls(
            config.hidden_units,
            return_sequences=True,
            dropout=config.dropout_rate,
            name=f"{config.decoder_type}_{layer_index + 1}",
        )(x, initial_state=initial_state)

    logits = tf.keras.layers.Dense(config.vocab_size, name="word_logits")(x)
    outputs = tf.keras.layers.Activation("softmax", name="word_softmax")(logits)

    model = tf.keras.Model(inputs=[feature_input, caption_input], outputs=outputs, name=config.name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def _validate_decoder_config(config: CaptionDecoderConfig) -> None:
    if config.vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if config.feature_dim <= 0:
        raise ValueError("feature_dim must be positive")
    if config.max_caption_length <= 0:
        raise ValueError("max_caption_length must be positive")
    if config.embed_dim <= 0 or config.hidden_units <= 0:
        raise ValueError("embed_dim and hidden_units must be positive")
    if config.num_recurrent_layers <= 0:
        raise ValueError("num_recurrent_layers must be positive")
    if config.injection_mode not in {"pre", "init"}:
        raise ValueError("injection_mode must be either 'pre' or 'init'")


def build_caption_decoder(config: CaptionDecoderConfig):
    if config.injection_mode == "pre":
        return build_preinject_decoder(config)
    if config.injection_mode == "init":
        return build_initinject_decoder(config)
    raise ValueError("injection_mode must be either 'pre' or 'init'")


def build_rnn_decoder(config: CaptionDecoderConfig):
    return build_caption_decoder(
        CaptionDecoderConfig(**{**config.__dict__, "decoder_type": "rnn"})
    )


def build_lstm_decoder(config: CaptionDecoderConfig):
    return build_caption_decoder(
        CaptionDecoderConfig(**{**config.__dict__, "decoder_type": "lstm"})
    )
