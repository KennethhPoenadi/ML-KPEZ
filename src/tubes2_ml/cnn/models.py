from __future__ import annotations
from dataclasses import dataclass, field
import tensorflow as tf

@dataclass(frozen=True)
class SharedConvCNNConfig:
    input_shape: tuple[int, int, int] = (150, 150, 3)  # input image shape as (height, width, channels)
    num_classes: int = 6  # intel dataset has 6 classes
    conv_filters: tuple[int, ...] = (32, 64)  # number of filters for each Conv2D layer
    kernel_sizes: tuple[int, ...] = (3, 3)  # kernel size for each Conv2D layer
    pooling_type: str = "max"  # either "max" or "average"
    dense_units: tuple[int, ...] = (128,)  # hidden dense layers after convolution blocks
    dropout_rate: float = 0.3  # dropout after each hidden dense layer
    learning_rate: float = 1e-3  # adam learning rate
    activation: str = "relu"  # activation used by Conv2D and hidden Dense layers
    name: str = "shared_conv_cnn"  # keras model name and artifact prefix
    compile_model: bool = True  # compile model for training when true
    metrics: tuple[str, ...] = field(default_factory=lambda: ("accuracy",))  # keras training metrics

def _repeat_or_validate(values: tuple[int, ...], expected_length: int, field_name: str) -> tuple[int, ...]:
    if len(values) == expected_length:
        return values
    if len(values) == 1:
        return values * expected_length
    raise ValueError(f"{field_name} must have length 1 or {expected_length}")

def _validate_config(config: SharedConvCNNConfig) -> tuple[int, ...]:
    if config.pooling_type not in {"max", "average"}:
        raise ValueError("pooling_type must be either 'max' or 'average'")
    if not config.conv_filters:
        raise ValueError("conv_filters must contain at least one layer")
    if config.num_classes <= 1:
        raise ValueError("num_classes must be greater than 1")
    if config.dropout_rate < 0 or config.dropout_rate >= 1:
        raise ValueError("dropout_rate must be in the range [0, 1)")
    return _repeat_or_validate(
        tuple(config.kernel_sizes),
        expected_length=len(config.conv_filters),
        field_name="kernel_sizes",
    )

def _add_conv_block(x,filters: int,kernel_size: int,pooling_type: str,activation: str,index: int):
    x = tf.keras.layers.Conv2D(filters=filters,kernel_size=(kernel_size, kernel_size),padding="same",activation=activation,name=f"conv_{index}")(x)

    pooling_layer = (
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name=f"pool_{index}")
        if pooling_type == "max"
        else tf.keras.layers.AveragePooling2D(pool_size=(2, 2), name=f"pool_{index}")
    )

    return pooling_layer(x)

def _add_dense_head(x, config: SharedConvCNNConfig):
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)

    for dense_index, units in enumerate(config.dense_units, start=1):
        x = tf.keras.layers.Dense(units, activation=config.activation, name=f"dense_{dense_index}")(x)
        if config.dropout_rate > 0:
            x = tf.keras.layers.Dropout(config.dropout_rate, name=f"dropout_{dense_index}")(x)

    return tf.keras.layers.Dense(config.num_classes,activation="softmax",name="class_probabilities")(x)

def build_shared_conv_cnn(config: SharedConvCNNConfig) -> tf.keras.Model:
    kernel_sizes = _validate_config(config)
    inputs = tf.keras.Input(shape=config.input_shape, name="image")
    x = inputs

    for layer_index, (filters, kernel_size) in enumerate(zip(config.conv_filters, kernel_sizes), start=1):
        x = _add_conv_block(
            x,
            filters=filters,
            kernel_size=kernel_size,
            pooling_type=config.pooling_type,
            activation=config.activation,
            index=layer_index,
        )

    outputs = _add_dense_head(x, config)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=config.name)

    if config.compile_model:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=list(config.metrics))

    return model