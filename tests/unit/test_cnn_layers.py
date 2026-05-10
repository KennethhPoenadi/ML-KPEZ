import numpy as np

from tubes2_ml.scratch.layers.conv import Conv2D
from tubes2_ml.scratch.layers.dense import Dense
from tubes2_ml.scratch.layers.flatten import Flatten
from tubes2_ml.scratch.layers.locally_connected import LocallyConnected2D
from tubes2_ml.scratch.layers.pooling import (AveragePooling2D,GlobalAveragePooling2D,GlobalMaxPooling2D,MaxPooling2D)

def test_conv2d_forward_valid_padding_batch_input():
    x = np.array(
        [
            [
                [[1.0], [2.0], [3.0]],
                [[4.0], [5.0], [6.0]],
                [[7.0], [8.0], [9.0]],
            ],
            [
                [[2.0], [3.0], [4.0]],
                [[5.0], [6.0], [7.0]],
                [[8.0], [9.0], [10.0]],
            ],
        ],
        dtype=np.float32,
    )
    kernel = np.array([[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]], dtype=np.float32)
    bias = np.array([0.5], dtype=np.float32)

    layer = Conv2D(strides=1, padding="valid", activation=None)
    layer.load_keras_weights([kernel, bias])

    output = layer.forward(x)

    expected = np.array(
        [
            [[[6.5], [8.5]], [[12.5], [14.5]]],
            [[[8.5], [10.5]], [[14.5], [16.5]]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(output, expected)

def test_conv2d_forward_same_padding_relu():
    x = np.array([[[[-1.0], [2.0]], [[3.0], [-4.0]]]], dtype=np.float32)
    kernel = np.ones((1, 1, 1, 1), dtype=np.float32)
    bias = np.array([0.0], dtype=np.float32)

    layer = Conv2D(kernel=kernel, bias=bias, padding="same", activation="relu")

    expected = np.array([[[[0.0], [2.0]], [[3.0], [0.0]]]], dtype=np.float32)
    np.testing.assert_allclose(layer.forward(x), expected)

def test_locally_connected2d_forward_uses_different_kernel_per_position():
    x = np.array(
        [
            [
                [[1.0], [2.0], [3.0]],
                [[4.0], [5.0], [6.0]],
                [[7.0], [8.0], [9.0]],
            ]
        ],
        dtype=np.float32,
    )
    kernel = np.array(
        [
            [[1.0], [0.0], [0.0], [0.0]],
            [[0.0], [1.0], [0.0], [0.0]],
            [[0.0], [0.0], [1.0], [0.0]],
            [[0.0], [0.0], [0.0], [1.0]],
        ],
        dtype=np.float32,
    )
    bias = np.array([[0.0], [10.0], [20.0], [30.0]], dtype=np.float32)

    layer = LocallyConnected2D(kernel_size=(2, 2), strides=1, padding="valid")
    layer.load_keras_weights([kernel, bias])

    output = layer.forward(x)

    expected = np.array([[[[1.0], [12.0]], [[24.0], [38.0]]]], dtype=np.float32)
    np.testing.assert_allclose(output, expected)

def test_max_and_average_pooling_forward():
    x = np.array(
        [[[[1.0], [2.0]], [[3.0], [4.0]]]],
        dtype=np.float32,
    )

    max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))

    np.testing.assert_allclose(max_pool.forward(x), np.array([[[[4.0]]]], dtype=np.float32))
    np.testing.assert_allclose(avg_pool.forward(x), np.array([[[[2.5]]]], dtype=np.float32))

def test_global_pooling_forward():
    x = np.array(
        [
            [
                [[1.0, 5.0], [2.0, 4.0]],
                [[3.0, 3.0], [4.0, 2.0]],
            ]
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        GlobalAveragePooling2D().forward(x),
        np.array([[2.5, 3.5]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        GlobalMaxPooling2D().forward(x),
        np.array([[4.0, 5.0]], dtype=np.float32),
    )

def test_flatten_uses_c_order_like_keras():
    x = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float32)

    output = Flatten().forward(x)

    np.testing.assert_allclose(output, np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))

def test_dense_forward_loads_keras_weights_and_applies_softmax():
    x = np.array([[1.0, 2.0]], dtype=np.float32)
    kernel = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    bias = np.array([0.0, 1.0], dtype=np.float32)

    layer = Dense(activation="softmax")
    layer.load_keras_weights([kernel, bias])
    output = layer.forward(x)

    expected_logits = np.array([[1.0, 3.0]], dtype=np.float32)
    expected = np.exp(expected_logits - expected_logits.max(axis=-1, keepdims=True))
    expected = expected / expected.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(output, expected)