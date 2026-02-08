"""Pre-training tests: validate model architectures before training."""
import numpy as np
import tensorflow as tf
from keras import Sequential, layers


def build_model_exp1():
    return Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])


def build_model_exp2():
    return Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])


def build_model_exp3():
    return Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10, activation="softmax"),
    ])


def build_model_exp4():
    return Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])


ALL_BUILDERS = [build_model_exp1, build_model_exp2, build_model_exp3, build_model_exp4]


class TestInputOutputShapes:
    def test_all_models_accept_784_input(self):
        for builder in ALL_BUILDERS:
            model = builder()
            assert model.input_shape == (None, 784), (
                f"{builder.__name__} input shape is {model.input_shape}"
            )

    def test_all_models_output_10_classes(self):
        for builder in ALL_BUILDERS:
            model = builder()
            assert model.output_shape == (None, 10), (
                f"{builder.__name__} output shape is {model.output_shape}"
            )


class TestActivations:
    def _get_dense_layers(self, model):
        return [l for l in model.layers if isinstance(l, layers.Dense)]

    def test_hidden_layers_use_relu(self):
        for builder in ALL_BUILDERS:
            model = builder()
            dense_layers = self._get_dense_layers(model)
            # All hidden (non-last) dense layers should use relu
            for layer in dense_layers[:-1]:
                cfg = layer.get_config()
                act = cfg["activation"]
                assert act == "relu", (
                    f"{builder.__name__}: {layer.name} uses {act}, expected relu"
                )

    def test_output_layer_uses_softmax(self):
        for builder in ALL_BUILDERS:
            model = builder()
            dense_layers = self._get_dense_layers(model)
            cfg = dense_layers[-1].get_config()
            assert cfg["activation"] == "softmax", (
                f"{builder.__name__}: output uses {cfg['activation']}"
            )


class TestLayerCounts:
    def _count_dense(self, model):
        return len([l for l in model.layers if isinstance(l, layers.Dense)])

    def _count_dropout(self, model):
        return len([l for l in model.layers if isinstance(l, layers.Dropout)])

    def test_exp1_layers(self):
        model = build_model_exp1()
        assert self._count_dense(model) == 2
        assert self._count_dropout(model) == 0

    def test_exp2_layers(self):
        model = build_model_exp2()
        assert self._count_dense(model) == 3
        assert self._count_dropout(model) == 0

    def test_exp3_layers(self):
        model = build_model_exp3()
        assert self._count_dense(model) == 3
        assert self._count_dropout(model) == 2

    def test_exp4_layers(self):
        model = build_model_exp4()
        assert self._count_dense(model) == 4
        assert self._count_dropout(model) == 2


class TestForwardPass:
    """Verify models can process a batch without errors."""
    def test_all_models_forward_pass(self):
        dummy_input = np.random.rand(4, 784).astype("float32")
        for builder in ALL_BUILDERS:
            model = builder()
            output = model(dummy_input, training=False)
            assert output.shape == (4, 10)
            # Softmax outputs should sum to ~1
            sums = tf.reduce_sum(output, axis=1).numpy()
            np.testing.assert_array_almost_equal(sums, 1.0, decimal=5)
