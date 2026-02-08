"""Pre-training tests: validate data preprocessing before any training."""
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical


def load_and_preprocess():
    """Replicate the notebook preprocessing steps."""
    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()

    # Normalize
    X_train_norm = X_train_raw.astype("float32") / 255.0
    X_test_norm = X_test_raw.astype("float32") / 255.0

    # Flatten
    X_train = X_train_norm.reshape(-1, 784)
    X_test = X_test_norm.reshape(-1, 784)

    # One-hot encode
    y_train = to_categorical(y_train_raw, num_classes=10)
    y_test = to_categorical(y_test_raw, num_classes=10)

    return X_train, X_test, y_train, y_test, y_train_raw, y_test_raw


class TestNormalization:
    def test_values_between_0_and_1(self):
        X_train, X_test, *_ = load_and_preprocess()
        assert X_train.min() >= 0.0
        assert X_train.max() <= 1.0
        assert X_test.min() >= 0.0
        assert X_test.max() <= 1.0

    def test_dtype_is_float32(self):
        X_train, X_test, *_ = load_and_preprocess()
        assert X_train.dtype == np.float32
        assert X_test.dtype == np.float32


class TestFlattening:
    def test_train_shape(self):
        X_train, *_ = load_and_preprocess()
        assert X_train.shape == (60000, 784)

    def test_test_shape(self):
        _, X_test, *_ = load_and_preprocess()
        assert X_test.shape == (10000, 784)


class TestOneHotEncoding:
    def test_train_shape(self):
        *_, y_train, _, _, _ = load_and_preprocess()
        assert y_train.shape == (60000, 10)

    def test_test_shape(self):
        *_, y_test, _, _ = load_and_preprocess()
        assert y_test.shape == (10000, 10)

    def test_rows_sum_to_one(self):
        *_, y_train, y_test, _, _ = load_and_preprocess()
        np.testing.assert_array_almost_equal(y_train.sum(axis=1), 1.0)
        np.testing.assert_array_almost_equal(y_test.sum(axis=1), 1.0)

    def test_values_are_binary(self):
        *_, y_train, _, _, _ = load_and_preprocess()
        unique_vals = np.unique(y_train)
        assert set(unique_vals) == {0.0, 1.0}

    def test_encoding_matches_labels(self):
        """Verify argmax of one-hot recovers original labels."""
        *_, y_train, _, y_train_raw, _ = load_and_preprocess()
        recovered = np.argmax(y_train, axis=1)
        np.testing.assert_array_equal(recovered, y_train_raw)
