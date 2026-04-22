import os
import unittest
import numpy as np
import tensorflow as tf

from src.multistep_forecast import (
    mae,
    rmse,
    make_windows,
    time_split,
    build_model,
    recursive_rollout_one_step,
    recursive_rollout_k_step_stride_k,
    recursive_rollout_k_step_stride_1,
    train_model,
    horizon_errors,
)


class DummyOneStep:
    """A deterministic 1-step model for testing rollout logic.

    predict(x) returns last input value + 1.
    """

    def predict(self, X, verbose=0):
        last = float(X[0, -1, 0])
        return np.array([[last + 1.0]], dtype=np.float32)


class DummyKStep:
    """A deterministic K-step model for testing rollout logic.

    predict(x) returns a vector [last+1, last+2, ..., last+k].
    """

    def __init__(self, k: int):
        self.k = k

    def predict(self, X, verbose=0):
        last = float(X[0, -1, 0])
        block = np.arange(1, self.k + 1, dtype=np.float32) + last
        return block[None, :]


class TestMetrics(unittest.TestCase):
    def test_mae_rmse(self):
        y_true = np.array([0, 1, 2], dtype=np.float32)
        y_pred = np.array([0, 2, 1], dtype=np.float32)
        self.assertAlmostEqual(mae(y_true, y_pred), 2 / 3, places=6)

        y_true = np.array([0, 0], dtype=np.float32)
        y_pred = np.array([3, 4], dtype=np.float32)
        self.assertAlmostEqual(rmse(y_true, y_pred), np.sqrt(12.5), places=6)


class TestWindowing(unittest.TestCase):
    def test_make_windows_h1(self):
        series = np.arange(10, dtype=np.float32)
        X, y = make_windows(series, window=3, horizon=1)
        self.assertEqual(X.shape, (7, 3, 1))
        self.assertEqual(y.shape, (7, 1))
        np.testing.assert_allclose(X[0, :, 0], [0, 1, 2])
        np.testing.assert_allclose(y[0, 0], 3)

    def test_make_windows_h2(self):
        series = np.arange(10, dtype=np.float32)
        X, y = make_windows(series, window=3, horizon=2)
        # N = 10 - 3 - 2 + 1 = 6
        self.assertEqual(X.shape, (6, 3, 1))
        self.assertEqual(y.shape, (6, 2))
        np.testing.assert_allclose(X[0, :, 0], [0, 1, 2])
        np.testing.assert_allclose(y[0, :], [3, 4])

    def test_time_split_boundary(self):
        series = np.arange(60, dtype=np.float32)
        X, y = make_windows(series, window=5, horizon=1)
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = time_split(X, y, 0.6, 0.2)
        self.assertEqual(len(X_tr) + len(X_val) + len(X_te), len(X))
        # ensure no shuffle: boundary targets should be consecutive
        self.assertAlmostEqual(float(y_tr[-1, 0] + 1.0), float(y_val[0, 0]), places=6)


class TestRollouts(unittest.TestCase):
    def test_one_step_rollout(self):
        model = DummyOneStep()
        init = np.array([0, 1, 2, 3], dtype=np.float32)
        pred = recursive_rollout_one_step(model, init, horizon=5)
        np.testing.assert_allclose(pred, [4, 5, 6, 7, 8])

    def test_k_step_stride_k_rollout(self):
        k = 4
        model = DummyKStep(k)
        init = np.array([0, 1, 2, 3], dtype=np.float32)
        pred = recursive_rollout_k_step_stride_k(model, init, k=k, horizon=8)
        np.testing.assert_allclose(pred, [4, 5, 6, 7, 8, 9, 10, 11])

    def test_k_step_stride_1_rollout(self):
        k = 4
        model = DummyKStep(k)
        init = np.array([0, 1, 2, 3], dtype=np.float32)
        pred = recursive_rollout_k_step_stride_1(model, init, k=k, horizon=5)
        np.testing.assert_allclose(pred, [4, 5, 6, 7, 8])


class TestModelSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.keras.utils.set_random_seed(123)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

    def test_build_model_shapes(self):
        model = build_model(window=20, output_dim=5, n_units=8, dense_units=8, dropout=0.0)
        self.assertEqual(model.input_shape, (None, 20, 1))
        self.assertEqual(model.output_shape, (None, 5))

    def test_train_model_smoke(self):
        rng = np.random.default_rng(11)
        t = np.arange(900, dtype=np.float32)
        series = (
            0.001 * t
            + 1.6 * np.sin(2 * np.pi * t / 53.0)
            + 0.7 * np.sin(2 * np.pi * t / 17.0)
            + rng.normal(0, 0.12, size=len(t)).astype(np.float32)
        )

        model, X_test, y_test = train_model(series, window=30, horizon=20, epochs=2, batch_size=32, seed=123, verbose=0)
        y_pred = model.predict(X_test[:3], verbose=0)
        self.assertEqual(y_pred.shape, (3, 20))
        self.assertTrue(np.isfinite(y_pred).all())


if __name__ == "__main__":
    unittest.main()
