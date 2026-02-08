"""Post-training tests: validate evaluation outputs and saved files."""
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


class TestResultFilesExist:
    """Verify all expected result files were saved."""

    def test_confusion_matrices_exist(self):
        for i in range(1, 5):
            path = os.path.join(RESULTS_DIR, f"confusion_matrix_exp{i}.png")
            assert os.path.isfile(path), f"Missing: {path}"

    def test_training_curves_exist(self):
        for i in range(1, 5):
            path = os.path.join(RESULTS_DIR, f"training_curves_exp{i}.png")
            assert os.path.isfile(path), f"Missing: {path}"

    def test_classification_reports_exist(self):
        for i in range(1, 5):
            path = os.path.join(RESULTS_DIR, f"classification_report_exp{i}.txt")
            assert os.path.isfile(path), f"Missing: {path}"

    def test_comparison_chart_exists(self):
        path = os.path.join(RESULTS_DIR, "experiment_comparison.png")
        assert os.path.isfile(path), f"Missing: {path}"

    def test_sample_digits_exist(self):
        path = os.path.join(RESULTS_DIR, "sample_digits.png")
        assert os.path.isfile(path), f"Missing: {path}"

    def test_sample_predictions_exist(self):
        path = os.path.join(RESULTS_DIR, "sample_predictions.png")
        assert os.path.isfile(path), f"Missing: {path}"


class TestClassificationReports:
    """Verify classification reports have valid content."""

    def test_reports_contain_accuracy(self):
        for i in range(1, 5):
            path = os.path.join(RESULTS_DIR, f"classification_report_exp{i}.txt")
            with open(path) as f:
                content = f.read()
            assert "Test Accuracy:" in content
            assert "precision" in content
            assert "recall" in content

    def test_accuracy_is_reasonable(self):
        """All experiments should achieve at least 95% on MNIST."""
        for i in range(1, 5):
            path = os.path.join(RESULTS_DIR, f"classification_report_exp{i}.txt")
            with open(path) as f:
                for line in f:
                    if line.startswith("Test Accuracy:"):
                        acc = float(line.split(":")[1].strip())
                        assert acc >= 0.95, (
                            f"exp{i} accuracy {acc} is below 95%"
                        )


class TestConfusionMatrixLogic:
    """Verify confusion matrix computation is correct."""

    def test_shape_is_10x10(self):
        y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1])
        y_pred = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2])
        cm = confusion_matrix(y_true, y_pred, labels=range(10))
        assert cm.shape == (10, 10)

    def test_diagonal_represents_correct(self):
        y_true = np.array([0, 0, 1, 1, 2])
        y_pred = np.array([0, 0, 1, 1, 2])
        cm = confusion_matrix(y_true, y_pred, labels=range(3))
        assert cm[0, 0] == 2
        assert cm[1, 1] == 2
        assert cm[2, 2] == 1
        assert cm.sum() == 5

    def test_off_diagonal_represents_errors(self):
        y_true = np.array([0, 0, 1])
        y_pred = np.array([0, 1, 1])  # One mistake: true=0, pred=1
        cm = confusion_matrix(y_true, y_pred, labels=range(2))
        assert cm[0, 1] == 1  # row=actual 0, col=predicted 1


class TestResultFilesNotEmpty:
    """Verify result files have actual content (not 0 bytes)."""

    def test_png_files_have_content(self):
        for name in [
            "confusion_matrix_exp1.png",
            "training_curves_exp1.png",
            "experiment_comparison.png",
            "sample_digits.png",
            "sample_predictions.png",
        ]:
            path = os.path.join(RESULTS_DIR, name)
            size = os.path.getsize(path)
            assert size > 1000, f"{name} is only {size} bytes"
