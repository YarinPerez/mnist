# MNIST Digit Classification with Keras

A neural network classifier for handwritten digits (0-9) using the MNIST dataset. Includes 4 experiments with increasing complexity to demonstrate the effects of depth, width, and dropout regularization.

## Results Summary

| Experiment | Architecture | Test Accuracy |
|-----------|-------------|---------------|
| 1 - Baseline | Dense(128) | 97.31% |
| 2 - Deeper | Dense(256) -> Dense(128) | 96.86% |
| 3 - Dropout | Dense(256) -> Drop(0.3) -> Dense(128) -> Drop(0.2) | 98.11% |
| 4 - Tuned | Dense(512) -> Drop(0.3) -> Dense(256) -> Drop(0.2) -> Dense(128) | 98.12% |

### Accuracy Comparison
![Experiment Comparison](results/experiment_comparison.png)

### Sample Digits
![Sample MNIST Digits](results/sample_digits.png)

### Confusion Matrices

| Experiment 1 (Baseline) | Experiment 2 (Deeper) |
|---|---|
| ![CM Exp1](results/confusion_matrix_exp1.png) | ![CM Exp2](results/confusion_matrix_exp2.png) |

| Experiment 3 (Dropout) | Experiment 4 (Tuned) |
|---|---|
| ![CM Exp3](results/confusion_matrix_exp3.png) | ![CM Exp4](results/confusion_matrix_exp4.png) |

### Training Curves

| Experiment 1 | Experiment 2 |
|---|---|
| ![TC Exp1](results/training_curves_exp1.png) | ![TC Exp2](results/training_curves_exp2.png) |

| Experiment 3 | Experiment 4 |
|---|---|
| ![TC Exp3](results/training_curves_exp3.png) | ![TC Exp4](results/training_curves_exp4.png) |

### Sample Predictions
![Sample Predictions](results/sample_predictions.png)

## Key Takeaways

1. **Baseline works well** - Even a single hidden layer achieves ~97% on MNIST
2. **Deeper alone can hurt** - Exp 2 shows that adding depth without regularization can lead to overfitting
3. **Dropout is critical** - Exp 3 jumps to 98.11% by adding dropout regularization
4. **Width + depth + dropout** - Exp 4 combines all three for the best result (98.12%)

## Setup

```bash
uv sync
uv run jupyter notebook mnist_classification.ipynb
```

## Testing

```bash
uv run pytest tests/ -v
```

## Project Structure

```
mnist/
├── mnist_classification.ipynb   # Main notebook with all experiments
├── results/                     # Saved plots and metrics
├── tests/                       # Unit tests
│   ├── test_preprocessing.py    # Pre-training: data validation
│   ├── test_model.py            # Pre-training: architecture validation
│   └── test_evaluation.py       # Post-training: output validation
└── docs/
    ├── PRD.md                   # Product requirements
    ├── TASKS.md                 # Task tracking
    └── PLANNING.md              # Architecture & planning
```
