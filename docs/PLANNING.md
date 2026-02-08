# Planning & Architecture: MNIST Digit Classification

## 1. Architecture Overview

```
mnist/
├── CLAUDE.md                        # Project instructions
├── README.md                        # Project summary with results
├── mnist_classification.ipynb       # Main Jupyter notebook
├── docs/
│   ├── PRD.md                       # Product requirements
│   ├── TASKS.md                     # Task tracking
│   └── PLANNING.md                  # This file
└── results/
    ├── sample_digits.png            # MNIST sample visualization
    ├── class_distribution.png       # Label distribution bar chart
    ├── confusion_matrix_exp1.png    # Experiment 1 confusion matrix
    ├── confusion_matrix_exp2.png    # Experiment 2 confusion matrix
    ├── confusion_matrix_exp3.png    # Experiment 3 confusion matrix
    ├── confusion_matrix_exp4.png    # Experiment 4 confusion matrix
    ├── training_curves_exp1.png     # Experiment 1 accuracy/loss
    ├── training_curves_exp2.png     # Experiment 2 accuracy/loss
    ├── training_curves_exp3.png     # Experiment 3 accuracy/loss
    ├── training_curves_exp4.png     # Experiment 4 accuracy/loss
    ├── experiment_comparison.png    # Accuracy comparison bar chart
    ├── sample_predictions.png       # Correct/incorrect predictions
    ├── classification_report_exp1.txt
    ├── classification_report_exp2.txt
    ├── classification_report_exp3.txt
│   └── classification_report_exp4.txt
└── tests/
    ├── test_preprocessing.py       # Pre-training: data preprocessing
    ├── test_model.py               # Pre-training: model architecture
    └── test_evaluation.py          # Post-training: evaluation helpers
```

## 2. Implementation Strategy

### Single Notebook Approach
Everything lives in one Jupyter notebook (`mnist_classification.ipynb`). This is the right choice because:
- The assignment requires a notebook deliverable
- It allows interleaving explanations with code naturally
- Results are visible inline alongside the code that produced them

### Helper Functions
To avoid repetition across 4 experiments, we define reusable helper functions early in the notebook:
- `plot_training_curves(history, exp_name)` - plots and saves accuracy/loss curves
- `plot_confusion_matrix(y_true, y_pred, exp_name)` - plots and saves confusion matrix
- `evaluate_model(model, X_test, y_test, exp_name)` - runs full evaluation pipeline

This keeps each experiment section clean and focused on architecture differences.

## 3. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Validation approach | `validation_split=0.1` during training | Simple, uses 10% of training data for validation monitoring |
| Epochs | 10 per experiment | Enough for convergence on MNIST, fast enough for iteration |
| Batch size | 128 | Good balance of speed and gradient stability for MNIST |
| Optimizer | Adam for all experiments | Consistent comparison; Adam is robust default |
| Loss function | Categorical crossentropy | Standard for multi-class with one-hot labels |
| Random seed | Set `tf.random.set_seed(42)` | Reproducible results across runs |

## 4. Testing Strategy

Tests are split into **pre-training** and **post-training** stages to catch errors early and avoid wasting compute.

### Pre-Training Tests (run BEFORE any training)
| Test File | When | What It Tests |
|-----------|------|---------------|
| `test_preprocessing.py` | After Phase 4 (preprocessing) | Normalization values in [0,1]; flattening yields shape (N, 784); one-hot encoding shape (N, 10) with rows summing to 1 |
| `test_model.py` | After Phase 6 (model builders) | Each model has correct input shape (784,), output shape (10,), expected layer count, correct activations (relu hidden, softmax output) |

**Why pre-training?** If preprocessing is wrong (e.g., wrong shape, unnormalized data) or a model architecture is broken, we'd waste minutes of training only to get garbage results. These tests catch such issues in seconds.

### Post-Training Tests (run AFTER experiments)
| Test File | When | What It Tests |
|-----------|------|---------------|
| `test_evaluation.py` | After Phase 11 (all experiments) | Confusion matrix shape (10x10), classification report format, plot files actually saved to disk |

All tests use small data subsets and 1 epoch to run fast (~seconds).

## 5. Task Tracking

Tasks are tracked in `docs/TASKS.md`. Each task will be marked `[x]` when completed. Implementation follows the phase order (Phase 1 through Phase 14) sequentially.

## 6. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| TensorFlow install issues | Use `uv` with explicit version pinning |
| Long training times | MNIST is small; 10 epochs ~30s per experiment on CPU |
| Notebook kernel crashes | Save results to files incrementally, not just at the end |
| Non-reproducible results | Set random seeds before each experiment |
