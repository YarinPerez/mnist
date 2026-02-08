# Implementation Tasks: MNIST Digit Classification

## Phase 1: Project Setup
- [x] **T1.1** Initialize `uv` virtual environment
- [x] **T1.2** Install dependencies: tensorflow, keras, matplotlib, seaborn, scikit-learn, jupyter, ipykernel, pytest
- [x] **T1.3** Create `results/` and `tests/` directories
- [x] **T1.4** Create notebook `mnist_classification.ipynb`

## Phase 2: Notebook - Introduction & Setup
- [x] **T2.1** Write introduction markdown section (what MNIST is, project goal)
- [x] **T2.2** Write imports cell (tensorflow, keras, numpy, matplotlib, seaborn, sklearn)

## Phase 3: Data Loading & Exploration
- [x] **T3.1** Write markdown explaining the MNIST dataset structure
- [x] **T3.2** Load MNIST dataset using `keras.datasets.mnist`
- [x] **T3.3** Print dataset shapes and basic statistics
- [x] **T3.4** Visualize a grid of sample images with their labels
- [x] **T3.5** Plot digit class distribution (bar chart of label counts)

## Phase 4: Data Preprocessing
- [x] **T4.1** Write markdown explaining normalization, flattening, and one-hot encoding
- [x] **T4.2** Normalize pixel values from [0,255] to [0,1]
- [x] **T4.3** Flatten images from 28x28 to 784-length vectors
- [x] **T4.4** One-hot encode labels using `keras.utils.to_categorical`
- [x] **T4.5** Print preprocessed data shapes to confirm transformations

## Phase 5: Pre-Training Tests (Preprocessing)
- [x] **T5.1** Create `tests/test_preprocessing.py` - test normalization (values in [0,1]), flattening (shape is 784), one-hot encoding (shape and sum=1)
- [x] **T5.2** Run preprocessing tests and verify they pass

## Phase 6: Helper Functions & Model Builders
- [x] **T6.1** Write helper functions: `plot_training_curves`, `plot_confusion_matrix`, `evaluate_model`
- [x] **T6.2** Write model builder functions for all 4 experiment architectures

## Phase 7: Pre-Training Tests (Model Architecture)
- [x] **T7.1** Create `tests/test_model.py` - test each model has correct input shape (784,), output shape (10,), expected layer count, and correct activations (relu hidden, softmax output)
- [x] **T7.2** Run model architecture tests and verify they pass

## Phase 8: Experiment 1 - Simple Baseline
- [x] **T8.1** Write markdown explaining the baseline architecture
- [x] **T8.2** Build model: Input(784) -> Dense(128, relu) -> Dense(10, softmax)
- [x] **T8.3** Compile with Adam optimizer, categorical crossentropy, accuracy metric
- [x] **T8.4** Print model summary
- [x] **T8.5** Train for 10 epochs with validation split=0.1
- [x] **T8.6** Plot training/validation accuracy and loss curves, save to `results/`
- [x] **T8.7** Evaluate on test set, print accuracy
- [x] **T8.8** Generate confusion matrix heatmap, save to `results/`
- [x] **T8.9** Generate and save classification report to `results/`

## Phase 9: Experiment 2 - Deeper Network
- [x] **T9.1** Write markdown explaining deeper architecture and why more layers help
- [x] **T9.2** Build model: Input(784) -> Dense(256, relu) -> Dense(128, relu) -> Dense(10, softmax)
- [x] **T9.3** Compile, train, evaluate (same as T8.3-T8.9)
- [x] **T9.4** Save confusion matrix and training curves to `results/`
- [x] **T9.5** Save classification report to `results/`

## Phase 10: Experiment 3 - Dropout Regularization
- [x] **T10.1** Write markdown explaining dropout and overfitting
- [x] **T10.2** Build model: Dense(256, relu) -> Dropout(0.3) -> Dense(128, relu) -> Dropout(0.2) -> Dense(10, softmax)
- [x] **T10.3** Compile, train, evaluate
- [x] **T10.4** Save confusion matrix and training curves to `results/`
- [x] **T10.5** Save classification report to `results/`

## Phase 11: Experiment 4 - Tuned Network
- [x] **T11.1** Write markdown explaining wider/deeper network benefits
- [x] **T11.2** Build model: Dense(512, relu) -> Dropout(0.3) -> Dense(256, relu) -> Dropout(0.2) -> Dense(128, relu) -> Dense(10, softmax)
- [x] **T11.3** Compile, train, evaluate
- [x] **T11.4** Save confusion matrix and training curves to `results/`
- [x] **T11.5** Save classification report to `results/`

## Phase 12: Post-Training Tests (Evaluation Helpers)
- [x] **T12.1** Create `tests/test_evaluation.py` - test confusion matrix shape (10x10), classification report format, plot file creation
- [x] **T12.2** Run evaluation tests and verify they pass

## Phase 13: Results Comparison
- [x] **T13.1** Write markdown summarizing all experiments
- [x] **T13.2** Create comparison bar chart of all experiment accuracies, save to `results/`
- [x] **T13.3** Show sample correct and incorrect predictions from best model
- [x] **T13.4** Write conclusions markdown cell

## Phase 14: Documentation
- [x] **T14.1** Create README.md with project description and embedded result images
