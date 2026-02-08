# Product Requirements Document: MNIST Digit Classification

## 1. Project Overview

### What is MNIST?
MNIST (Modified National Institute of Standards and Technology) is the "Hello World" of machine learning. It contains **70,000 grayscale images** of handwritten digits (0-9), each sized **28x28 pixels**. The dataset is split into:
- **Training set:** 60,000 images
- **Test set:** 10,000 images

### Why MNIST?
MNIST is the ideal starting point for learning neural networks because:
- It's small enough to train quickly on any machine (no GPU required)
- It's complex enough to demonstrate real neural network concepts
- It has a known "ceiling" (~99.7% accuracy with CNNs), so you can benchmark your results
- Every image is pre-centered and size-normalized, removing the need for complex preprocessing

### Project Goal
Build a neural network classifier using Keras that achieves high accuracy on MNIST digit classification, while **learning and documenting every step** of the process. We will run multiple experiments to understand how different configurations affect performance.

---

## 2. Technical Concepts (Educational)

### 2.1 Neural Network Basics
A neural network is a function that maps inputs to outputs through layers of learned transformations:

```
Input (784 pixels) --> Hidden Layer(s) --> Output (10 digit probabilities)
```

Each layer performs: `output = activation(weights * input + bias)`

- **Weights:** Learnable parameters that scale input signals
- **Bias:** Learnable offset added before activation
- **Activation function:** Non-linear function that allows the network to learn complex patterns

### 2.2 Key Concepts We'll Explore

| Concept | What It Does | Why It Matters |
|---------|-------------|----------------|
| **Normalization** | Scales pixel values from [0,255] to [0,1] | Helps gradient descent converge faster |
| **One-hot encoding** | Converts label `5` to `[0,0,0,0,0,1,0,0,0,0]` | Required for categorical cross-entropy loss |
| **Flattening** | Converts 28x28 image to 784-length vector | Dense layers expect 1D input |
| **Dense layers** | Fully connected layers where every neuron connects to every neuron in the next layer | Core building block for classification |
| **Activation: ReLU** | `max(0, x)` - zeroes out negative values | Prevents vanishing gradients, fast to compute |
| **Activation: Softmax** | Converts raw scores to probabilities (sum=1) | Gives us interpretable digit probabilities |
| **Loss: Categorical Crossentropy** | Measures how far predicted probabilities are from true labels | Standard loss for multi-class classification |
| **Optimizer: Adam** | Adaptive learning rate optimization | Usually converges faster than basic SGD |
| **Batch size** | Number of samples processed before updating weights | Affects training speed and generalization |
| **Epochs** | Number of full passes through the training data | More epochs = more learning (up to a point) |

### 2.3 Confusion Matrix
A confusion matrix is an N x N table (10x10 for MNIST) where:
- **Rows** represent the actual digit
- **Columns** represent the predicted digit
- **Diagonal** values are correct predictions
- **Off-diagonal** values are misclassifications

This helps us see which digits the model confuses (e.g., 4 vs 9, 3 vs 8).

---

## 3. Experiments

We will run **4 experiments** with increasing complexity to show how each change improves (or affects) performance:

### Experiment 1: Simple Baseline
- **Architecture:** Input(784) -> Dense(128, relu) -> Output(10, softmax)
- **Purpose:** Establish a baseline with the simplest possible network
- **Expected accuracy:** ~97%

### Experiment 2: Deeper Network
- **Architecture:** Input(784) -> Dense(256, relu) -> Dense(128, relu) -> Output(10, softmax)
- **Purpose:** Show the effect of adding more layers and neurons
- **Expected accuracy:** ~97.5-98%

### Experiment 3: Deeper + Dropout Regularization
- **Architecture:** Input(784) -> Dense(256, relu) -> Dropout(0.3) -> Dense(128, relu) -> Dropout(0.2) -> Output(10, softmax)
- **Purpose:** Demonstrate how dropout prevents overfitting
- **Expected accuracy:** ~98%+ with better generalization

### Experiment 4: Tuned Network
- **Architecture:** Input(784) -> Dense(512, relu) -> Dropout(0.3) -> Dense(256, relu) -> Dropout(0.2) -> Dense(128, relu) -> Output(10, softmax)
- **Purpose:** Show the effect of a wider and deeper network
- **Expected accuracy:** ~98.5%+

---

## 4. Evaluation Metrics

For each experiment, we will compute and save:

### 4.1 Metrics
| Metric | Description |
|--------|-------------|
| **Accuracy** | Percentage of correctly classified digits |
| **Precision** | Of all digits predicted as X, how many were actually X? |
| **Recall** | Of all actual X digits, how many were correctly predicted? |
| **F1-Score** | Harmonic mean of precision and recall |

### 4.2 Visualizations (saved to `results/` directory)
1. **Confusion matrix heatmap** for each experiment
2. **Training/validation accuracy curves** for each experiment
3. **Training/validation loss curves** for each experiment
4. **Comparison bar chart** showing accuracy across all experiments
5. **Sample predictions** showing correct and incorrect classifications
6. **Sample MNIST images** showing what the data looks like

---

## 5. Deliverables

| Deliverable | Location | Description |
|-------------|----------|-------------|
| Jupyter Notebook | `mnist_classification.ipynb` | Main notebook with all code, explanations, and experiments |
| Results directory | `results/` | All saved plots and metrics |
| Confusion matrices | `results/confusion_matrix_exp{N}.png` | One per experiment |
| Training curves | `results/training_curves_exp{N}.png` | Accuracy and loss plots |
| Comparison chart | `results/experiment_comparison.png` | Side-by-side accuracy comparison |
| Classification report | `results/classification_report_exp{N}.txt` | Detailed per-class metrics |
| README | `README.md` | Project summary with embedded visualizations |

---

## 6. Technical Stack

| Component | Tool | Why |
|-----------|------|-----|
| Runtime | `uv` virtual environment | Per project requirements |
| Framework | TensorFlow/Keras | Assignment requirement |
| Visualization | Matplotlib, Seaborn | Confusion matrices and training curves |
| Metrics | scikit-learn | Classification report, confusion matrix computation |
| Format | Jupyter Notebook | Assignment requirement |
| Data | `keras.datasets.mnist` | Built-in, no manual download needed |

---

## 7. Notebook Structure

The notebook will be divided into these clearly labeled sections:

1. **Introduction** - What we're building and why
2. **Setup & Imports** - All dependencies
3. **Data Loading & Exploration** - Load MNIST, visualize samples
4. **Data Preprocessing** - Normalization, flattening, one-hot encoding (with explanations)
5. **Experiment 1: Simple Baseline** - Build, train, evaluate
6. **Experiment 2: Deeper Network** - Build, train, evaluate
7. **Experiment 3: Dropout Regularization** - Build, train, evaluate
8. **Experiment 4: Tuned Network** - Build, train, evaluate
9. **Results Comparison** - Compare all experiments side by side
10. **Conclusions** - What we learned

Each section will have:
- A markdown cell explaining the concept
- Code cells with inline comments
- Output cells showing results
