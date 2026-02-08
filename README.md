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

## LLM Q&A

### Q: Why did you choose the ReLU activation function?

**ReLU (Rectified Linear Unit)** computes `max(0, x)` — it outputs the input directly if positive, or zero otherwise. We chose it for the hidden layers for several reasons:

1. **Avoids the vanishing gradient problem.** Older activations like sigmoid and tanh squash outputs into small ranges (0-1 or -1 to 1). During backpropagation, gradients get multiplied through many layers, and these small values cause gradients to shrink to near zero — making deep networks almost impossible to train. ReLU's gradient is either 0 or 1, so gradients flow through without shrinking.

2. **Computationally fast.** ReLU is just a comparison (`max(0, x)`), while sigmoid and tanh require exponential calculations (`e^x`). This makes training noticeably faster, especially over millions of parameters and many epochs.

3. **Produces sparse activations.** ReLU zeros out all negative values, meaning many neurons output exactly 0 at any given time. This sparsity makes the network more efficient and can act as a mild form of regularization.

4. **It's the proven default.** ReLU has been the standard activation for hidden layers since ~2012 (AlexNet). For a straightforward classification task like MNIST with Dense layers, there's no reason to use anything more exotic (like LeakyReLU or GELU), which are typically used for more complex architectures.

**Note:** ReLU is only used for *hidden* layers. The *output* layer uses **softmax**, which converts raw scores into probabilities that sum to 1 — required for multi-class classification.

---

### Q: How and why did you choose the number of neurons for each layer in each experiment?

The neuron counts follow a **funnel (narrowing) pattern**: wide at the input, progressively narrower toward the output. This is a standard design principle — early layers capture many low-level features, and later layers compress them into higher-level abstractions.

**The constraints that guided our choices:**
- **Input is fixed at 784** (28x28 pixels flattened)
- **Output is fixed at 10** (one neuron per digit class)
- Hidden layers must bridge the gap between 784 and 10

**Experiment 1 — Dense(128):**
128 is a common starting point for simple baselines. It's a power of 2 (which aligns well with GPU/CPU memory), and it's small enough to be fast while large enough to learn the patterns in MNIST. Going much lower (e.g., 32) would underfit — the layer wouldn't have enough capacity to distinguish 10 digit classes from 784-dimensional input.

**Experiment 2 — Dense(256) → Dense(128):**
We doubled the first layer to 256 to give the network more capacity to learn features, then funnel down to 128. The idea is: the first layer (256) learns many basic patterns (edges, curves), and the second layer (128) combines them into digit-level features. We kept powers of 2 for computational efficiency.

**Experiment 3 — Dense(256) → Dense(128) (+ Dropout):**
Same neuron counts as Exp 2, but with dropout. The point of this experiment is to isolate the effect of dropout — by keeping the architecture identical, any accuracy difference is purely from regularization, not from changing the network's capacity.

**Experiment 4 — Dense(512) → Dense(256) → Dense(128):**
We scaled up to 512 in the first layer (more features to learn from), then funnel through 256 → 128. This creates a 3-level feature hierarchy:
- Layer 1 (512): low-level features (strokes, edges)
- Layer 2 (256): mid-level features (loops, curves, angles)
- Layer 3 (128): high-level features (digit-like shapes)

**Why powers of 2 (128, 256, 512)?**
This is a practical convention, not a strict rule. Powers of 2 align with how memory is allocated on modern hardware (32/64-bit architectures, GPU warp sizes), which can result in slightly faster computation. But 100, 200, 500 would work nearly as well.

**Why not go bigger (e.g., 1024, 2048)?**
MNIST is a simple dataset — 28x28 grayscale images with centered digits. Overly large networks would memorize the training data (overfit) rather than learning generalizable patterns, and would train slower for no meaningful accuracy gain.

---

### Q: Explain the effectiveness of Dropout and how did you choose the percentages?

**What is Dropout?**
Dropout is a regularization technique that randomly "turns off" (sets to zero) a fraction of neurons during each training step. If a layer has `Dropout(0.3)`, then in every training batch, each neuron in that layer has a 30% chance of being temporarily disabled.

**Why is it effective?**

1. **Prevents co-adaptation.** Without dropout, neurons can become overly dependent on each other — neuron A always relies on neuron B's output. Dropout forces each neuron to learn useful features *independently*, because it can't count on any particular neighbor being active. This makes the learned features more robust and generalizable.

2. **Acts like an ensemble.** Each training step uses a different random subset of neurons, which is effectively a different "sub-network." Over many batches, the model trains thousands of overlapping sub-networks. At inference time (when dropout is turned off), the full network acts like an average of all these sub-networks — similar to how ensemble methods (like Random Forest) combine many weak models into a strong one.

3. **Reduces overfitting.** Overfitting happens when a model memorizes training data instead of learning general patterns. Dropout prevents this by making memorization harder — the network can't rely on specific neuron combinations to memorize specific examples.

**Evidence in our experiments:**
- Exp 2 (no dropout): **96.86%** — the deeper network actually *decreased* accuracy vs baseline, suggesting overfitting
- Exp 3 (with dropout, same architecture): **98.11%** — a **1.25% jump** just from adding dropout
- This demonstrates dropout's power: same network capacity, much better generalization

**How we chose the percentages (0.3 and 0.2):**

We used `Dropout(0.3)` after the first hidden layer and `Dropout(0.2)` after the second. This follows a **decreasing dropout rate** pattern toward the output:

1. **Higher dropout (0.3) on wider layers.** The first hidden layer (256 or 512 neurons) has the most parameters and the highest risk of overfitting. A 30% dropout rate provides stronger regularization where it's needed most.

2. **Lower dropout (0.2) on narrower layers.** The second hidden layer (128 neurons) has fewer neurons — too aggressive a dropout here would destroy too much information before the output layer, hurting the model's ability to make accurate predictions.

3. **No dropout on the output layer.** We need all 10 output neurons active to produce valid probability distributions. Dropping output neurons would mean some digit classes get zero probability, making predictions unreliable.

4. **No dropout on the input layer.** While some architectures do use input dropout (typically very low, ~0.1), for MNIST the input is already sparse (most pixels are black/zero), so additional dropout on the input would discard too much of the limited signal.

**Why 0.3/0.2 specifically and not other values?**
- These are well-established defaults from the original dropout paper (Srivastava et al., 2014), which found that dropout rates between 0.2 and 0.5 work well for most tasks
- For MNIST (a relatively easy dataset), moderate rates (0.2-0.3) are sufficient — higher rates (0.5+) would over-regularize and slow convergence unnecessarily
- The 0.3 → 0.2 decreasing pattern is a common heuristic: regularize more aggressively where there are more parameters to overfit

---

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
