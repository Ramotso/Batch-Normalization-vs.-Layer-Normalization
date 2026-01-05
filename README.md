# Batch Normalization vs Layer Normalization on MNIST

This repository contains a controlled PyTorch experiment comparing **Batch Normalization** and **Layer Normalization** in a simple feedforward neural network trained on the MNIST dataset. The implementation is intended to illustrate practical differences between normalization techniques under identical architectural and optimization conditions.

## Background and Motivation

Normalization techniques are widely used to stabilize and accelerate neural network training. While **Batch Normalization (BN)** normalizes activations across the batch dimension, **Layer Normalization (LN)** normalizes across feature dimensions within each individual sample.

This work is directly motivated by the original **Layer Normalization** paper:

> **Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016).**
> *Layer Normalization.*
> University of Toronto and Google Inc.

The paper introduces Layer Normalization as an alternative to Batch Normalization, particularly suited for scenarios where batch statistics are unstable or unavailable (e.g., recurrent networks, small batch sizes, or online learning).

This repository demonstrates those ideas in a concrete supervised learning setting using MNIST.

## Experiment Overview

The experiment trains two models that differ **only** in the normalization strategy applied after the first linear layer:

* **Batch Normalization (`BatchNorm1d`)**
* **Layer Normalization (`LayerNorm`)**

All other factors are held constant:

* Architecture
* Dataset
* Optimizer
* Learning rate
* Number of epochs
* Random seed

This allows for a clean comparison of training dynamics.

## Model Architecture

Each classifier follows the structure:

```
Input (784)
 → Linear (784 → 256)
 → Normalization (BatchNorm or LayerNorm)
 → ReLU
 → Linear (256 → 10)
```

Key design choices:

* Bias is disabled in the first linear layer when using Batch Normalization, as the normalization step cancels out the bias term.
* Bias is retained when using Layer Normalization.

## Code Structure

### `NormClassifier`

A configurable neural network that switches between:

* Batch Normalization
* Layer Normalization
* No normalization (identity)

based on the `norm_type` argument.

### `run_experiment(norm_type)`

* Trains the model using stochastic gradient descent
* Records average training loss per epoch
* Outputs loss curves for comparison

### Visualization

Training loss is plotted over epochs to compare convergence behavior between normalization methods.

## Dependencies

* Python 3.x
* PyTorch
* torchvision
* matplotlib

Install dependencies with:

```bash
pip install torch torchvision matplotlib
```

## Running the Experiment

Simply run the script:

```bash
python main.py
```

The script will:

1. Download MNIST (if not already present)
2. Train a model with Batch Normalization
3. Train a model with Layer Normalization
4. Plot the training loss curves for both methods

## Expected Observations

Consistent with the findings of Ba et al. (2016):

* Batch Normalization often converges faster with sufficiently large batch sizes
* Layer Normalization provides more stable behavior independent of batch size
* Training dynamics differ even when final performance is similar

The experiment is intentionally minimal to emphasize these effects.

## Reference

Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton.
**“Layer Normalization.”**
University of Toronto; Google Inc., 2016.
