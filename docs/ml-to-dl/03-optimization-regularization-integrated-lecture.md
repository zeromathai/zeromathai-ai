
# 03. Optimization and Regularization

## 📌 Lecture Overview

This lecture explains how **optimization** and **regularization** shape the training process in deep learning, and why these two ideas must be understood together.

Optimization makes a model fit the data better by updating its parameters.  
Regularization prevents the model from simply memorizing the training set and helps it generalize to new data.

Together, they answer two central questions of learning:

- How do we make the model learn effectively?
- How do we keep the learned model from overfitting?

---

## 📖 Full Article

- English  
  https://zeromathai.com/en/dl-optimization-regularization-integrated-lecture-en/

- Korean  
  https://zeromathai.com/dl-optimization-regularization-integrated-lecture/

👉 Other languages are available on the website

---

## 📚 Table of Contents

- Why optimization is necessary  
- Gradient descent and moving averages  
- Learning rate and learning rate scheduling  
- Why we need regularization  
- Norm penalties (L1 / L2 / Elastic Net)  
- Early stopping with validation data  
- Dropout  

---

## 1. Why Optimization Is Necessary

Optimization is the process of adjusting model parameters so that the loss function becomes as small as possible.

A lower loss means:

- predictions are closer to targets  
- the model fits the data better  
- learning is progressing in the right direction  

👉 In simple terms:  
optimization makes the model **less wrong, step by step**.

In deep learning, this is usually done through gradient-based training.  
The gradient tells us how the loss changes when each parameter changes, and gives the direction for reducing the loss.

---

## 2. Gradient Descent and Moving Averages

The most basic optimization idea is **gradient descent**.

It repeatedly updates parameters in the direction opposite to the gradient.

👉 Intuition:  
it is like **walking downhill toward a lower point**, guided by the local slope.

In practice, deep learning usually uses **mini-batch gradients**, not full-dataset gradients.  
These updates are efficient, but also noisy.

To reduce instability, modern optimizers use moving averages of gradients.

### Main examples

- **Momentum**  
  smooths the update direction and reduces zig-zagging  

- **RMSProp**  
  adapts step sizes per parameter using recent gradient magnitudes  

- **Adam**  
  combines Momentum and RMSProp, and is often the default optimizer in modern deep learning  

These methods improve stability and often speed up convergence.

---

## 3. Learning Rate and Learning Rate Scheduling

The **learning rate** controls how large each parameter update is.

### If the learning rate is too large
- training may oscillate  
- loss may diverge  
- optimization can become unstable  

### If the learning rate is too small
- training becomes very slow  
- the model may get stuck in weak local regions  

👉 In simple terms:  
the learning rate is the **step-size knob** of training.

### Learning rate scheduling

A common practice is:

- start with a larger learning rate for fast early progress  
- reduce it later for careful fine-tuning  

This often improves both training speed and final accuracy.

---

## 4. Why We Need Regularization

Regularization helps prevent **overfitting**.

Overfitting happens when the model fits the training data extremely well but performs poorly on unseen data.

👉 In simple terms:  
regularization is the mechanism that encourages the model to **learn the rule**, not **memorize the answer sheet**.

Regularization affects:

- how large parameters can grow  
- how long training continues  
- how strongly the network depends on specific pathways or features  

Together with train / validation / test splits, regularization is essential for controlling generalization error.

---

## 5. Norm Penalties (L1 / L2 / Elastic Net)

Norm penalties are regularization methods that constrain the size of model weights.

### L1 regularization (Lasso)
- penalizes the absolute values of weights  
- encourages sparsity  
- can drive many weights exactly to zero  

### L2 regularization (Ridge / weight decay)
- penalizes squared weights  
- discourages large parameter values  
- promotes smoother and more stable solutions  

### Elastic Net
- combines L1 and L2  
- balances sparsity and stability  
- useful when features are correlated  

👉 Main idea:  
keep parameters within a reasonable range so the model does not become overly sensitive or extreme.

---

## 6. Early Stopping with Validation Data

As training continues:

- training loss often keeps decreasing  
- validation loss may eventually stop improving  
- after that point, validation performance may get worse  

This is a strong sign of overfitting.

### Early stopping
Early stopping monitors validation loss and stops training at the point where validation performance is best.

Why it works:
- prevents over-specialization to the training set  
- improves generalization  
- does not require changing the model architecture  

It is one of the simplest and most effective regularization techniques in practice.

---

## 7. Dropout

Dropout randomly removes a subset of neurons during training.

This forces the network to avoid relying too heavily on any single path or feature combination.

👉 In simple terms:  
dropout trains the network to solve the problem in **many different ways**, so it becomes more robust.

### Why dropout helps
- reduces co-adaptation between neurons  
- acts as an implicit regularizer  
- often improves test performance  

At training time, some units are dropped.  
At inference time, the full network is used.

---

## 🎯 Key Insight

Optimization and regularization are not separate topics.

They are two sides of the same training problem:

- **Optimization** improves fit to the data  
- **Regularization** improves generalization beyond the data  

A model that only optimizes may memorize.  
A model that is too strongly regularized may underfit.

👉 The practical goal is to train a model that learns efficiently **and** generalizes reliably.

---

## 📚 Related Advanced Topics

- [Optimization in Machine Learning](https://zeromathai.com/en/optimization-in-machine-learning-en/)
- [Adaptive Optimization and Learning Rate Scheduling](https://zeromathai.com/en/adaptive-optimization-en/)
- [Regularization in Machine Learning and Deep Learning](https://zeromathai.com/en/regularization-generalization-en/)

---

## ⭐ Note

This material is organized and reconstructed based on lectures by **Professor Sungroh Yoon (Seoul National University)**.

---

## 🔗 Navigation

← Previous: [Foundations of AI and Deep Learning](02-foundations-of-ai.md)  
→ Next: [MLP](04-mlp-representation-and-backprop.md)
