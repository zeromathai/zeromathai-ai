
# 04. Multilayer Perceptron (MLP)

## 📌 Lecture Overview

Multilayer Perceptron (MLP) is the most fundamental neural network architecture that learns complex input–output relationships by stacking multiple layers.

By introducing hidden layers and nonlinear activation functions, an MLP can model patterns that linear models cannot capture.

This lecture connects the full pipeline:

- structure → representation → probability → learning → optimization

so you can understand neural networks as one coherent system.

---

## 📖 Full Article

- English  
  https://zeromathai.com/en/dl-mlp-representation-and-backprop-en/

- Korean  
  https://zeromathai.com/dl-mlp-representation-and-backprop/

👉 Other languages are available on the website

---

## 📚 Table of Contents

- MLP structure and information flow  
- Nonlinear representation learning and hidden layers  
- Output layer and probabilistic interpretation  
- Learning signal and backpropagation  
- Optimization and architecture interaction  

---

## 1. MLP Structure and Information Flow

An MLP consists of:

- Input layer  
- Hidden layers  
- Output layer  

Each layer performs:

- linear transformation  
- nonlinear activation  

The core computation is:

\[
h^{(l)} = \sigma(W^{(l)} h^{(l-1)} + b^{(l)})
\]

👉 Meaning:

- transform previous representation  
- reshape it with nonlinearity  
- pass it forward  

👉 Intuition:

MLP builds **increasingly abstract features layer by layer**.

---

## 2. Nonlinear Representation Learning

Without nonlinearity, multiple layers collapse into a single linear function.

Hidden layers solve this by:

- transforming data into new feature spaces  
- making complex patterns separable  

👉 Key idea:

MLP performs **representation learning**

- not manually designed features  
- learned directly from data  

---

## 3. Output Layer and Probabilistic Interpretation

In classification, outputs are interpreted as probabilities.

The most common function is **softmax**:

P(y=i \mid x) = e^{z_i} / \sum_j e^{z_j}

👉 What softmax does:

- converts logits → probabilities  
- ensures outputs sum to 1  

👉 Example:

logits: [2.0, 1.0, 0.1]  
→ probabilities: [0.66, 0.24, 0.10]

---

## 4. Learning Signal and Backpropagation

MLP learns by minimizing loss.

Backpropagation computes how each parameter affects the loss.

Core idea:

\[
\frac{\partial L}{\partial W^{(l)}} =
\frac{\partial L}{\partial h^{(l)}} \cdot
\frac{\partial h^{(l)}}{\partial W^{(l)}}
\]

👉 Meaning:

- error flows backward  
- each layer receives responsibility signal  
- parameters are updated accordingly  

👉 Intuition:

Backprop answers:

> "Which connections caused the error?"

---

## 5. Optimization and Architecture Interaction

Training depends on both:

- optimization (how we update)  
- architecture (how model is structured)  

### Key trade-offs

- Deep networks  
  → more expressive  
  → harder to train (vanishing gradients)

- Wide networks  
  → more capacity  
  → higher overfitting risk  

👉 Important insight:

Architecture and optimization form a **coupled system**

---

## 🎯 Key Insight

MLP is the foundation of modern deep learning.

It shows how:

- linear transformation + nonlinearity  
- stacked into layers  
- trained with backprop  

can produce powerful models.

👉 Everything else (CNN, RNN, Transformer)  
is an extension of this idea.

---

## 📚 Related Advanced Topics

- [MLP Intuition and Core Components](https://zeromathai.com/en/mlp-intuition-components-en/)
- [Output Layer and Probabilistic Interpretation](https://zeromathai.com/en/output-layer-probabilistic-interpretation-en/)
- [Backpropagation Fundamentals](https://zeromathai.com/en/training-signals-back-fundamentals-en/)
- [Optimization and Architecture Design](https://zeromathai.com/en/optimization-architecture-en/)

---

## ⭐ Note

This material is organized and reconstructed based on lectures by **Professor Sungroh Yoon (Seoul National University)**.

---

## 🔗 Navigation

← Previous: [Optimization](03-optimization-regularization-integrated-lecture.md)  
→ Next: [CNNs](05-convolutional-neural-networks-cnn.md)
