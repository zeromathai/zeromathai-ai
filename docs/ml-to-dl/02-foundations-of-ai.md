# 02. Foundations of AI and Deep Learning

## 📌 Lecture Overview

This lecture provides an integrated overview of the basic concepts, relationships, historical development, and theoretical foundations of Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL).

Instead of treating these topics as isolated ideas, this section organizes them into a single flow:

**concept → context → meaning**

This helps explain how AI evolved from philosophical questions about intelligent machines to modern deep learning systems that learn from data and build hierarchical representations.

---

## 📖 Full Article

- English  
  https://zeromathai.com/en/dl-foundations-of-ai-en/

- Korean  
  https://zeromathai.com/dl-foundations-of-ai/

👉 Other languages are available on the website

---

## 📚 Table of Contents

- Concept of Artificial Intelligence (AI)  
- Relationship Between Deep Learning and AI  
- Three Waves of Deep Learning  
- Definition of Machine Learning  
- Machine Learning Tasks and Evaluation  
- Model Complexity and Generalization  
- Linear Models  
- Fundamentals of Neural Networks  
- Theoretical Foundations of Deep Learning  

---

## 1. Concept of Artificial Intelligence (AI)

Artificial Intelligence aims to build systems that can perform intelligent behaviors such as:

- thinking  
- learning  
- reasoning  
- problem solving  

AI was formally established around the Dartmouth Conference in 1956, but its roots go back to the philosophical question:

👉 **Can machines think?**

Early AI emphasized **Symbolic AI**, where human-designed rules and symbolic structures played a central role.

Modern AI goes further. Instead of simply automating fixed routines, it aims to build systems that:

- set goals  
- learn from data  
- improve autonomously  
- adapt to uncertainty  

---

## 2. Relationship Between Deep Learning and AI

AI is the broadest concept.

Inside AI, **Machine Learning (ML)** focuses on learning patterns from data.

Inside ML, **Deep Learning (DL)** uses multi-layer neural networks to learn hierarchical feature representations automatically.

👉 In simple form:

- **AI** = What intelligent behavior do we want?  
- **ML** = How can a system learn from data?  
- **DL** = How can layered neural networks learn rich representations?  

This hierarchical relationship is one of the most important conceptual maps in modern AI.

---

## 3. Three Waves of Deep Learning

The evolution of deep learning is often described through three major waves.

### ① First wave: Perceptron era (1950s–60s)
- introduced the artificial neuron  
- established the idea of machine learning with simple neural units  
- faced major limitations such as the XOR problem  

### ② Second wave: Backpropagation era (1980s–90s)
- backpropagation enabled practical training of multilayer networks  
- revived interest in connectionist neural networks  

### ③ Third wave: Modern deep learning era (2010s–present)
- GPUs, big data, and deep learning frameworks enabled large-scale models  
- deep learning became practically dominant in vision, speech, and language  

👉 These three waves can be summarized as:

**conceptual proposal → learning algorithm → large-scale deployment**

---

## 4. Definition of Machine Learning

Machine Learning is the idea that a system improves its performance through experience rather than explicit rule programming.

A classic definition comes from Tom Mitchell:

A program learns from experience **E** with respect to tasks **T** and performance measure **P** if its performance on **T**, measured by **P**, improves with experience **E**.

### Core idea
- more experience  
- better performance  
- measurable improvement  

This is what we call learning in Machine Learning.

---

## 5. Machine Learning Tasks and Evaluation

Machine Learning tasks vary depending on what kind of output or structure we want to learn.

### Representative tasks
- **Classification** — assign an input to a category  
- **Regression** — predict a continuous value  
- **Clustering** — discover groups in unlabeled data  
- **Dimensionality Reduction** — compress data into more compact structure  
- **Reinforcement Learning** — learn policies through reward and interaction  

### Evaluation
Evaluation depends on the task.

- classification → accuracy, precision, recall, F1-score  
- regression → MSE, MAE, R²  
- model validation → train / validation / test split, cross-validation  

The key point is that learning and evaluation are always connected.

---

## 6. Model Complexity and Generalization

A model must be expressive enough to capture real patterns, but not so complex that it memorizes the training data.

### Key concepts
- **Capacity** — how rich a function family the model can represent  
- **Generalization** — how well the model performs on unseen data  
- **Overfitting** — fits training data too closely, fails on new data  
- **Underfitting** — too simple to capture the true structure  

### Related theoretical ideas
- bias–variance tradeoff  
- Occam’s Razor  
- VC dimension  
- regularization  

The practical goal is to balance complexity and generalization.

---

## 7. Linear Models

Linear models are among the most fundamental tools in Machine Learning.

They compute outputs as weighted sums of input features and are used for:

- regression  
- classification  
- probabilistic prediction  

### Representative examples
- Linear Regression  
- Logistic Regression  
- Perceptron  

### Why they matter
- fast to train  
- easy to interpret  
- mathematically transparent  
- strong baseline models  

At the same time, their expressive power is limited when the true data structure is highly nonlinear.

---

## 8. Fundamentals of Neural Networks

Neural networks extend linear models by adding nonlinear activation functions and stacking multiple layers.

### Basic building blocks
- weighted sum  
- bias  
- activation function  
- hidden layers  
- output layer  

A network transforms input into output through layered computation, gradually learning more abstract internal representations.

### Why this matters
This layered representation learning is the central idea behind deep learning.

---

## 9. Theoretical Foundations of Deep Learning

The theoretical foundations of deep learning draw from several major ideas.

### Information theory
- entropy  
- cross-entropy  
- KL divergence  

These ideas help explain uncertainty, learning objectives, and loss functions.

### Estimation theory
- maximum likelihood estimation  
- probabilistic interpretation of training  

This explains why minimizing losses such as cross-entropy corresponds to fitting model parameters to observed data.

### Manifold assumption
Although high-dimensional data look complex, many real-world data distributions lie near lower-dimensional manifolds.

Deep networks learn internal representations that make this hidden structure easier to model.

### Optimization and regularization
Deep learning succeeds not only because of expressive models, but also because of:

- loss design  
- optimization algorithms  
- regularization strategies  
- large-scale training data  

---

## 🎯 Key Insight

This lecture builds the conceptual bridge from broad AI ideas to modern deep learning.

It shows that:

- AI defines the overall goal of intelligent behavior  
- Machine Learning provides the data-driven learning framework  
- Deep Learning extends that framework through layered neural representations  

👉 In this sense, deep learning is not separate from AI — it is one of the most powerful modern realizations of it.

---

## 📚 Related Advanced Topics

- [Concept of Artificial Intelligence](https://zeromathai.com/en/concept-of-ai-en/)
- [Relationship Between Deep Learning and AI](https://zeromathai.com/en/relationship-dl-ai-en/)
- [Three Waves in the Evolution of Deep Learning](https://zeromathai.com/en/three-waves-en/)
- [Definition of Machine Learning](https://zeromathai.com/en/definition-of-machine-learning-en/)
- [Machine Learning Tasks and Evaluation](https://zeromathai.com/en/machine-learning-tasks-and-evaluation-en/)
- [Model Complexity and Generalization](https://zeromathai.com/en/model-complexity-and-generalization-en/)
- [Linear Models](https://zeromathai.com/en/linear-models-en/)
- [Fundamentals of Neural Networks](https://zeromathai.com/en/fundamentals-of-neural-networks-en/)
- [Theoretical Foundations of Deep Learning](https://zeromathai.com/en/theoretical-foundations-of-dl-en/)

---

## ⭐ Note

This material is organized and reconstructed based on lectures by **Professor Sungroh Yoon (Seoul National University)**.

---

## 🔗 Navigation

← Previous: [Traditional Machine Learning](01-traditional-ml-overview.md)  
→ Next: [Optimization](03-optimization-regularization-integrated-lecture.md)
