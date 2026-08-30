# GAN — Learning Data Distributions and Deep Generative Models

## 📌 Lecture Overview

A GAN is easier to understand when it is placed inside the broader problem of generative modeling. Real images, text, and speech live in very high-dimensional spaces, but meaningful data occupies only a structured portion of those spaces. A generative model aims to represent that structure as a probability distribution, learn it from observed samples, and, when necessary, infer the hidden variables or representations that may have produced those observations.

This perspective connects GANs to three recurring challenges: **Representation**, **Learning**, and **Inference**. Increasing model flexibility can make probability evaluation and posterior inference difficult, creating a tension between expressive power and computational tractability. Autoregressive models, VAEs, flow-based models, GANs, and diffusion models can therefore be understood as different answers to the same question: how can we model complex data distributions while keeping learning and generation computationally feasible?

---

## 📖 Full Article - English

[https://zeromathai.com/en/gan-course-en/](https://zeromathai.com/en/gan-course-en/)

* Korean
  [https://zeromathai.com/gan-course/](https://zeromathai.com/gan-course/)

👉 Other languages are available on the website

---

## 📚 Table of Contents

* [1. From Discrimination to Generative Modeling](#1-from-discrimination-to-generative-modeling)
* [2. Why Learn a Data Distribution?](#2-why-learn-a-data-distribution)
* [3. Representation, Learning, and Inference](#3-representation-learning-and-inference)
* [4. Posterior Inference and Approximation](#4-posterior-inference-and-approximation)
* [5. Learning a Model Distribution](#5-learning-a-model-distribution)
* [6. Tractability, Flexibility, and Normalization](#6-tractability-flexibility-and-normalization)
* [7. Deep Generative Modeling Strategies](#7-deep-generative-modeling-strategies)
* [8. Tradeoffs Across Generative Models](#8-tradeoffs-across-generative-models)

---

## 1. From Discrimination to Generative Modeling

In supervised classification, the input $x$ and its label $y$ are observed together. Both discriminative and generative approaches ultimately need to determine the appropriate class for a new input, but they learn different probabilistic objects on the way to that decision.

A **Discriminative Model** directly models the conditional probability $p(y\mid x)$. Classification can therefore be written as

```math
f(x)=\underset{y}{\arg\max}\;p(y\mid x)
```

The model does not need to explain how each class is distributed throughout the entire input space. From a classification perspective, it can focus on the relationships needed to separate classes and form a useful **decision boundary**. Support Vector Machine is a representative example of this discriminative viewpoint.

A **Generative Model**, in the class-conditional supervised setting, approaches the same problem from the opposite direction. It first models $p(x\mid y)$: the distribution of inputs associated with each class. It also uses the class prior $p(y)$. Bayes Rule then connects this generative description to the posterior needed for classification:

```math
p(y\mid x)=\frac{p(x\mid y)p(y)}{p(x)}
```

For a fixed input $x$, the denominator $p(x)$ is identical for every candidate class. The classification rule can therefore be written as

```math
f(x)=\underset{y}{\arg\max}\;p(x\mid y)p(y)
```

The two approaches may choose the same class while reaching that result through different probabilistic paths. A discriminative model asks directly, **"Given this input, which class is most likely?"** A generative model first asks, **"If this were a particular class, how likely would this input be?"**

In this class-conditional comparison:

* A discriminative model directly learns $p(y\mid x)$ and emphasizes relationships needed for classification and decision boundaries.
* A generative model learns $p(x\mid y)$, combines it with $p(y)$, and represents the probability structure of the data within each class.
* Support Vector Machine represents the discriminative viewpoint, while Gaussian Mixture Model and Bayes nets are examples of generative modeling.

This use of the term *generative* should not yet be confused with unrestricted unsupervised modeling of $p(x)$. Here, the distinction is specifically about learning class-conditional distributions $p(x\mid y)$ rather than directly learning the classification posterior.

Once labels disappear, however, the generative viewpoint naturally expands. There is no longer a class boundary to learn. The target becomes the data distribution itself:

```math
p(x)
```

**Density Estimation** asks how likely different values of $x$ are under this distribution, while **Sample Generation** asks how to draw new values from the learned distribution.

This is difficult because real data is high-dimensional but highly structured. Randomly choosing image pixels produces mostly meaningless noise, and most possible word sequences do not form natural language. From a **Data Manifold** perspective, meaningful observations occupy restricted, structured regions inside a much larger high-dimensional space.

Generative modeling therefore requires more than memorizing possible combinations. The model must capture the structured regions in which real data is concentrated and represent them probabilistically.

---

## 2. Why Learn a Data Distribution?

Generating new samples is only one consequence of learning a data distribution. Once a model captures where meaningful data lies and how probability is organized within that space, the learned structure can support several other tasks.

In **Reinforcement Learning**, a learned model can generate possible future states for simulation. In **Continual Learning**, **Generative Replay** can reproduce samples resembling earlier data rather than requiring every past observation to remain stored. For **Missing Data**, a learned distribution can support **Imputation** by estimating unobserved values from the observed ones. In **Semi-Supervised Learning**, the structure contained in unlabeled data can contribute to learning even when labels are available for only part of the dataset.

Distribution learning is also closely related to **Representation Learning**. Different observations may look different while sharing recurring high-level structure. If a generative model captures such structure through a **Latent Representation**, that representation can serve as a useful feature for other learning problems.

Sampling, missing-data modeling, semi-supervised learning, and representation learning are therefore not unrelated capabilities. They all become possible because the model has learned something about the probability structure underlying the observations.

The same reasoning leads to **Inverse Problems**. A forward problem maps a cause or input $x$ to an observed result $y$. An inverse problem starts from $y$ and asks which values of $x$ could plausibly have produced it. Super-resolution, denoising, image inpainting, image colorization, and CT image reconstruction can be viewed from this perspective.

A single observation may correspond to multiple possible underlying causes, so finding only one deterministic solution can be insufficient. Modeling a conditional distribution such as $p(x\mid y)$ provides a way to represent multiple plausible solutions.

This exposes the deeper difficulty of generative modeling. Data can be high-dimensional, several latent causes can explain the same observation, and some important variables may never be observed directly. A successful model must therefore do three things: represent a complex distribution, learn it from data, and infer hidden structure when required.

---

## 3. Representation, Learning, and Inference

The central problems of generative modeling can be organized around three axes.

**Representation** asks how to compactly describe a high-dimensional joint distribution produced by many random variables. For data such as images, explicitly representing every possible variable combination is infeasible, so the model needs a structure that captures dependencies efficiently.

**Learning** asks how to reduce the difference between the unknown **Data Distribution** and a trainable **Model Distribution**. A distribution-level discrepancy must ultimately become an objective that can be computed from observed samples and model parameters.

**Inference** asks how to recover a **Hidden Variable** or **Latent Structure** from an observed input $x$. If generation runs from hidden causes toward observations, inference runs in the reverse direction.

These problems are tightly coupled. Increasing **Model Expressivity** can make probability evaluation and learning harder. Introducing a compact latent variable $z$ can simplify the representation of observations, but it may create a difficult **Posterior Inference** problem when the model must infer $z$ from $x$.

The design question is therefore not simply how large or flexible a neural network should be. A useful generative model must retain enough expressive power while keeping learning and inference computationally manageable.

---

## 4. Posterior Inference and Approximation

Suppose a probabilistic model describes an observed variable $x$ together with an unobserved variable $z$ through the joint distribution $p(z,x)$. If the model specifies how $z$ can produce $x$, observing $x$ naturally creates the reverse question: which values of $z$ could have produced this observation?

The answer is the posterior distribution

```math
p(z\mid x)=\frac{p(z,x)}{p(x)}
```

**Posterior Inference** means computing this distribution or using it to calculate quantities such as expectations.

The **Expectation-Maximization (EM) Algorithm** illustrates this role clearly. Because the hidden variable is not observed directly, the E-step uses the posterior under the current parameters to compute the expected Complete-Data Log-Likelihood:

```math
Q(\theta,\theta^{old})
=
\sum_z
p(z\mid x;\theta^{old})
\log p(x,z;\theta)
```

For simple probabilistic models, such calculations may be exact. As the latent space becomes high-dimensional and dependencies become more complex, however, the required sums or integrals can become computationally difficult. The richer representation that made the model expressive can make its exact posterior intractable.

This motivates **Approximate Inference**.

Two representative strategies are **Markov Chain Monte Carlo (MCMC)** and **Variational Inference (VI)**.

MCMC constructs an ergodic Markov chain whose stationary distribution is the target posterior. Rather than deriving the posterior in a directly manageable form, it uses samples from that chain to approximate properties of the distribution. The method can approach the desired distribution with sufficient sampling, but the computational cost can become substantial for complex models and large datasets.

Variational Inference converts inference into an optimization problem. Instead of calculating the difficult posterior $p(z\mid x)$ exactly, it defines a tractable family of candidate distributions $\mathcal{Q}$ and searches for the member closest to the posterior:

```math
q^{*}(z)
=
\underset{q(z)\in\mathcal{Q}}{\arg\min}\;
D_{KL}\left[q(z)\parallel p(z\mid x)\right]
```

Restricting $\mathcal{Q}$ to factorized or otherwise tractable parametric distributions makes large-scale optimization possible, but it introduces approximation error whenever the chosen family cannot adequately represent the true posterior.

MCMC and VI therefore accept different costs. MCMC pays for sampling, while VI accepts approximation error in exchange for a computationally manageable optimization problem. This distinction becomes especially important in the VAE, where an encoder acts as a **Recognition Network** that approximates an otherwise difficult posterior.

---

## 5. Learning a Model Distribution

Inference starts from an observation and reasons about hidden structure. Learning moves in the other direction: it starts from observed samples and attempts to construct the distribution that could explain them.

The exact **Data Distribution** is unknown. What we have is a finite collection of samples drawn from it. We therefore define a parameterized model distribution $p_{model}(x;\theta)$ and adjust $\theta$ so that the observed data becomes likely under that model.

A well-trained model distribution should support related capabilities. Samples drawn from it should resemble real data, high-density regions should correspond to plausible observations when density evaluation is available, and recurring structure learned from the data may provide useful representations.

In likelihood-based modeling, the relation between the data and model distributions can be expressed through KL Divergence:

```math
D_{KL}\left[p_{data}(x)\parallel p_{model}(x;\theta)\right]
=
-H(p_{data})
+
\mathbb{E}_{x\sim p_{data}}
\left[
-\log p_{model}(x;\theta)
\right]
```

The entropy term $H(p_{data})$ depends only on the true data distribution, not on the model parameters $\theta$. From the perspective of parameter learning, minimizing the KL Divergence therefore reduces to minimizing the expected **Negative Log-Likelihood (NLL)**.

In other words, assigning high likelihood to real observations can move the model distribution closer to the data distribution.

But knowing the objective does not solve the full problem. For high-dimensional observations whose variables interact in complicated ways, $p(x)$ itself may be extremely difficult to evaluate. The model therefore needs a representation that makes its learning objective computationally accessible.

This is why the next question is not merely **which objective should we optimize?** It is **how should we structure the distribution so that the objective can actually be computed?**

Conditional factorization, latent variables, and invertible transformations are different answers to that question.

---

## 6. Tractability, Flexibility, and Normalization

Generative modeling repeatedly encounters a **Tractability-Flexibility Tradeoff**.

A simple Gaussian distribution has high tractability: probability evaluation and sampling are straightforward. But one simple Gaussian may be too restrictive to represent a complex high-dimensional distribution such as real images or speech.

A highly flexible function can represent more complicated structure, but evaluating probabilities, learning parameters, or performing inference may then become difficult.

Consider using a deep neural network $f_\theta(x)$ to define an expressive probability model. Arbitrary network outputs cannot directly serve as probabilities: they may be negative, and they are not guaranteed to integrate to one.

Exponentiating the output makes it positive, after which it can be normalized:

```math
p_\theta(x)=\frac{e^{f_\theta(x)}}{Z_\theta}
```

where

```math
Z_\theta=\int e^{f_\theta(x)}dx
```

The **Normalizing Constant** $Z_\theta$ ensures that the resulting probability distribution integrates to one.

The problem is that this integral may itself be intractable. If a flexible deep network defines $f_\theta(x)$ over a high-dimensional space, calculating the contribution from every possible $x$ can be prohibitively difficult.

The model has gained flexibility but lost tractability.

This normalization problem is one reason deep generative models adopt very different architectural strategies. Some factor a complex joint distribution into tractable conditional distributions. Some transform a simple distribution through carefully constrained mappings. Others introduce latent variables and approximate the resulting posterior. GANs avoid explicit density evaluation, while score-based diffusion approaches can work with gradients of log density rather than directly evaluating a normalized density.

These models look different because they make different compromises around the same computational obstacle.

---

## 7. Deep Generative Modeling Strategies

A useful broad distinction is between **Likelihood-based** and **Likelihood-Free** approaches.

Within the classification used here, autoregressive models, VAEs, flow-based models, and diffusion models belong to the likelihood-based family, while GANs represent a likelihood-free approach. This classification does not mean that every likelihood-based model evaluates normalized probability density in exactly the same way. In particular, the score-based formulation of diffusion models learns gradients of log density rather than directly evaluating the normalized density itself.

The important question is therefore not the label attached to a model family, but how each design handles representation, tractability, learning, and generation.

### Conditional Factorization — Autoregressive Models

One solution is to avoid modeling a complicated joint distribution as a single unrestricted object. The probability chain rule decomposes it into conditional distributions:

```math
p(x)
=
\prod_{i=1}^{n}
p(x_i\mid x_1,\ldots,x_{i-1})
```

Instead of representing $p(x)$ all at once, the model imposes an ordering on its variables and predicts each variable conditioned on those that came before it.

GPT, RNN, PixelRNN, PixelCNN, WaveNet, NADE, and MADE are examples of this perspective.

The important design choice is structural: organizing the representation as a product of conditional distributions makes probability evaluation and learning tractable while still allowing a complex joint distribution to be represented.

### Latent Variables and Variational Approximation — VAE

A **Variational Autoencoder (VAE)** takes a different route. It assumes that a compact latent variable $z$ lies behind a more complicated observation $x$.

The generative process first draws $z$ from a prior $p(z)$ and then generates $x$ through a conditional distribution $p(x\mid z)$. Marginalizing over all possible latent values gives

```math
p(x)=\int p(z)p(x\mid z)dz
```

The encoder and decoder should not be understood merely as two symmetric neural networks.

The **Decoder** directly participates in the generative process by defining $p(x\mid z)$. The **Encoder** is not part of the generative equation above. Instead, it serves as a **Recognition Network** that approximates the difficult posterior $p(z\mid x)$.

The VAE uses this approximate posterior within Variational Inference and maximizes the **ELBO**. The architecture is therefore a direct response to the earlier inference problem: if the exact posterior is too difficult to compute, learn a tractable approximation to it.

### Invertible Transformations — Flow-Based Models

A **Flow-based Model** begins with a tractable base distribution and transforms it into a more complicated target distribution through a sequence of **Invertible Transformations**.

The requirement of invertibility is a structural restriction. In exchange for accepting that restriction, the model can track how probability density changes through the transformations and retain explicit density evaluation.

This makes direct NLL optimization possible.

Flow-based models therefore obtain flexibility while deliberately constraining the architecture enough to preserve tractable probability evaluation.

### Learning the Generation Process — GAN

A **Generative Adversarial Network** changes the problem more fundamentally.

Rather than trying to build an explicitly evaluable probability density, a GAN learns the generation process itself. The **Generator** takes an input such as noise and produces synthetic data. The **Discriminator** attempts to distinguish real samples from generated ones.

Their objectives oppose one another. The Generator tries to make generated samples difficult to distinguish from real data, while the Discriminator tries to improve that distinction. Training therefore takes the form of a **Minimax Problem** between two networks.

This shift is the central idea behind GANs.

The Generator does not need to calculate the exact value of $p(x)$ for a particular image, nor does it need to compute a normalizing constant across the entire input space. Instead, it learns to produce samples that are difficult to distinguish from data drawn from the real distribution.

A GAN is therefore an **Implicit Generative Model**. In the classification used here, it is a representative form of **Likelihood-Free Modeling**.

The original question

> How can we calculate a complex probability density?

has effectively been replaced by

> How can we learn a process that generates samples resembling those from the real data distribution?

That change in formulation is what separates GANs from explicit density-modeling approaches.

### Noise Corruption and Reverse Generation — Diffusion Models

A **Diffusion Model** chooses another strategy. Instead of attempting to generate a complicated observation in one step, it gradually corrupts real data with Gaussian noise.

Starting from $x_0$, the **Forward Diffusion Process** moves through $x_1,x_2,\ldots,x_T$. As noise accumulates, the original data structure gradually disappears. After sufficiently many steps, the final state approaches an **Isotropic Gaussian Distribution**.

Generation learns the reverse direction. The **Reverse Diffusion Process** starts from Gaussian noise and progressively restores data structure.

From the score-based viewpoint, the model can work with the **Score Function**, the gradient of log probability density:

```math
s(x)=\nabla_x\log p(x)
```

The score indicates the direction in which log probability density increases. A normalizing constant that does not depend on $x$ disappears under this gradient, so the model can work with distributional structure without directly computing that normalizing constant.

**SMLD** estimates scores across multiple noise scales and uses them for sampling. **DDPM** learns a probabilistic reverse process that progressively undoes the injected noise. Their detailed learning procedures differ, but both share the broad idea of moving real data toward a manageable noise distribution and learning how to reverse that process.

In continuous state spaces, the DDPM objective is also connected to the score at each noise scale.

---

## 8. Tradeoffs Across Generative Models

Deep generative models split into multiple families because no single design ideally satisfies every desirable property at once. Probability evaluation, sample quality, diversity, training stability, sampling speed, and model flexibility can pull the architecture in different directions.

The earlier Tractability-Flexibility Tradeoff therefore appears as concrete strengths and limitations:

* **VAE:** Connects latent-variable modeling, probabilistic generation, and representation learning through Variational Inference, but sample quality and dependence on a surrogate loss are representative limitations.
* **Flow-based Model:** Supports explicit probability-density evaluation and direct NLL training, but requires specialized architectures built from invertible transformations.
* **GAN:** Learns a generation process without explicit density evaluation, but adversarial training can be unstable, and generated samples can suffer from limited diversity.
* **Diffusion Model:** Can provide strong quality and diversity, but generation requires a sequence of reverse diffusion steps, making sampling computationally long.

The simplified quality-diversity-speed comparison in the source emphasizes representative rather than absolute tradeoffs. VAE and flow families are characterized as favorable in diversity and speed but constrained in quality; GANs as favorable in quality and speed but constrained in diversity; and diffusion models as favorable in quality and diversity but constrained in speed.

These are not universal rankings across every dataset and implementation. They are a conceptual comparison of the consequences of each modeling strategy.

Understanding GANs therefore requires more than memorizing the roles of the Generator and Discriminator.

The starting point is the complex probability distribution formed by real data in high-dimensional space. Modeling that distribution requires an appropriate representation, a learning procedure that brings the model distribution toward the data distribution, and, when latent variables are involved, a method for inference. Increasing expressivity can then create computational obstacles such as intractable posteriors or normalizing constants.

Autoregressive models address the problem through conditional factorization. VAEs introduce latent variables and approximate difficult posteriors with Variational Inference. Flow-based models impose invertibility so that density evaluation remains tractable. GANs leave explicit density modeling behind and learn the generation process through adversarial competition. Diffusion models progressively corrupt data into noise and learn the reverse process.

Different architectures emerge from the same underlying question:

**How can a model represent a complex real-world data distribution while remaining trainable and capable of generating new samples?**

---

## 🎯 Key Insight

GAN is not best understood as an isolated two-network architecture. It is one answer to the broader computational problem of deep generative modeling.

Generative models must balance expressive representations of complex data with tractable learning, inference, probability evaluation, and generation. Explicit likelihood models preserve probability structure by introducing factorizations, latent-variable approximations, or architectural constraints. GANs instead avoid explicit density evaluation and learn an implicit generation process through competition between a Generator and a Discriminator. Diffusion models take yet another route by learning how to reverse a gradual corruption process.

The differences among these model families are therefore different solutions to the same fundamental tension between **what a model can represent** and **what it can practically compute**.

---

## 📚 Related Advanced Topics

* [Discriminative vs Generative Models](https://zeromathai.com/en/discriminative-vs-generative-models-course-en/)

---

## ⭐ Note

This GitHub version is adapted from the full ZeroMathAI lecture for repository-friendly reading.

---

## 🔗 Navigation

This is the only lecture in the current set.
